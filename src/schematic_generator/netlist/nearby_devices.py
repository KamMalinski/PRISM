from __future__ import annotations

import math
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from schematic_generator.contact_solver_facade import is_contact_solver, apply_contact_solver, save_solver_diagnostics
from schematic_generator.connection_graph_facade import active_electrical_edges, build_connection_graph
from schematic_generator.mask_contacts import components_traces, label_traces_near_pad
from schematic_generator.models import Element, Net, Pad, HolePair
from schematic_generator.component_solver import ComponentCandidate, select_candidates_globally

LogFn = Callable[[str], None]



def _detect_nearby_devices(pads: list[Pad], skip: set[str], log: LogFn | None) -> list[list[Pad]]:
    """Detect nearby devices candidates used by component reconstruction."""
    candidates = [pad for pad in pads if pad.node not in skip]
    if len(candidates) < 2:
        return []
    sets = UnionFind()
    for pad in candidates:
        sets.add(pad.node)
    by_node = {pad.node: pad for pad in candidates}
    connections = 0
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a.net and b.net and a.net == b.net:
                continue
            distance = math.hypot(a.x - b.x, a.y - b.y)
            limit = 2.0 * ((a.radius * 2.0 + b.radius * 2.0) / 2.0)
            if distance < limit:
                sets.connect(a.node, b.node)
                connections += 1
    groups: dict[str, list[Pad]] = {}
    for node in sets.parent:
        groups.setdefault(sets.find(node), []).append(by_node[node])
    result = [
        sorted(group, key=lambda p: (p.y, p.x))
        for group in groups.values()
        if len(group) >= 2
    ]
    _log(log, f"Elements: nearby pads: nearby_pairs={connections}, device_groups={len(result)}.")
    return result

def _detect_two_pin_by_nets(pads: list[Pad], skip: set[str], log: LogFn | None) -> list[list[Pad]]:
    """Detect two pin by nets candidates used by component reconstruction."""
    candidates = [
        pad for pad in pads
        if pad.node not in skip and pad.net and pad.net != "NET?"
    ]
    if len(candidates) < 2:
        _log(log, "Two-pin elements: too few free pads.")
        return []

    by_net: dict[str, list[Pad]] = {}
    for pad in pads:
        if pad.net and pad.net != "NET?":
            by_net.setdefault(pad.net, []).append(pad)

    xs = [pad.x for pad in pads]
    ys = [pad.y for pad in pads]
    diagonal = max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    radius_median = float(np.median([pad.radius for pad in candidates]))
    min_distance = max(24.0, radius_median * 2.5)
    max_distance = max(180.0, min(900.0, diagonal * 0.58))

    suggestions: list[tuple[float, float, Pad, Pad]] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            if first.net == second.net:
                continue
            distance = math.hypot(first.x - second.x, first.y - second.y)
            if distance < min_distance or distance > max_distance:
                continue
            if not _similar_radii(first, second):
                continue
            support = _external_net_support(first, second, by_net)
            if support <= 0:
                continue
            large_pad_penalty = (first.radius + second.radius) / max(1.0, radius_median * 2.0)
            result = support - distance / max(1.0, diagonal) - max(0.0, large_pad_penalty - 1.15) * 0.4
            suggestions.append((result, distance, first, second))

    suggestions.sort(key=lambda entry: (-entry[0], entry[1]))
    used: set[str] = set()
    result: list[list[Pad]] = []
    for _score, _distance, first, second in suggestions:
        if first.node in used or second.node in used:
            continue
        used.add(first.node)
        used.add(second.node)
        result.append(sorted([first, second], key=lambda pad: (pad.y, pad.x)))

    _log(
        log,
        f"Two-pin elements: candidates={len(suggestions)}, accepted_groups={len(result)}.",
    )
    return result

def _detect_isolated_two_pin_footprints(
    pads: list[Pad],
    skip: set[str],
    log: LogFn | None,
) -> list[list[Pad]]:
    """Detect isolated two pin footprints candidates used by component reconstruction."""
    candidates = [
        pad for pad in pads
        if (
            pad.node not in skip
            and pad.type not in {"ignore", "mounting_hole", "testpoint"}
            and pad.status == "auto"
            and pad.net
            and pad.net != "NET?"
        )
    ]
    if len(candidates) < 4:
        _log(log, "Isolated two-pin elements: too few automatic pads for a conservative fallback.")
        return []

    radius_median = float(np.median([pad.radius for pad in candidates]))
    if radius_median < 6.0:
        _log(log, "Isolated two-pin elements: pads do not look like THT, fallback skipped.")
        return []

    by_net: dict[str, int] = {}
    for pad in candidates:
        by_net[pad.net] = by_net.get(pad.net, 0) + 1
    singleton_ratio = sum(1 for pad in candidates if by_net.get(pad.net, 0) == 1) / max(1, len(candidates))
    if singleton_ratio < 0.65:
        _log(
            log,
            f"Isolated two-pin elements: nets are not isolated enough "
            f"(singleton_ratio={singleton_ratio:.2f}).",
        )
        return []

    suggestions: list[tuple[float, float, Pad, Pad]] = []
    for side in sorted({pad.side for pad in candidates}):
        side_pads = [pad for pad in candidates if pad.side == side and pad.radius >= radius_median * 0.72]
        if len(side_pads) < 4:
            continue
        diagonal = _pads_diagonal(side_pads)
        min_distance = max(28.0, radius_median * 2.6)
        max_distance = max(70.0, min(240.0, diagonal * 0.36))
        for first in side_pads:
            neighbors: list[tuple[float, Pad]] = []
            for second in side_pads:
                if first.node == second.node or first.net == second.net:
                    continue
                distance = math.hypot(first.x - second.x, first.y - second.y)
                if distance < min_distance or distance > max_distance:
                    continue
                if not _similar_radii(first, second):
                    continue
                neighbors.append((distance, second))
            neighbors.sort(key=lambda item: (item[0], item[1].node))
            for rank, (distance, second) in enumerate(neighbors[:3], start=1):
                pair = sorted([first, second], key=lambda pad: (pad.y, pad.x))
                radius_score = _similarity_radii_groups(pair)
                closeness = max(0.0, 1.0 - distance / max(1.0, max_distance))
                ranking = 0.54 * closeness + 0.34 * radius_score + 0.12 * max(0.0, 4 - rank) / 3.0
                suggestions.append((ranking, distance, pair[0], pair[1]))

    if not suggestions:
        _log(log, "Isolated two-pin elements: missing pairs that satisfy fallback footprint geometry.")
        return []

    suggestions.sort(key=lambda item: (-item[0], item[1], item[2].node, item[3].node))
    ranking, distance, first, second = suggestions[0]
    _log(
        log,
        f"Isolated two-pin elements: accepting one conservative pair {first.node},{second.node} "
        f"(score={ranking:.3f}, distance={distance:.1f}px, singleton_ratio={singleton_ratio:.2f}).",
    )
    return [[first, second]]

def _similar_radii(first: Pad, second: Pad) -> bool:
    """Return whether two pads have similar enough radii to belong to one footprint."""
    larger_radius = max(first.radius, second.radius, 1.0)
    return abs(first.radius - second.radius) / larger_radius <= 0.35

def _external_net_support(first: Pad, second: Pad, by_net: dict[str, list[Pad]]) -> float:
    """Score whether each pad has continuation evidence away from the candidate component."""
    vx = second.x - first.x
    vy = second.y - first.y
    distance = math.hypot(vx, vy)
    if distance <= 1e-6:
        return 0.0
    ux = vx / distance
    uy = vy / distance
    margin_a = max(10.0, first.radius * 1.3)
    margin_b = max(10.0, second.radius * 1.3)
    support_a = _best_directional_support(first, by_net.get(first.net, []), ux, uy, -margin_a)
    support_b = _best_directional_support(second, by_net.get(second.net, []), ux, uy, margin_b)
    if support_a <= 0.0 or support_b <= 0.0:
        return 0.0
    return min(2.0, support_a + support_b)

def _best_directional_support(
    pad: Pad,
    same_net_pads: list[Pad],
    ux: float,
    uy: float,
    threshold: float,
) -> float:
    """Find the strongest same-net pad located in the expected direction from a candidate pin."""
    best_support = 0.0
    for other in same_net_pads:
        if other.node == pad.node:
            continue
        projection = (other.x - pad.x) * ux + (other.y - pad.y) * uy
        if threshold < 0.0 and projection >= threshold:
            continue
        if threshold > 0.0 and projection <= threshold:
            continue
        lateral_distance = abs((other.x - pad.x) * (-uy) + (other.y - pad.y) * ux)
        lateral_limit = max(18.0, (pad.radius + other.radius) * 1.4)
        if lateral_distance > lateral_limit:
            continue
        best_support = max(best_support, min(1.0, abs(projection) / max(1.0, pad.radius * 8.0)))
    return best_support
