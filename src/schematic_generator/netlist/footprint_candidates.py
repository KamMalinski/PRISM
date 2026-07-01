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



def _candidates_pairs_two_pin_global(
    pads: list[Pad],
    log: LogFn | None,
) -> list[tuple[list[Pad], float, list[str], dict[str, Any]]]:
    """Propose two-pin candidates whose external net geometry supports a component between them."""
    candidates = [pad for pad in pads if pad.net and pad.net != "NET?"]
    if len(candidates) < 2:
        return []
    by_net: dict[str, list[Pad]] = {}
    for pad in pads:
        if pad.net and pad.net != "NET?":
            by_net.setdefault(pad.net, []).append(pad)

    xs = [pad.x for pad in pads]
    ys = [pad.y for pad in pads]
    diagonal = max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    radius_median = _radius_median(candidates)
    min_distance = max(24.0, radius_median * 2.5)
    max_distance = max(180.0, min(900.0, diagonal * 0.58))
    suggestions: list[tuple[float, float, list[Pad], float, list[str], dict[str, Any]]] = []
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
            closeness = max(0.0, 1.0 - distance / max(1.0, max_distance))
            if support <= 0.0:
                continue
            score_hint = min(0.18, support * 0.08 + closeness * 0.06)
            ranking = score_hint + support * 0.08 + closeness * 0.05
            evidence = [f"net_support={support:.3f}", f"distance={distance:.1f}px"]
            features = {"net_support": round(support, 3), "pair_distance": round(distance, 2)}
            suggestions.append((ranking, distance, sorted([first, second], key=lambda pad: (pad.y, pad.x)), score_hint, evidence, features))

    suggestions.sort(key=lambda item: (-item[0], item[1], item[2][0].node, item[2][1].node))
    result = [(group, score_hint, evidence, features) for _ranking, _distance, group, score_hint, evidence, features in suggestions[:320]]
    _log(log, f"Elements global_solver: two-pin net-pattern candidates={len(result)}.")
    return result

def _candidates_footprint_two_pin_global(
    pads: list[Pad],
    log: LogFn | None,
) -> list[tuple[list[Pad], float, list[str], dict[str, Any]]]:
    """Propose nearest similar two-pad footprints for the global component solver."""
    candidates = [pad for pad in pads if pad.type not in {"ignore", "mounting_hole"}]
    if len(candidates) < 2:
        return []
    radius_median = _radius_median(candidates)
    diagonal = _pads_diagonal(candidates)
    min_distance = max(16.0, radius_median * 2.2)
    max_distance = max(55.0, min(240.0, diagonal * 0.32))
    suggestions: dict[tuple[str, str], tuple[float, float, list[Pad], float, list[str], dict[str, Any]]] = {}

    for first in candidates:
        neighbors = []
        for second in candidates:
            if first.node == second.node:
                continue
            if first.side != second.side:
                continue
            distance = math.hypot(first.x - second.x, first.y - second.y)
            if distance < min_distance or distance > max_distance:
                continue
            if not _similar_radii(first, second):
                continue
            neighbors.append((distance, second))
        neighbors.sort(key=lambda item: (item[0], item[1].node))
        for rank, (distance, second) in enumerate(neighbors[:5], start=1):
            pair = sorted([first, second], key=lambda pad: (pad.y, pad.x))
            key = tuple(sorted(pad.node for pad in pair))
            radius_score = _similarity_radii_groups(pair)
            net_diversity = len({pad.net for pad in pair if pad.net and pad.net != "NET?"})
            closeness = max(0.0, 1.0 - distance / max(1.0, max_distance))
            same_net_penalty = 0.18 if net_diversity < 2 else 0.0
            ranking = 0.46 * closeness + 0.34 * radius_score + 0.08 * max(0.0, 6 - rank) / 5.0 - same_net_penalty
            if ranking <= 0.18:
                continue
            score_hint = min(0.16, max(0.03, ranking * 0.11))
            evidence = [
                "footprint_template=2pin_nearest_similar_pads",
                f"distance={distance:.1f}px",
                f"neighbor_rank={rank}",
            ]
            features = {
                "template_support": round(ranking, 3),
                "pair_distance": round(distance, 2),
                "neighbor_rank": rank,
                "pair_kind": "two_pin_nearest_similar",
            }
            previous = suggestions.get(key)
            entry = (ranking, distance, pair, score_hint, evidence, features)
            if previous is None or (ranking, -distance) > (previous[0], -previous[1]):
                suggestions[key] = entry

    ordered_candidates = sorted(suggestions.values(), key=lambda item: (-item[0], item[1], item[2][0].node, item[2][1].node))
    result = [(group, score_hint, evidence, features) for _ranking, _distance, group, score_hint, evidence, features in ordered_candidates[:420]]
    _log(log, f"Elements global_solver: candidates footprint_template_2pin={len(result)}.")
    return result

def _global_three_pin_footprint_candidates(
    pads: list[Pad],
    log: LogFn | None,
) -> list[tuple[list[Pad], float, list[str], dict[str, Any]]]:
    """Propose compact three-pad footprints such as small transistors or regulators."""
    candidates = [pad for pad in pads if pad.type not in {"ignore", "mounting_hole"}]
    if len(candidates) < 3:
        return []
    radius_median = _radius_median(candidates)
    diagonal = _pads_diagonal(candidates)
    max_span = max(45.0, min(210.0, diagonal * 0.24))
    min_spacing = max(10.0, radius_median * 1.8)
    suggestions: dict[tuple[str, str, str], tuple[float, float, list[Pad], float, list[str], dict[str, Any]]] = {}

    for first in candidates:
        neighbors = [
            (math.hypot(first.x - second.x, first.y - second.y), second)
            for second in candidates
            if second.node != first.node
            and second.side == first.side
            and _similar_radii(first, second)
        ]
        neighbors = [(distance, pad) for distance, pad in neighbors if min_spacing <= distance <= max_span]
        neighbors.sort(key=lambda item: (item[0], item[1].node))
        nearest_pads = [pad for _distance, pad in neighbors[:7]]
        for i, second in enumerate(nearest_pads):
            for third in nearest_pads[i + 1:]:
                group = sorted([first, second, third], key=lambda pad: (pad.y, pad.x))
                key = tuple(sorted(pad.node for pad in group))
                if key in suggestions:
                    continue
                distances = [
                    math.hypot(a.x - b.x, a.y - b.y)
                    for index, a in enumerate(group)
                    for b in group[index + 1:]
                ]
                span = max(distances)
                if span > max_span or min(distances) < min_spacing:
                    continue
                radius_score = _similarity_radii_groups(group)
                geometry = _candidate_geometry_metrics(group)
                net_diversity = len({pad.net for pad in group if pad.net and pad.net != "NET?"})
                compactness = max(0.0, 1.0 - span / max(1.0, max_span))
                net_score = min(1.0, net_diversity / 3.0)
                ranking = 0.34 * compactness + 0.28 * radius_score + 0.22 * float(geometry["geometry_score"]) + 0.16 * net_score
                if ranking <= 0.24:
                    continue
                score_hint = min(0.16, max(0.03, ranking * 0.10))
                evidence = [
                    "footprint_template=3pin_compact_cluster",
                    f"span={span:.1f}px",
                    f"net_diversity={net_diversity}",
                ]
                features = {
                    "template_support": round(ranking, 3),
                    "cluster_span": round(span, 2),
                    "pair_kind": "three_pin_compact_cluster",
                    "net_diversity": net_diversity,
                }
                suggestions[key] = (ranking, span, group, score_hint, evidence, features)

    ordered_candidates = sorted(suggestions.values(), key=lambda item: (-item[0], item[1], item[2][0].node))
    result = [(group, score_hint, evidence, features) for _ranking, _span, group, score_hint, evidence, features in ordered_candidates[:240]]
    _log(log, f"Elements global_solver: candidates footprint_template_3pin={len(result)}.")
    return result

def _pads_diagonal(pads: list[Pad]) -> float:
    """Return the diagonal of the bounding box containing the supplied pads."""
    if not pads:
        return 1.0
    xs = [pad.x for pad in pads]
    ys = [pad.y for pad in pads]
    return max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))

def _element_from_candidate(
    candidate: ComponentCandidate,
    pads_by_node: dict[str, Pad],
    pad_texts: dict[str, str],
    ocr_texts: list[dict[str, float | int | str]],
    type_counters: dict[str, int],
) -> Element:
    """Convert the selected solver candidate into an Element with pins and decision metadata."""
    group = [pads_by_node[node] for node in candidate.pin_order]
    ref_ocr = candidate.proposed_ref or _label_from_ocr(group, ocr_texts) or _label_from_pads(group, pad_texts)
    type, prefix, value, footprint = _type_devices_from_ocr(ref_ocr, len(group), group)
    type_counters.setdefault(prefix, 1)
    ref = ref_ocr if ref_ocr and ref_ocr.startswith(prefix) else f"{prefix}{type_counters[prefix]}"
    type_counters[prefix] += 1
    x = sum(p.x for p in group) / len(group)
    y = sum(p.y for p in group) / len(group)
    dx = group[-1].x - group[0].x
    dy = group[-1].y - group[0].y
    pins = {str(i): pad.net or "NET?" for i, pad in enumerate(group, start=1)}
    pin_descriptions = {
        str(i): pad_texts[pad.node]
        for i, pad in enumerate(group, start=1)
        if pad.node in pad_texts
    }
    reasons = [
        "solver=global_set_packing",
        f"candidate_id={candidate.identifier}",
        f"candidate_source={candidate.source}",
        f"candidate_weight={candidate.weight:.3f}",
        *candidate.evidence,
    ]
    reasons.extend(f"risk={risk}" for risk in candidate.risks)
    return Element(
        ref=ref,
        type=type,
        value=value,
        footprint=footprint,
        x=x,
        y=y,
        rotation=math.degrees(math.atan2(dy, dx)),
        pins=pins,
        pin_descriptions=pin_descriptions,
        pin_pad_nodes=_map_pins_to_pads(group),
        confidence=min(p.confidence for p in group),
        decision_source="global_solver",
        decision_score=round(max(0.0, min(1.0, candidate.score)), 3),
        decision_reasons=reasons,
    )
