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



def _detect_devices_by_silkscreen(
    pads_by_side: dict[str, list[Pad]],
    skip: set[str],
    ocr_texts: list[dict[str, float | int | str]],
    side_images: dict[str, np.ndarray],
    mask_traces: dict[str, np.ndarray],
    samples_colors: dict[str, dict[str, tuple[int, int, int]]],
    log: LogFn | None,
) -> list[list[Pad]]:
    """Detect devices by silkscreen candidates used by component reconstruction."""
    groups: list[list[Pad]] = []
    for side, pads in pads_by_side.items():
        image = side_images.get(side)
        side_trace_mask = mask_traces.get(side)
        if image is None or side_trace_mask is None:
            _log(log, f"Elements silkscreen {side}: missing image or trace mask, skipping.")
            continue
        candidates = [pad for pad in pads if pad.node not in skip]
        if len(candidates) < 2:
            continue
        mask_silkscreen = _mask_silkscreen(image, side_trace_mask, samples_colors.get(side, {}), log, side)
        groups_side = _pad_groups_touched_by_silkscreen(candidates, mask_silkscreen, log, side)
        occupied = {pad.node for group in groups_side for pad in group}
        groups_ocr = _groups_pads_near_ocr(
            [pad for pad in candidates if pad.node not in occupied],
            ocr_texts,
            log,
            side,
        )
        groups.extend(_without_overlapping_groups([*groups_side, *groups_ocr]))
    _log(log, f"Elements silkscreen: found {len(groups)} pad groups connected by thin markings or symbols.")
    return groups

def _detect_devices_by_ocr_ref(
    pads_by_side: dict[str, list[Pad]],
    skip: set[str],
    ocr_texts: list[dict[str, float | int | str]],
    log: LogFn | None,
) -> list[list[Pad]]:
    """Detect devices by ocr ref candidates used by component reconstruction."""
    groups: list[list[Pad]] = []
    for side, pads in pads_by_side.items():
        candidates = [pad for pad in pads if pad.node not in skip]
        groups.extend(_groups_pads_near_ocr(candidates, ocr_texts, log, side))
    result = _without_overlapping_groups(groups)
    _log(log, f"Elements OCR/refdes: found {len(result)} groups before silkscreen analysis.")
    return result

def _groups_pads_near_ocr(
    pads: list[Pad],
    ocr_texts: list[dict[str, float | int | str]],
    log: LogFn | None,
    side: str,
) -> list[list[Pad]]:
    """Group pads near recognized reference labels when the expected pin count is known."""
    texts = [
        entry for entry in ocr_texts
        if str(entry.get("side", "")).upper() == side.upper()
    ]
    if len(pads) < 2 or not texts:
        return []

    radius_median = float(np.median([pad.radius for pad in pads]))
    xs = [pad.x for pad in pads]
    ys = [pad.y for pad in pads]
    diagonal = max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    groups: list[list[Pad]] = []
    used: set[str] = set()
    for entry in sorted(texts, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True):
        ref = _clean_ref(str(entry.get("text", "")))
        pin_count = _expected_pin_count_from_ref(ref)
        if not ref or pin_count is None:
            continue
        available = [pad for pad in pads if pad.node not in used]
        if len(available) < pin_count:
            continue
        tx = float(entry.get("x", 0.0)) + float(entry.get("w", 0.0)) / 2.0
        ty = float(entry.get("y", 0.0)) + float(entry.get("h", 0.0)) / 2.0
        limit = min(
            max(125.0, radius_median * 30.0, max(float(entry.get("w", 0.0)), float(entry.get("h", 0.0))) * 6.0),
            diagonal * 0.48,
        )
        group = _select_pads_for_ocr_ref(available, tx, ty, pin_count, limit)
        if len(group) != pin_count:
            continue
        for pad in group:
            used.add(pad.node)
            pad.name = ref
        _log(
            log,
            f"Elements OCR+silkscreen {side}: '{ref}' grouping {len(group)} pads "
            f"({', '.join(pad.node for pad in group)}), limit={limit:.1f}px.",
        )
        groups.append(group)
    _log(log, f"Elements OCR+silkscreen {side}: accepted {len(groups)} groups.")
    return groups

def _expected_pin_count_from_ref(ref: str) -> int | None:
    """Infer a likely pin count from the normalized reference designator prefix."""
    prefix = _prefix_ref(ref)
    if prefix in {"R", "C", "D", "L", "F", "Y", "X", "BT"}:
        return 2
    if prefix == "Q":
        return 3
    return None

def _prefix_ref(ref: str) -> str:
    """Extract the alphabetic reference prefix used for component type inference."""
    result = []
    for char in ref.upper():
        if char.isalpha():
            result.append(char)
        else:
            break
    prefix = "".join(result)
    if prefix.startswith("LED"):
        return "D"
    return prefix

def _select_pads_for_ocr_ref(pads: list[Pad], tx: float, ty: float, pin_count: int, limit: float) -> list[Pad]:
    """Select pads for ocr ref from candidate data."""
    if pin_count == 2:
        return _select_pad_pair_for_ocr(pads, tx, ty, limit)
    candidates = [
        (math.hypot(pad.x - tx, pad.y - ty), pad)
        for pad in pads
        if math.hypot(pad.x - tx, pad.y - ty) <= limit
    ]
    candidates.sort(key=lambda item: item[0])
    group = [pad for _distance, pad in candidates[:pin_count]]
    if len(group) == pin_count and len({pad.net for pad in group if pad.net}) >= 2:
        return sorted(group, key=lambda pad: (pad.y, pad.x))
    return []

def _select_pad_pair_for_ocr(pads: list[Pad], tx: float, ty: float, limit: float) -> list[Pad]:
    """Select pad pair for ocr from candidate data."""
    best: tuple[float, Pad, Pad] | None = None
    for index, first in enumerate(pads):
        d1 = math.hypot(first.x - tx, first.y - ty)
        if d1 > limit:
            continue
        for second in pads[index + 1:]:
            if first.net and second.net and first.net == second.net:
                continue
            d2 = math.hypot(second.x - tx, second.y - ty)
            if d2 > limit:
                continue
            pad_distance = math.hypot(first.x - second.x, first.y - second.y)
            if pad_distance < max(18.0, (first.radius + second.radius) * 2.2):
                continue
            sx = (first.x + second.x) / 2.0
            sy = (first.y + second.y) / 2.0
            distance_center = math.hypot(sx - tx, sy - ty)
            if distance_center > max(80.0, pad_distance * 0.58):
                continue
            imbalance_penalty = abs(d1 - d2) / max(1.0, pad_distance)
            result = distance_center + (d1 + d2) * 0.12 + imbalance_penalty * 25.0
            if best is None or result < best[0]:
                best = (result, first, second)
    if best is None:
        return []
    return sorted([best[1], best[2]], key=lambda pad: (pad.y, pad.x))

def _without_overlapping_groups(groups: list[list[Pad]]) -> list[list[Pad]]:
    """Keep only non-overlapping pad groups, preferring larger groups first."""
    result: list[list[Pad]] = []
    used: set[str] = set()
    for group in sorted(groups, key=lambda item: (-len(item), min((pad.y, pad.x) for pad in item))):
        clean_group = [pad for pad in group if pad.node not in used]
        if len(clean_group) < 2:
            continue
        result.append(clean_group)
        for pad in clean_group:
            used.add(pad.node)
    return result

def _mask_silkscreen(
    image: np.ndarray,
    mask_traces: np.ndarray,
    samples: dict[str, tuple[int, int, int]],
    log: LogFn | None,
    side: str,
) -> np.ndarray:
    """Build a silkscreen mask by separating bright non-trace pixels from the board image."""
    _log(log, f"Elements silkscreen {side}: building a non-background, non-trace color mask.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    background_bgr = samples.get("background_bgr")
    path_bgr = samples.get("trace_bgr")
    if background_bgr and path_bgr:
        background_lab = _sample_bgr_to_lab(background_bgr)
        path_lab = _sample_bgr_to_lab(path_bgr)
    else:
        background_lab, path_lab = _estimate_background_and_trace_lab(image)
        _log(log, f"Elements silkscreen {side}: missing complete samples, estimating colors from the image.")

    background_distance = np.linalg.norm(lab.astype(np.float32) - background_lab, axis=2)
    trace_distance = np.linalg.norm(lab.astype(np.float32) - path_lab, axis=2)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bright_silkscreen = (v >= np.percentile(v, 72)) & (s <= np.percentile(s, 58))
    other_color = (background_distance > 18.0) & (trace_distance > 16.0)
    mask = np.where(bright_silkscreen & other_color, 255, 0).astype(np.uint8)

    traces_expanded = cv2.dilate((mask_traces > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    mask[traces_expanded > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask
