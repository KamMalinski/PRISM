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



def _pad_groups_touched_by_silkscreen(
    pads: list[Pad],
    mask_silkscreen: np.ndarray,
    log: LogFn | None,
    side: str,
) -> list[list[Pad]]:
    """Find pads connected by the same plausible silkscreen component outline."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask_silkscreen > 0).astype(np.uint8), 8)
    if count <= 1:
        _log(log, f"Elements silkscreen {side}: no silkscreen components found.")
        return []
    pads_by_component: dict[int, list[Pad]] = {}
    for pad in pads:
        for label in _labels_silkscreen_near_pad(labels, stats, pad):
            pads_by_component.setdefault(label, []).append(pad)

    sets = UnionFind()
    by_node = {pad.node: pad for pad in pads}
    connections = 0
    for label, pads_component in pads_by_component.items():
        unique = _unique_pads(pads_component)
        if len(unique) < 2:
            continue
        if not _component_silkscreen_ok(stats, label, unique, mask_silkscreen.shape):
            continue
        for pad in unique:
            sets.add(pad.node)
        for first, second in zip(unique, unique[1:], strict=False):
            if first.net and second.net and first.net == second.net:
                continue
            sets.connect(first.node, second.node)
            connections += 1

    groups: dict[str, list[Pad]] = {}
    for node in sets.parent:
        groups.setdefault(sets.find(node), []).append(by_node[node])
    result = [sorted(group, key=lambda p: (p.y, p.x)) for group in groups.values() if len(group) >= 2]
    _log(log, f"Elements silkscreen {side}: connections={connections}, groups={len(result)}.")
    return result

def _labels_silkscreen_near_pad(labels: np.ndarray, stats: np.ndarray, pad: Pad) -> list[int]:
    """Return label information for labels silkscreen near pad matching."""
    x = int(round(pad.x))
    y = int(round(pad.y))
    radius = int(max(8, round(pad.radius * 3.0)))
    x1, x2 = max(0, x - radius), min(labels.shape[1], x + radius + 1)
    y1, y2 = max(0, y - radius), min(labels.shape[0], y + radius + 1)
    slice_labels = labels[y1:y2, x1:x2]
    if slice_labels.size == 0:
        return []
    values = [int(v) for v in np.unique(slice_labels[slice_labels > 0])]
    return [
        value for value in values
        if stats[value, cv2.CC_STAT_AREA] >= 3
    ]

def _component_silkscreen_ok(
    stats: np.ndarray,
    label: int,
    pads: list[Pad],
    shape: tuple[int, int],
) -> bool:
    """Validate that a silkscreen connected component has a plausible size and placement."""
    area = int(stats[label, cv2.CC_STAT_AREA])
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    if area < 3 or area > shape[0] * shape[1] * 0.03:
        return False
    min_x = min(p.x for p in pads)
    max_x = max(p.x for p in pads)
    min_y = min(p.y for p in pads)
    max_y = max(p.y for p in pads)
    margin = max(18.0, max(p.radius for p in pads) * 4.0)
    if w > max(220.0, (max_x - min_x) + 2.0 * margin):
        return False
    if h > max(160.0, (max_y - min_y) + 2.0 * margin):
        return False
    if x > max_x + margin or x + w < min_x - margin:
        return False
    if y > max_y + margin or y + h < min_y - margin:
        return False
    return True

def _unique_pads(pads: list[Pad]) -> list[Pad]:
    """Return pads without duplicate nodes while preserving their first occurrence."""
    result: dict[str, Pad] = {}
    for pad in pads:
        result[pad.node] = pad
    return sorted(result.values(), key=lambda p: (p.y, p.x))

def _sample_bgr_to_lab(sample: tuple[int, int, int]) -> np.ndarray:
    """Convert one BGR color sample to LAB space for color-distance comparisons."""
    pixel = np.array([[sample]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

def _estimate_background_and_trace_lab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate representative board-background and trace colors from image clusters."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    step = max(1, int(math.sqrt((h * w) / 18000)))
    sample_lab = lab[::step, ::step].reshape(-1, 3).astype(np.float32)
    sample_hsv = hsv[::step, ::step].reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.3)
    _compactness, labels, hsv_centers = cv2.kmeans(sample_hsv, 4, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    lab_centers = np.array([sample_lab[labels == i].mean(axis=0) for i in range(4)], dtype=np.float32)
    counters = np.array([np.count_nonzero(labels == i) for i in range(4)])
    background_index = int(np.argmax(counters))
    remaining = [i for i in range(4) if i != background_index]
    trace_index = max(remaining, key=lambda i: float(np.linalg.norm(lab_centers[i] - lab_centers[background_index])) + float(hsv_centers[i][2]) * 0.2)
    return lab_centers[background_index], lab_centers[trace_index]

def _type_devices_from_ocr(ref: str | None, pin_count: int, pads: list[Pad] | None = None) -> tuple[str, str, str, str]:
    """Infer component type, reference prefix, value, and footprint from OCR and pad geometry."""
    prefix = _prefix_ref(ref or "")
    tht = _looks_like_tht(pads or [])
    if prefix in {"J", "P", "JP", "K", "CON", "CONN"}:
        return "PinRow", "J", f"{pin_count} pin", f"Connector:PinHeader_1x{pin_count}_P2.54mm"
    if pin_count == 2 and prefix in {"R", "C", "D", "L", "F", "Y", "X", "BT"}:
        data = _map_types_two_pin(tht)
        return data[prefix]
    if pin_count == 2 and not prefix:
        footprint = (
            "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"
            if tht
            else "Resistor_SMD:R_0603_1608Metric"
        )
        return "Resistor", "R", "R", footprint
    if pin_count == 3 and prefix == "Q":
        return "Transistor", "Q", "Q", "Transistor_THT:TO-92_Inline"
    return "Device", "U", f"{pin_count} pin", ""

def _map_types_two_pin(tht: bool) -> dict[str, tuple[str, str, str, str]]:
    """Return default two-pin type metadata for THT or SMD-looking footprints."""
    if tht:
        return {
            "R": ("Resistor", "R", "R", "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
            "C": ("Capacitor", "C", "C", "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm"),
            "D": ("Diode", "D", "D", "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal"),
            "L": ("Inductor", "L", "L", "Inductor_THT:L_Axial_L7.0mm_D3.3mm_P12.70mm_Horizontal_Fastron_MICC"),
            "F": ("Fuse", "F", "Fuse", ""),
            "Y": ("Crystal", "Y", "Crystal", ""),
            "X": ("Crystal", "X", "Crystal", ""),
            "BT": ("Battery", "BT", "Battery", ""),
        }
    return {
        "R": ("Resistor", "R", "R", "Resistor_SMD:R_0603_1608Metric"),
        "C": ("Capacitor", "C", "C", "Capacitor_SMD:C_0603_1608Metric"),
        "D": ("Diode", "D", "D", "Diode_SMD:D_0603_1608Metric"),
        "L": ("Inductor", "L", "L", "Inductor_SMD:L_0603_1608Metric"),
        "F": ("Fuse", "F", "Fuse", ""),
        "Y": ("Crystal", "Y", "Crystal", ""),
        "X": ("Crystal", "X", "Crystal", ""),
        "BT": ("Battery", "BT", "Battery", ""),
    }

def _looks_like_tht(pads: list[Pad]) -> bool:
    """Estimate whether pad sizes and spacing look like a through-hole footprint."""
    if not pads:
        return False
    radii = [pad.radius for pad in pads]
    if float(np.median(radii)) >= 6.0:
        return True
    if len(pads) >= 2:
        distances = [
            math.hypot(a.x - b.x, a.y - b.y)
            for i, a in enumerate(pads)
            for b in pads[i + 1:]
        ]
        if distances and max(distances) >= max(45.0, float(np.median(radii)) * 8.0):
            return True
    return False

def _score_element_decision(pads: list[Pad], source: str, ref: str | None = None) -> float:
    """Calculate element decision for ranking component candidates."""
    if not pads:
        return 0.0
    base = min(pad.confidence for pad in pads)
    source_bonus = {
        "OCR/refdes": 0.18,
        "silkscreen": 0.12,
        "row_geometry": 0.10,
        "footprint_geometry": 0.04,
        "footprint_isolated_2pin": -0.02,
        "net_pattern": 0.06,
        "manual": 0.25,
        "fallback": -0.25,
    }.get(source, 0.0)
    nets = {pad.net for pad in pads if pad.net and pad.net != "NET?"}
    net_bonus = 0.06 if len(nets) >= min(2, len(pads)) else -0.08
    ref_bonus = 0.07 if ref else 0.0
    tht_bonus = 0.03 if _looks_like_tht(pads) else 0.0
    return round(max(0.0, min(1.0, base + source_bonus + net_bonus + ref_bonus + tht_bonus)), 3)

def _element_decision_reasons(pads: list[Pad], source: str, ref: str | None = None) -> list[str]:
    """Build compact audit strings that explain why a component group was accepted."""
    reasons = [f"source={source}", f"pin_count={len(pads)}"]
    if pads:
        reasons.append("pin_order=sorted_yx")
        reasons.append("pad_nodes=" + ",".join(pad.node for pad in pads))
    if ref:
        reasons.append(f"ref={ref}")
    if source == "footprint_isolated_2pin":
        reasons.append("fallback=no_other_component_groups")
        reasons.append("risk=isolated_two_pin_footprint_without_refdes")
    nets = sorted({pad.net for pad in pads if pad.net and pad.net != "NET?"})
    reasons.append(f"distinct_nets={len(nets)}")
    if len(pads) >= 2:
        distances = [
            math.hypot(a.x - b.x, a.y - b.y)
            for i, a in enumerate(pads)
            for b in pads[i + 1:]
        ]
        if distances:
            reasons.append(f"max_pad_distance={max(distances):.1f}px")
    if _looks_like_tht(pads):
        reasons.append("footprint_hint=THT")
    if len(nets) < min(2, len(pads)):
        reasons.append("risk=same_or_unknown_net")
    return reasons

def _count_physical_pads_in_nets(pads: list[Pad]) -> dict[str, int]:
    """Count non-ignored physical pads assigned to each electrical net."""
    counter: dict[str, int] = {}
    for pad in pads:
        if pad.net and pad.net != "NET?":
            counter[pad.net] = counter.get(pad.net, 0) + 1
    return counter

def _deduplicate_refs(elements: list[Element]) -> None:
    """Rename repeated reference designators by appending a numeric suffix."""
    seen: dict[str, int] = {}
    for element in elements:
        base = element.ref or element.type[:1] or "U"
        counter = seen.get(base, 0)
        if counter:
            element.ref = f"{base}_{counter + 1}"
        seen[base] = counter + 1

def _clean_ref(text: str) -> str:
    """Strip OCR noise and keep only uppercase reference-designator characters."""
    ref = "".join(ch for ch in text.upper() if ch.isalnum() or ch in "_-")
    return ref[:16] if len(ref) >= 2 else ""
