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



def _normalize_ref_ocr(text: str) -> str:
    """Normalize common OCR mistakes in reference designators before grouping pads."""
    ref = _clean_ref(text)
    # O/0 at the beginning of a reference designator is unusual on PCB labels;
    # OCR often reads it instead of D, so only rewrite a simple prefix+number form.
    if len(ref) >= 2 and ref[0] in {"O", "0"} and ref[1:].isdigit():
        return f"D{ref[1:]}"
    return ref

def _connect_pads_from_mask(
    sets: UnionFind,
    pads: list[Pad],
    mask: np.ndarray,
    side: str,
) -> None:
    """Connect pads using pads from mask evidence."""
    count, labels, stats, _ = components_traces(mask)
    pads_by_label: dict[int, list[Pad]] = {}
    for pad in pads:
        label = _label_under_pad(labels, pad)
        if label and label < count and _component_traces_ok(stats, mask.shape, label):
            pads_by_label.setdefault(label, []).append(pad)

    limit_pads = max(8, int(len(pads) * 0.2))
    for label, pads_component in pads_by_label.items():
        if len(pads_component) > limit_pads:
            continue
        for pad in pads_component:
            sets.connect(pad.node, f"{side}:SCIEZKA:{label}")

def _connect_pads_from_plane(
    sets: UnionFind,
    pads: list[Pad],
    plane: np.ndarray,
    side: str,
) -> None:
    """Connect pads using pads from plane evidence."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((plane > 0).astype(np.uint8), 8)
    if count <= 1:
        return
    min_area = max(400, int(plane.size * 0.006))
    for pad in pads:
        label = _label_plane_under_pad(labels, stats, pad, min_area)
        if label:
            sets.connect(pad.node, f"{side}:PLANE:{label}")

def _label_plane_under_pad(labels: np.ndarray, stats: np.ndarray, pad: Pad, min_area: int) -> int:
    """Return label information for plane under pad matching."""
    x = int(round(pad.x))
    y = int(round(pad.y))
    radius_inner = int(max(3, round(pad.radius * 0.8)))
    radius_outer = int(max(10, round(pad.radius * 2.4)))
    x1, x2 = max(0, x - radius_outer), min(labels.shape[1], x + radius_outer + 1)
    y1, y2 = max(0, y - radius_outer), min(labels.shape[0], y + radius_outer + 1)
    if x1 >= x2 or y1 >= y2:
        return 0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    # Check for plane contact around the pad. With clearance, the ring stays empty.
    ring = (distance_sq >= radius_inner * radius_inner) & (distance_sq <= radius_outer * radius_outer)
    slice_labels = labels[y1:y2, x1:x2]
    values, counters = np.unique(slice_labels[ring & (slice_labels > 0)], return_counts=True)
    if len(values) == 0:
        return 0
    best = int(values[int(np.argmax(counters))])
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return 0
    coverage = int(np.max(counters)) / max(1, int(np.count_nonzero(ring)))
    return best if coverage >= 0.08 else 0

def _component_traces_ok(stats: np.ndarray, shape: tuple[int, int], label: int) -> bool:
    """Reject trace components that look too large or touch image boundaries."""
    height, width = shape
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    area = stats[label, cv2.CC_STAT_AREA]
    if area > height * width * 0.16:
        return False
    return x >= 0 and y >= 0

def _label_under_pad(labels: np.ndarray, pad: Pad) -> int:
    """Return label information for under pad matching."""
    return label_traces_near_pad(labels, pad)

def _save_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text to a path, creating parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _log(log: LogFn | None, text: str) -> None:
    """Forward a message to the optional logger callback."""
    if log:
        log(text)
