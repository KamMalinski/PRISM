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



def _add_testpoints_for_remaining(
    elements: list[Element],
    pads_physical: list[Pad],
    pad_texts: dict[str, str],
    used: set[str],
) -> None:
    """Create fallback testpoint elements for unused pads that still have meaningful net context."""
    pads_per_net_count = _count_physical_pads_in_nets(pads_physical)
    start_tp = len(elements) + 1
    remaining = [p for p in sorted(pads_physical, key=lambda p: (p.y, p.x)) if p.node not in used]
    single_trace_pads = [
        pad for pad in remaining
        if (
            pad.type == "testpoint"
            and pad.net
            and pad.net != "NET?"
            and pads_per_net_count.get(pad.net, 0) > 1
        )
    ]
    for index, pad in enumerate(single_trace_pads, start=start_tp):
        pin_description = pad_texts.get(pad.node, "")
        elements.append(Element(
            ref=f"TP{index}",
            type="TestPoint",
            value=pin_description or pad.net or "NET?",
            footprint="TestPoint:TestPoint_Pad_D1.0mm",
            x=pad.x,
            y=pad.y,
            rotation=0.0,
            pins={"1": pad.net or "NET?"},
            pin_descriptions={"1": pin_description} if pin_description else {},
            pin_pad_nodes={"1": pad.node},
            confidence=pad.confidence,
            decision_source="fallback",
            decision_score=round(min(1.0, pad.confidence * 0.55), 3),
            decision_reasons=["single_testpoint_pad", f"pad={pad.node}", f"net={pad.net or 'NET?'}"],
        ))

def _save_global_solver_diagnostics(
    diagnostics_folder: str | Path | None,
    candidates: list[ComponentCandidate],
    solver_result,
) -> None:
    """Save global solver diagnostics diagnostics or output data."""
    if not diagnostics_folder:
        return
    folder = Path(diagnostics_folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "component_candidates.json").write_text(
        json.dumps([candidate.to_json() for candidate in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (folder / "component_solver.json").write_text(
        json.dumps(solver_result.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _radius_median(pads: list[Pad]) -> float:
    """Return the median pad radius, or zero when no pads are available."""
    if not pads:
        return 1.0
    return float(np.median([pad.radius for pad in pads]))

def _detect_rows_pads(pads: list[Pad], log: LogFn | None) -> list[list[Pad]]:
    """Detect rows pads candidates used by component reconstruction."""
    if len(pads) < 4:
        return []
    radius_median = float(np.median([p.radius for p in pads]))
    line_tolerance = max(8.0, radius_median * 2.2)
    pitch_tolerance = max(10.0, radius_median * 2.8)
    _log(
        log,
        f"Elements: pad row parameters: line_tolerance={line_tolerance:.1f}px, "
        f"pitch_tolerance={pitch_tolerance:.1f}px, min_pinow=4.",
    )
    candidates: list[list[Pad]] = []
    candidates.extend(_rows_on_axis(pads, "x", line_tolerance, pitch_tolerance))
    candidates.extend(_rows_on_axis(pads, "y", line_tolerance, pitch_tolerance))
    candidates.sort(key=lambda r: (len(r), _length_row(r)), reverse=True)

    result: list[list[Pad]] = []
    used: set[str] = set()
    for row in candidates:
        clean_row = [pad for pad in row if pad.node not in used]
        if len(clean_row) < 4:
            continue
        for pad in clean_row:
            used.add(pad.node)
        result.append(clean_row)
    _log(log, f"Elements: detected {len(result)} non-overlapping pad rows.")
    return result

def _rows_on_axis(pads: list[Pad], direction: str, line_tolerance: float, pitch_tolerance: float) -> list[list[Pad]]:
    """Scan pads along one axis and return regular rows that pass spacing checks."""
    if direction == "x":
        sorted_pads = sorted(pads, key=lambda p: (p.y, p.x))
        os_linii = lambda p: p.y
        os_row = lambda p: p.x
    else:
        sorted_pads = sorted(pads, key=lambda p: (p.x, p.y))
        os_linii = lambda p: p.x
        os_row = lambda p: p.y

    groups: list[list[Pad]] = []
    for pad in sorted_pads:
        added = False
        for group in groups:
            if abs(os_linii(pad) - np.mean([os_linii(p) for p in group])) <= line_tolerance:
                group.append(pad)
                added = True
                break
        if not added:
            groups.append([pad])

    rows: list[list[Pad]] = []
    for group in groups:
        group = sorted(group, key=os_row)
        if len(group) < 4:
            continue
        typical = max(pitch_tolerance, _typical_raster(group))
        biezacy = [group[0]]
        for previous, pad in zip(group, group[1:], strict=False):
            distance = abs(os_row(pad) - os_row(previous))
            if distance <= max(pitch_tolerance * 4.0, typical * 1.8):
                biezacy.append(pad)
            else:
                if _row_is_regular(biezacy):
                    rows.append(biezacy)
                biezacy = [pad]
        if _row_is_regular(biezacy):
            rows.append(biezacy)
    return rows

def _typical_raster(group: list[Pad]) -> float:
    """Estimate the typical spacing between neighboring pads in a candidate row."""
    if len(group) < 2:
        return 0.0
    distances = [math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(group, group[1:], strict=False)]
    return float(np.median(distances))

def _row_is_regular(row: list[Pad]) -> bool:
    """Check whether a row has reasonably consistent spacing between adjacent pads."""
    if len(row) < 4:
        return False
    distances = [math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(row, row[1:], strict=False)]
    median = float(np.median(distances))
    if median <= 0:
        return False
    return float(np.max(np.abs(np.array(distances) - median))) <= max(12.0, median * 0.45)

def _length_row(row: list[Pad]) -> float:
    """Return the geometric length between the first and last pad in a row."""
    return math.hypot(row[-1].x - row[0].x, row[-1].y - row[0].y)

def _label_from_ocr(row: list[Pad], ocr_texts: list[dict[str, float | int | str]] | None) -> str | None:
    """Return label information for from ocr matching."""
    if not ocr_texts:
        return None
    x = sum(p.x for p in row) / len(row)
    y = sum(p.y for p in row) / len(row)
    best = None
    best_distance = math.inf
    for entry in ocr_texts:
        tx = int(entry["x"]) + int(entry["w"]) / 2.0
        ty = int(entry["y"]) + int(entry["h"]) / 2.0
        distance = math.hypot(tx - x, ty - y)
        if distance < best_distance:
            best = str(entry["text"])
            best_distance = distance
    if best and best_distance <= max(80.0, _length_row(row) * 0.6):
        return _clean_ref(best)
    return None

def _assign_ocr_to_pads(
    pads: list[Pad],
    ocr_texts: list[dict[str, float | int | str]] | None,
    log: LogFn | None,
) -> dict[str, str]:
    """Attach nearby OCR labels to pads when the text is close enough to be useful."""
    assigned: dict[str, str] = {}
    if not ocr_texts or not pads:
        _log(log, "Elements OCR: missing texts or pads to assign.")
        return assigned
    for entry in ocr_texts:
        text = _clean_ref(str(entry.get("text", "")))
        if not text:
            continue
        tx = float(entry["x"]) + float(entry["w"]) / 2.0
        ty = float(entry["y"]) + float(entry["h"]) / 2.0
        pad = min(pads, key=lambda p: math.hypot(p.x - tx, p.y - ty))
        distance = math.hypot(pad.x - tx, pad.y - ty)
        limit = max(45.0, pad.radius * 10.0)
        if distance <= limit and pad.node not in assigned:
            assigned[pad.node] = text
            _log(log, f"Elements OCR: assigned text '{text}' to {pad.node}, distance={distance:.1f}px.")
    _log(log, f"Elements OCR: assigned {len(assigned)} texts to nearest pads.")
    return assigned

def _label_from_pads(row: list[Pad], pad_texts: dict[str, str]) -> str | None:
    """Return the first cleaned label already attached to any pad in the row."""
    for pad in row:
        text = pad_texts.get(pad.node, "")
        if text:
            return text
    for pad in row:
        text = _clean_ref(pad.name)
        if text:
            return text
    return None

def _filter_out_suspicious_multi_pin_groups(
    groups: list[list[Pad]],
    source: str,
    log: LogFn | None,
) -> list[list[Pad]]:
    """Drop multi-pin groups that likely describe shared trace geometry rather than components."""
    result: list[list[Pad]] = []
    rejected_count = 0
    for group in groups:
        nets = {pad.net for pad in group if pad.net and pad.net != "NET?"}
        # Without a refdes/OCR label, a multi-pin group with only one or two nets often
        # means thin silkscreen or an outline connected several independent elements.
        if len(group) >= 4 and len(nets) <= 2:
            rejected_count += 1
            _log(
                log,
                f"Elements {source}: rejecting suspicious group of {len(group)} pads "
                f"with {len(nets)} nets ({', '.join(pad.node for pad in group)}).",
            )
            continue
        result.append(group)
    if rejected_count:
        _log(log, f"Elements {source}: rejected {rejected_count} suspicious multi-pin groups.")
    return result
