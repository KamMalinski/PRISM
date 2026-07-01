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





def create_elements(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    ocr_texts: list[dict[str, float | int | str]] | None = None,
    side_images: dict[str, np.ndarray] | None = None,
    mask_traces: dict[str, np.ndarray] | None = None,
    samples_colors: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    component_solver_mode: str = "sequential",
    diagnostics_folder: str | Path | None = None,
    log: LogFn | None = None,
) -> list[Element]:
    """Create schematic components from OCR, silkscreen, geometry, and remaining pads."""

    _log(log, "Elements: building the physical pad list without duplicated paired BOTTOM pads.")
    top_pads = [pad for pad in top_pads if pad.type not in {"ignore", "mounting_hole"}]
    bottom_pads = [pad for pad in bottom_pads if pad.type not in {"ignore", "mounting_hole"}]
    paired_bottom = {pair.pad_bottom for pair in pairs}
    pads_physical = list(top_pads)
    pads_physical.extend(pad for pad in bottom_pads if pad.node not in paired_bottom)

    _log(log, "Elements: assigning OCR texts to the nearest pads.")
    pad_texts = _assign_ocr_to_pads(pads_physical, ocr_texts, log)
    if _is_global_solver(component_solver_mode):
        return _create_elements_with_global_solver(
            top_pads,
            bottom_pads,
            pads_physical,
            paired_bottom,
            pad_texts,
            ocr_texts or [],
            side_images or {},
            mask_traces or {},
            samples_colors or {},
            diagnostics_folder,
            log,
        )

    elements: list[Element] = []
    used: set[str] = set()
    pads_by_side = {"TOP": top_pads, "BOTTOM": [pad for pad in bottom_pads if pad.node not in paired_bottom]}

    _log(log, "Elements: grouping pads directly by OCR/refdes labels.")
    groups_ocr_ref = _detect_devices_by_ocr_ref(
        pads_by_side,
        used,
        ocr_texts or [],
        log,
    )
    for group in groups_ocr_ref:
        for pad in group:
            used.add(pad.node)

    _log(log, "Elements: searching for silkscreen markings or symbols that connect pads into devices.")
    groups_silkscreen = _filter_out_suspicious_multi_pin_groups(
        _detect_devices_by_silkscreen(
            pads_by_side,
            used,
            ocr_texts or [],
            side_images or {},
            mask_traces or {},
            samples_colors or {},
            log,
        ),
        "silkscreen",
        log,
    )
    for group in groups_silkscreen:
        for pad in group:
            used.add(pad.node)

    _log(log, "Elements: detecting pad rows with horizontal and vertical grouping.")
    rows = _detect_rows_pads([pad for pad in pads_physical if pad.node not in used], log)
    for index, row in enumerate(rows, start=1):
        for pad in row:
            used.add(pad.node)
        ref = _label_from_ocr(row, ocr_texts) or _label_from_pads(row, pad_texts) or f"J{index}"
        x = sum(p.x for p in row) / len(row)
        y = sum(p.y for p in row) / len(row)
        dx = row[-1].x - row[0].x
        dy = row[-1].y - row[0].y
        rotation = math.degrees(math.atan2(dy, dx))
        pins = {str(i): pad.net or "NET?" for i, pad in enumerate(row, start=1)}
        pin_descriptions = {
            str(i): pad_texts[pad.node]
            for i, pad in enumerate(row, start=1)
            if pad.node in pad_texts
        }
        elements.append(Element(
            ref=ref,
            type="PinRow",
            value=f"{len(row)} pin",
            footprint=f"Connector:PinHeader_1x{len(row)}_P2.54mm",
            x=x,
            y=y,
            rotation=rotation,
            pins=pins,
            pin_descriptions=pin_descriptions,
            pin_pad_nodes=_map_pins_to_pads(row),
            confidence=min(p.confidence for p in row),
            decision_source="row_geometry",
            decision_score=_score_element_decision(row, "row_geometry", ref),
            decision_reasons=_element_decision_reasons(row, "row_geometry", ref),
        ))

    _log(log, "Elements: grouping nearby unconnected pads as discrete or multi-pin devices.")
    groups_nearby = _filter_out_suspicious_multi_pin_groups(
        _detect_nearby_devices(pads_physical, used, log),
        "footprint_geometry",
        log,
    )
    for group in groups_nearby:
        for pad in group:
            used.add(pad.node)

    _log(log, "Elements: searching for two-pin devices from net geometry.")
    groups_two_pin = _detect_two_pin_by_nets(pads_physical, used, log)
    groups_isolated_two_pin: list[list[Pad]] = []
    if not elements and not groups_ocr_ref and not groups_silkscreen and not groups_nearby and not groups_two_pin:
        groups_isolated_two_pin = _detect_isolated_two_pin_footprints(pads_physical, used, log)
    groups_devices = [
        *((group, "OCR/refdes") for group in groups_ocr_ref),
        *((group, "silkscreen") for group in groups_silkscreen),
        *((group, "footprint_geometry") for group in groups_nearby),
        *((group, "net_pattern") for group in groups_two_pin),
        *((group, "footprint_isolated_2pin") for group in groups_isolated_two_pin),
    ]
    type_counters: dict[str, int] = {"R": 1, "C": 1, "D": 1, "L": 1, "U": 1}
    for group, decision_source in groups_devices:
        for pad in group:
            used.add(pad.node)
        ref_ocr = _label_from_ocr(group, ocr_texts) or _label_from_pads(group, pad_texts)
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
        elements.append(Element(
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
            decision_source=decision_source,
            decision_score=_score_element_decision(group, decision_source, ref_ocr),
            decision_reasons=_element_decision_reasons(group, decision_source, ref_ocr),
        ))

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
    skipped_holes = len(remaining) - len(single_trace_pads)
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
            decision_reasons=[
                "single_testpoint_pad",
                f"pad={pad.node}",
                f"net={pad.net or 'NET?'}",
            ],
        ))
    _deduplicate_refs(elements)
    _log(
        log,
        f"Elements: created {len(rows)} devices PinRow, "
        f"{len(groups_ocr_ref)} OCR/refdes devices, "
        f"{len(groups_silkscreen)} silkscreen devices, {len(groups_nearby)} nearby-pad devices, "
        f"{len(groups_two_pin)} two-pin devices from net geometry, "
        f"{len(groups_isolated_two_pin)} fallback two-pin devices from isolated footprints, "
        f"and {len(single_trace_pads)} single testpoints. "
        f"Skipped {skipped_holes} isolated holes/pads without a second point on the same net.",
    )
    return elements
