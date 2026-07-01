from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from schematic_generator.corrections.io import component_pads, component_type, component_value
from schematic_generator.corrections.models import CorrectionState
from schematic_generator.models import Element, HolePair, ManualCorrection, Pad


def add_correction(
    state: CorrectionState,
    type: str,
    description: str,
    data: dict[str, Any] | None = None,
    suggestions: list[dict[str, Any]] | None = None,
) -> ManualCorrection:
    """Append a manual correction entry and return the created model object."""
    correction = ManualCorrection(
        f"K{len(state.corrections) + 1:04d}",
        type,
        description,
        data or {},
        suggestions or [],
        datetime.now().isoformat(timespec="seconds"),
    )
    state.corrections.append(correction)
    return correction


def find_similar_pads(pad: Pad, pads: list[Pad], limit: int = 50) -> list[Pad]:
    """Find same-side pads with similar radius and confidence for deliberate bulk editing."""
    result: list[Pad] = []
    pad_confidence = _pad_confidence(pad)
    for candidate in pads:
        if candidate.node == pad.node or candidate.side != pad.side:
            continue
        radius_reference = max(1.0, pad.radius)
        if abs(candidate.radius - pad.radius) / radius_reference > 0.22:
            continue
        if abs(_pad_confidence(candidate) - pad_confidence) > 0.35:
            continue
        result.append(candidate)
    result.sort(key=lambda candidate: (abs(candidate.radius - pad.radius), abs(_pad_confidence(candidate) - pad_confidence)))
    return result[:limit]


def add_manual_element(
    state: CorrectionState,
    ref: str,
    type: str,
    value: str,
    footprint: str,
    pad_nodes: list[str],
) -> dict[str, Any]:
    """Add a manually defined component to correction state and return its JSON dictionary."""
    component = {
        "ref": ref,
        "type": type,
        "value": value,
        "footprint": footprint,
        "pads": list(pad_nodes),
    }
    state.manual_components.append(component)
    return component


def manual_elements(state: CorrectionState) -> list[Element]:
    """Convert editable manual component dictionaries into Element models for netlist generation."""
    pads_by_node = {pad.node: pad for pad in [*state.top_pads, *state.bottom_pads]}
    canonical_node = canonical_pair_nodes(state.pairs)
    result: list[Element] = []
    for component in state.manual_components:
        pads = _component_model_pads(component, pads_by_node, canonical_node)
        if not pads:
            continue
        x = sum(pad.x for pad in pads) / len(pads)
        y = sum(pad.y for pad in pads) / len(pads)
        dx = pads[-1].x - pads[0].x if len(pads) > 1 else 0.0
        dy = pads[-1].y - pads[0].y if len(pads) > 1 else 0.0
        pins = {str(index): pad.net or "NET?" for index, pad in enumerate(pads, start=1)}
        pin_pad_nodes = {str(index): pad.node for index, pad in enumerate(pads, start=1)}
        pin_descriptions = _pin_descriptions_from_ocr(component)
        element = Element(
            str(component.get("ref", "U?")),
            component_type(component),
            component_value(component) or component_type(component),
            str(component.get("footprint", "")),
            x,
            y,
            math.degrees(math.atan2(dy, dx)) if len(pads) > 1 else 0.0,
            pins,
            pin_descriptions,
            pin_pad_nodes,
            1.0,
            "manual",
            1.0,
            ["manual_component"],
        )
        result.append(element)
    return result


def canonical_pair_nodes(pairs: list[HolePair]) -> dict[str, str]:
    """Map every paired node to its TOP-side canonical node for duplicate suppression."""
    result: dict[str, str] = {}
    for pair in pairs:
        result[pair.pad_top] = pair.pad_top
        result[pair.pad_bottom] = pair.pad_top
    return result


def filter_auto_elements(elements_auto: list[Element], elements_manual: list[Element]) -> list[Element]:
    """Remove automatic components that conflict with manually defined components."""
    if not elements_manual:
        return elements_auto
    manual_refs = {element.ref for element in elements_manual if element.ref}
    manual_sets = [(net_set(element), element.x, element.y) for element in elements_manual]
    result: list[Element] = []
    for element in elements_auto:
        if element.ref in manual_refs or element.type == "TestPoint":
            continue
        nets = net_set(element)
        if nets and any(nets == manual_nets and math.hypot(element.x - x, element.y - y) < 80.0 for manual_nets, x, y in manual_sets):
            continue
        result.append(element)
    return result


def net_set(element: Element) -> frozenset[str]:
    """Return meaningful net names used by an element, excluding placeholder values."""
    return frozenset(net for net in element.pins.values() if net and net != "NET?")


def _component_model_pads(component: dict[str, Any], pads_by_node: dict[str, Pad], canonical_node: dict[str, str]) -> list[Pad]:
    """Resolve component pad node strings into unique, non-mechanical Pad objects."""
    pads: list[Pad] = []
    seen_nodes: set[str] = set()
    for node in component_pads(component):
        canonical = canonical_node.get(node, node)
        if canonical in seen_nodes or canonical not in pads_by_node:
            continue
        pad = pads_by_node[canonical]
        if pad.type == "mounting_hole":
            continue
        pads.append(pad)
        seen_nodes.add(canonical)
    return pads


def _pin_descriptions_from_ocr(component: dict[str, Any]) -> dict[str, str]:
    """Use attached OCR text as a first-pin description when the component has OCR metadata."""
    ocr = dict(component.get("ocr", {}))
    text = str(ocr.get("text", ""))
    return {"1": text} if text else {}


def _pad_confidence(pad: Pad) -> float:
    """Return pad detection confidence."""
    return float(pad.confidence)
