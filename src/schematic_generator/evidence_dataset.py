from __future__ import annotations

import math
from collections import Counter
from typing import Any

from schematic_generator.models import Element, Net, Pad, HolePair


SCHEMA_VERSION = 1

CORRECTION_TYPES = [
    "false_pad",
    "missing_pad",
    "wrong_pair",
    "missing_trace_contact",
    "false_trace_contact",
    "wrong_component_group",
    "wrong_pin_order",
]


def build_evidence_dataset(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets: list[Net],
    elements: list[Element],
    graph: dict[str, Any],
    manual_corrections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Builds a stable diagnostic evidence dataset without using ground truth."""

    # The dataset is intentionally derived from reconstruction evidence only.
    # It can be inspected, compared, or used for later training data curation
    # without requiring a golden schematic for the board.
    pads = [pad for pad in [*top_pads, *bottom_pads] if pad.type not in {"ignore", "mounting_hole"}]
    pads_by_node = {pad.node: pad for pad in pads}
    pair_by_node = _pair_index(pairs)
    net_size_by_name = {net.name: len(net.pads) for net in nets}
    graph_edges = list(graph.get("edges", []) or [])
    graph_nodes = list(graph.get("nodes", []) or [])

    # Candidate sections all share the same rough shape: identity, source,
    # geometry, related graph nodes, feature values, and decision reasons.
    # Keeping that shape consistent makes the JSON easier to diff and consume.
    pad_candidates = [
        _pad_candidate(pad, pair_by_node, net_size_by_name)
        for pad in sorted(pads, key=lambda item: (item.side, item.y, item.x, item.identifier))
    ]
    via_pair_candidates = [_via_pair_candidate(pair) for pair in pairs]
    trace_contacts = [
        _edge_candidate(edge, "trace_contact")
        for edge in graph_edges
        if edge.get("type") == "trace_contact"
    ]
    plane_contacts = [
        _edge_candidate(edge, "plane_contact")
        for edge in graph_edges
        if edge.get("type") == "plane_contact"
    ]
    mask_bridges = [
        _edge_candidate(edge, "mask_bridge")
        for edge in graph_edges
        if edge.get("type") == "mask_bridge"
    ]
    component_groups = [
        _component_group(element, pads_by_node)
        for element in sorted(elements, key=lambda item: item.ref)
    ]
    pin_assignments = [
        assignment
        for element in sorted(elements, key=lambda item: item.ref)
        for assignment in _pin_assignments(element)
    ]
    correction_examples = _correction_examples(manual_corrections or [])

    # diagnostic_layers are summary views over the detailed candidate lists.
    # They make quick quality checks possible without scanning every item.
    return {
        "schema": "schematic_generator.evidence_dataset",
        "schema_version": SCHEMA_VERSION,
        "object_classes": {
            "pad_candidates": len(pad_candidates),
            "via_pair_candidates": len(via_pair_candidates),
            "trace_contact_candidates": len(trace_contacts),
            "plane_contact_candidates": len(plane_contacts),
            "mask_bridge_candidates": len(mask_bridges),
            "component_groups": len(component_groups),
            "pin_assignment_candidates": len(pin_assignments),
            "correction_examples": len(correction_examples["items"]),
        },
        "pad_candidates": pad_candidates,
        "via_pair_candidates": via_pair_candidates,
        "trace_contact_candidates": trace_contacts,
        "plane_contact_candidates": plane_contacts,
        "mask_bridge_candidates": mask_bridges,
        "component_groups": component_groups,
        "pin_assignment_candidates": pin_assignments,
        "diagnostic_layers": _diagnostic_layers(
            pad_candidates,
            via_pair_candidates,
            trace_contacts,
            plane_contacts,
            mask_bridges,
            component_groups,
            graph_nodes,
            graph.get("net_explanations", []) or [],
        ),
        "correction_examples": correction_examples,
    }


def _pad_candidate(pad: Pad, pair_by_node: dict[str, HolePair], net_size_by_name: dict[str, int]) -> dict[str, Any]:
    """Serialize one detected pad as a candidate with geometry, pairing, net-size, and risk evidence."""
    pair = pair_by_node.get(pad.node)
    reasons = [f"status={pad.status}", f"type={pad.type}"]
    if pair:
        reasons.append("paired_top_bottom")
    if pad.net:
        reasons.append(f"net={pad.net}")
    if pad.status == "inferred":
        reasons.append("risk=inferred_pad")
    if not pair and pad.type in {"pad", "via"}:
        reasons.append("risk=unpaired_pad_candidate")
    if pad.net and net_size_by_name.get(pad.net, 0) <= 1:
        reasons.append("risk=single_pad_net")
    return {
        "id": pad.node,
        "source": pad.status or "auto",
        "side": pad.side,
        "confidence": round(float(pad.confidence), 3),
        "bbox": _bbox_pad(pad),
        "points": [[round(float(pad.x), 2), round(float(pad.y), 2)]],
        "related_graph_nodes": [pad.node],
        "features": {
            "radius": round(float(pad.radius), 2),
            "pad_type": pad.type,
            "status": pad.status,
            "net": pad.net,
            "net_size": int(net_size_by_name.get(pad.net, 0) or 0),
            "paired": bool(pair),
            "pair_confidence": round(float(pair.confidence), 3) if pair else 0.0,
        },
        "reason": reasons,
    }


def _via_pair_candidate(pair: HolePair) -> dict[str, Any]:
    """Serialize one TOP/BOTTOM pad pair as evidence for a through-hole or via relationship."""
    return {
        "id": f"{pair.pad_top}<->{pair.pad_bottom}",
        "source": "top_bottom_pairing",
        "side": "both",
        "confidence": round(float(pair.confidence), 3),
        "bbox": [],
        "points": [],
        "related_graph_nodes": [pair.pad_top, pair.pad_bottom],
        "features": {
            "distance": round(float(pair.distance), 2),
            "pad_top": pair.pad_top,
            "pad_bottom": pair.pad_bottom,
        },
        "reason": ["paired_top_bottom"],
    }


def _edge_candidate(edge: dict[str, Any], expected_type: str) -> dict[str, Any]:
    """Serialize a graph edge that represents trace, plane, or mask-bridge contact evidence."""
    geometry = edge.get("geometry", {}) or {}
    attrs = edge.get("attrs", {}) or {}
    related = [str(edge.get("source", "")), str(edge.get("target", ""))]
    reasons = [str(edge.get("reason", ""))]
    if not bool(edge.get("active", True)):
        reasons.append("inactive_for_netlist")
    if attrs.get("risky_large_trace_component"):
        reasons.append("risk=risky_large_trace_component")
    contact_mode = str(geometry.get("contact_mode", attrs.get("contact_mode", "")) or "")
    if contact_mode:
        reasons.append(f"contact_mode={contact_mode}")
    return {
        "id": f"{expected_type}:{edge.get('source', '')}->{edge.get('target', '')}",
        "source": str(edge.get("source_kind", "auto")),
        "side": _side_from_nodes(related),
        "confidence": round(float(edge.get("confidence", 0.0) or 0.0), 3),
        "active": bool(edge.get("active", True)),
        "bbox": geometry.get("component_bbox") or geometry.get("source_bbox") or [],
        "points": _edge_points(geometry),
        "related_graph_nodes": related,
        "features": {
            "edge_type": expected_type,
            "contact_mode": contact_mode,
            "trace_component": attrs.get("trace_component", ""),
            "trace_pads_touching": int(attrs.get("trace_pads_touching", 0) or 0),
            "trace_pad_activation_limit": int(attrs.get("trace_pad_activation_limit", 0) or 0),
            "component_area": int(geometry.get("component_area", 0) or 0),
            "dilation_px": int(geometry.get("dilation_px", 0) or 0),
        },
        "reason": [reason for reason in reasons if reason],
    }


def _component_group(element: Element, pads_by_node: dict[str, Pad]) -> dict[str, Any]:
    """Serialize one reconstructed component with its pads, pins, source, score, and grouping risks."""
    pad_nodes = list(element.pin_pad_nodes.values())
    pads = [pads_by_node[node] for node in pad_nodes if node in pads_by_node]
    risks = [reason for reason in element.decision_reasons if reason.startswith("risk=")]
    return {
        "id": f"component:{element.ref}",
        "source": element.decision_source,
        "side": _dominant_side(pads),
        "confidence": round(float(element.confidence), 3),
        "bbox": _bbox_pads(pads) if pads else _bbox_element(element),
        "points": [[round(float(element.x), 2), round(float(element.y), 2)]],
        "related_graph_nodes": [f"EL:{element.ref}", *pad_nodes],
        "features": {
            "ref": element.ref,
            "component_type": element.type,
            "value": element.value,
            "footprint": element.footprint,
            "pin_count": len(element.pins),
            "pin_pad_nodes": dict(element.pin_pad_nodes),
            "pins": dict(element.pins),
            "decision_score": round(float(element.decision_score), 3),
            "risk_count": len(risks),
        },
        "reason": list(element.decision_reasons),
    }


def _pin_assignments(element: Element) -> list[dict[str, Any]]:
    """Serialize each component pin-to-pad assignment as an independently inspectable candidate."""
    assignments = []
    for pin, pad_node in sorted(element.pin_pad_nodes.items(), key=lambda item: _natural_pin_key(item[0])):
        assignments.append({
            "id": f"pin_assignment:{element.ref}.{pin}",
            "source": element.decision_source,
            "side": _side_from_nodes([pad_node]),
            "confidence": round(float(element.confidence), 3),
            "bbox": [],
            "points": [],
            "related_graph_nodes": [f"EL:{element.ref}", pad_node],
            "features": {
                "ref": element.ref,
                "pin": str(pin),
                "pad_node": pad_node,
                "net": element.pins.get(pin, ""),
                "decision_score": round(float(element.decision_score), 3),
            },
            "reason": [f"source={element.decision_source}", "pin_order=component_order"],
        })
    return assignments


def _diagnostic_layers(
    pad_candidates: list[dict[str, Any]],
    via_pairs: list[dict[str, Any]],
    trace_contacts: list[dict[str, Any]],
    plane_contacts: list[dict[str, Any]],
    mask_bridges: list[dict[str, Any]],
    component_groups: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
    net_explanations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate candidate lists into compact counters for quick quality review and regression checks."""
    pad_risks = Counter(
        reason
        for item in pad_candidates
        for reason in item.get("reason", [])
        if str(reason).startswith("risk=")
    )
    component_sources = Counter(item.get("source", "") for item in component_groups)
    component_risks = Counter(
        reason
        for item in component_groups
        for reason in item.get("reason", [])
        if str(reason).startswith("risk=")
    )
    trace_components = [node for node in graph_nodes if node.get("type") == "trace_component"]
    rejected_trace_components = [
        node for node in trace_components
        if not bool((node.get("attrs", {}) or {}).get("active_for_netlist", True))
    ]
    net_flags = Counter(
        flag
        for item in net_explanations
        for flag in item.get("diagnostic_flags", []) or []
    )
    return {
        "pad_detection": {
            "candidate_count": len(pad_candidates),
            "inferred_count": sum(1 for item in pad_candidates if item.get("features", {}).get("status") == "inferred"),
            "unpaired_count": sum(1 for item in pad_candidates if not item.get("features", {}).get("paired")),
            "single_pad_net_count": int(pad_risks.get("risk=single_pad_net", 0)),
            "risk_counts": dict(sorted(pad_risks.items())),
        },
        "pair_matching": {
            "candidate_count": len(via_pairs),
            "low_confidence_count": sum(1 for item in via_pairs if float(item.get("confidence", 0.0) or 0.0) < 0.5),
        },
        "trace_contact_evidence": {
            "trace_contact_count": len(trace_contacts),
            "active_trace_contact_count": sum(1 for item in trace_contacts if item.get("active", True)),
            "inactive_trace_contact_count": sum(1 for item in trace_contacts if not item.get("active", True)),
            "plane_contact_count": len(plane_contacts),
            "active_plane_contact_count": sum(1 for item in plane_contacts if item.get("active", True)),
            "mask_bridge_count": len(mask_bridges),
            "active_mask_bridge_count": sum(1 for item in mask_bridges if item.get("active", True)),
            "trace_component_count": len(trace_components),
            "rejected_trace_component_count": len(rejected_trace_components),
            "net_diagnostic_flags": dict(sorted(net_flags.items())),
        },
        "component_grouping": {
            "component_count": len(component_groups),
            "decision_sources": dict(sorted(component_sources.items())),
            "risk_counts": dict(sorted(component_risks.items())),
            "pin_assignment_count": sum(int(item.get("features", {}).get("pin_count", 0) or 0) for item in component_groups),
        },
    }


def _correction_examples(manual_corrections: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy manual correction history into a label-like section that can seed later review or training workflows."""
    items = []
    for index, correction in enumerate(manual_corrections, start=1):
        category = str(correction.get("dataset_label", "") or correction.get("type", "") or "manual_correction")
        items.append({
            "id": str(correction.get("identifier", f"manual:{index:04d}")),
            "category": category,
            "source": "manual_correction",
            "payload": correction,
        })
    counts = Counter(item["category"] for item in items)
    return {
        "supported_categories": list(CORRECTION_TYPES),
        "items": items,
        "counts": {key: int(counts.get(key, 0)) for key in CORRECTION_TYPES},
    }


def _pair_index(pairs: list[HolePair]) -> dict[str, HolePair]:
    """Build a lookup that maps each pad node in a pair to the pair object."""
    result: dict[str, HolePair] = {}
    for pair in pairs:
        result[pair.pad_top] = pair
        result[pair.pad_bottom] = pair
    return result


def _bbox_pad(pad: Pad) -> list[float]:
    """Return a pad bounding box in image coordinates using the pad radius."""
    r = max(1.0, float(pad.radius))
    return [round(float(pad.x) - r, 2), round(float(pad.y) - r, 2), round(float(pad.x) + r, 2), round(float(pad.y) + r, 2)]


def _bbox_pads(pads: list[Pad]) -> list[float]:
    """Return the union bounding box for a group of pads."""
    boxes = [_bbox_pad(pad) for pad in pads]
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def _bbox_element(element: Element) -> list[float]:
    """Estimate a component bounding box when no pad geometry is available."""
    pin_count = max(1, len(element.pins))
    w = 12.0 if pin_count <= 2 else 18.0
    h = max(10.0, pin_count * 2.54 + 4.0)
    return [
        round(float(element.x) - w / 2, 2),
        round(float(element.y) - h / 2, 2),
        round(float(element.x) + w / 2, 2),
        round(float(element.y) + h / 2, 2),
    ]


def _edge_points(geometry: dict[str, Any]) -> list[list[float]]:
    """Extract representative points from edge geometry for visualization and review tools."""
    points = geometry.get("points")
    if isinstance(points, list):
        return points
    pad = geometry.get("pad")
    if isinstance(pad, list):
        return [pad]
    samples = geometry.get("sample_pixels")
    if isinstance(samples, list):
        return samples[:8]
    return []


def _side_from_nodes(nodes: list[str]) -> str:
    """Infer whether related graph nodes belong to TOP, BOTTOM, both sides, or no board side."""
    sides = {str(node).split(":", 1)[0] for node in nodes if ":" in str(node)}
    sides = {side for side in sides if side in {"TOP", "BOTTOM"}}
    if len(sides) == 1:
        return next(iter(sides))
    if len(sides) > 1:
        return "both"
    return ""


def _dominant_side(pads: list[Pad]) -> str:
    """Infer the dominant board side for a group of pads."""
    sides = Counter(pad.side for pad in pads)
    if not sides:
        return ""
    if len(sides) == 1:
        return next(iter(sides))
    return "both"


def _natural_pin_key(value: str) -> tuple[int, str]:
    """Sort numeric pin names numerically while keeping nonnumeric pin names stable."""
    text = str(value)
    return (int(text), text) if text.isdigit() else (math.inf, text)
