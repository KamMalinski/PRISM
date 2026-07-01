from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from schematic_generator.mask_contacts import components_traces, labels_traces_near_pad
from schematic_generator.connection_graph.models import GraphEdge, GraphNode
from schematic_generator.models import Element, Net, Pad, HolePair


def build_connection_graph(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    top_mask: np.ndarray | None = None,
    bottom_mask: np.ndarray | None = None,
    plane_top: np.ndarray | None = None,
    plane_bottom: np.ndarray | None = None,
    nets: list[Net] | None = None,
    elements: list[Element] | None = None,
    expand_contacts_non_green: bool = False,
) -> dict[str, Any]:
    """Build a diagnostic graph that explains pad, trace, plane, net, and component relationships."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    pads = [pad for pad in [*top_pads, *bottom_pads] if pad.type not in {"ignore", "mounting_hole"}]
    pads_by_node = {pad.node: pad for pad in pads}
    counterpart_by_pad = _map_counterparts(pairs)
    pads_from_contact_traces: set[str] = set()

    for pad in pads:
        nodes.append(GraphNode(
            id=pad.node,
            type="pad",
            side=pad.side,
            x=round(pad.x, 2),
            y=round(pad.y, 2),
            bbox=_bbox_pad(pad),
            attrs={
                "radius": round(pad.radius, 2),
                "confidence": round(pad.confidence, 3),
                "net": pad.net,
                "pad_type": pad.type,
                "status": pad.status,
            },
        ))

    for side, pads_side, mask in (
        ("TOP", [p for p in pads if p.side == "TOP"], top_mask),
        ("BOTTOM", [p for p in pads if p.side == "BOTTOM"], bottom_mask),
    ):
        if mask is not None:
            n, e = _graph_mask_traces(
                side,
                pads_side,
                mask,
                {pair.pad_top for pair in pairs} | {pair.pad_bottom for pair in pairs},
                expand_contacts_non_green,
            )
            nodes.extend(n)
            edges.extend(e)
            pads_from_contact_traces.update(
                edge.source
                for edge in e
                if edge.type == "trace_contact" and edge.active
            )

    for side, pads_side, plane in (
        ("TOP", [p for p in pads if p.side == "TOP"], plane_top),
        ("BOTTOM", [p for p in pads if p.side == "BOTTOM"], plane_bottom),
    ):
        if plane is not None:
            n, e = _graph_plane(side, pads_side, plane, pads_from_contact_traces, counterpart_by_pad)
            nodes.extend(n)
            edges.extend(e)

    for pair in pairs:
        top = pads_by_node.get(pair.pad_top)
        bottom = pads_by_node.get(pair.pad_bottom)
        active = bool(top and bottom)
        geometry: dict[str, Any] = {}
        if top and bottom:
            geometry = {
                "points": [[round(top.x, 2), round(top.y, 2)], [round(bottom.x, 2), round(bottom.y, 2)]],
                "length": round(math.hypot(top.x - bottom.x, top.y - bottom.y), 2),
            }
        edges.append(GraphEdge(
            source=pair.pad_top,
            target=pair.pad_bottom,
            type="via_pair",
            source_kind="auto_or_manual",
            confidence=round(pair.confidence, 3),
            active=active,
            reason="TOP/BOTTOM hole pair",
            geometry=geometry,
            attrs={"distance": round(pair.distance, 2)},
        ))

    if nets:
        for net in nets:
            nodes.append(GraphNode(id=f"NET:{net.name}", type="net", attrs={"name": net.name, "pads": len(net.pads)}))
            for node in net.pads:
                if node in pads_by_node:
                    edges.append(GraphEdge(
                        source=node,
                        target=f"NET:{net.name}",
                        type="net_membership",
                        source_kind="derived",
                        confidence=0.75,
                        active=True,
                        reason="connection graph grouping result",
                    ))

    if elements:
        for element in elements:
            nodes.append(GraphNode(
                id=f"EL:{element.ref}",
                type="component",
                x=round(element.x, 2),
                y=round(element.y, 2),
                bbox=_bbox_element(element),
                attrs={
                    "ref": element.ref,
                    "component_type": element.type,
                    "value": element.value,
                    "footprint": element.footprint,
                    "confidence": round(element.confidence, 3),
                    "decision_source": element.decision_source,
                    "decision_score": round(element.decision_score, 3),
                    "decision_reasons": list(element.decision_reasons),
                    "pin_pad_nodes": dict(element.pin_pad_nodes),
                    "pins": len(element.pins),
                },
            ))
            for pin, net in element.pins.items():
                if net and net != "NET?":
                    pad_node = element.pin_pad_nodes.get(pin, "")
                    edges.append(GraphEdge(
                        source=f"EL:{element.ref}",
                        target=f"NET:{net}",
                        type="component_pin",
                        source_kind="auto_or_manual",
                        confidence=round(element.confidence, 3),
                        active=True,
                        reason="component pin assigned to net",
                        attrs={"pin": pin, "pad_node": pad_node},
                    ))

    result = {
        "schema": 2,
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
        "summary": _summary(nodes, edges),
    }
    if nets:
        result["net_explanations"] = _net_explanations(nets, pairs, edges, pads_by_node)
    return result


def active_electrical_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return graph edges that currently participate in electrical connectivity."""

    types = {"trace_contact", "plane_contact", "via_pair", "manual_connection", "mask_bridge"}
    return [
        edge
        for edge in graph.get("edges", [])
        if edge.get("active", True) and edge.get("type") in types
    ]


def refresh_net_explanations(
    graph: dict[str, Any],
    nets: list[Net],
    pairs: list[HolePair],
    top_pads: list[Pad],
    bottom_pads: list[Pad],
) -> dict[str, Any]:
    """Refreshes graph summary and net explanations after opt-in edge decisions."""

    edges = [_edge_from_data(edge) for edge in graph.get("edges", []) or []]
    pads_by_node = {
        pad.node: pad
        for pad in [*top_pads, *bottom_pads]
        if pad.type not in {"ignore", "mounting_hole"}
    }
    graph["summary"] = _summary_data(graph.get("nodes", []) or [], graph.get("edges", []) or [])
    graph["net_explanations"] = _net_explanations(nets, pairs, edges, pads_by_node)
    return graph


def _graph_mask_traces(
    side: str,
    pads: list[Pad],
    mask: np.ndarray,
    paired_nodes: set[str],
    expand_contacts_non_green: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Add trace-component nodes and pad-contact edges for one PCB side."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    mask_without_pads = _remove_pads_from_mask_traces(mask, pads, paired_nodes)
    count, labels, statistics, centroids = components_traces(mask_without_pads)
    contacts_by_label: dict[int, list[tuple[Pad, str]]] = {}
    for pad in pads:
        contacts_pad = _contacts_traces_pad(
            labels,
            statistics,
            pad,
            pad.node in paired_nodes,
            expand_contacts_non_green,
        )
        for label, mode in contacts_pad:
            if label and label < count:
                contacts_by_label.setdefault(label, []).append((pad, mode))

    limit_pads = max(8, int(len(pads) * 0.35))
    active_labels: set[int] = set()
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        node_id = f"{side}:TRACE:{label}"
        bbox = _bbox_stat(statistics, label)
        cx, cy = centroids[label]
        component_ok = _component_traces_ok(statistics, mask.shape, label)
        component_contacts = contacts_by_label.get(label, [])
        component_pads = [pad for pad, _mode in component_contacts]
        active = component_ok and len(component_pads) <= limit_pads
        touched_pads = sorted({pad.node for pad in component_pads})
        risky_large = active and len(touched_pads) >= max(4, int(math.ceil(limit_pads * 0.65)))
        if active:
            active_labels.add(label)
        nodes.append(GraphNode(
            id=node_id,
            type="trace_component",
            side=side,
            x=round(float(cx), 2),
            y=round(float(cy), 2),
            bbox=bbox,
            area=area,
            attrs={
                "pads_touching": len(touched_pads),
                "pads_touching_nodes": touched_pads,
                "pad_activation_limit": limit_pads,
                "potential_pin_pairs": _count_pairs(len(touched_pads)),
                "risky_large_trace_component": risky_large,
                "active_for_netlist": active,
                "rejected_reason": "" if active else _trace_rejection_reason(component_ok, len(component_pads), limit_pads),
            },
        ))
        for pad, mode in component_contacts:
            relaxed = mode == "relaxed_paired_pad"
            edges.append(GraphEdge(
                source=pad.node,
                target=node_id,
                type="trace_contact",
                source_kind="auto",
                confidence=round(_contact_confidence(pad, statistics, label) * (0.82 if relaxed else 1.0), 3),
                active=active,
                reason=(
                    "paired THT pad is close to a trace-mask component"
                    if active and relaxed
                    else "pad touches a trace-mask component"
                    if active
                    else "trace component rejected for netlist connectivity"
                ),
                geometry={
                    "pad": [round(pad.x, 2), round(pad.y, 2)],
                    "component_bbox": bbox,
                    "component_area": area,
                    "sample_pixels": _samples_component(labels, label, pad),
                    "contact_mode": mode,
                },
                attrs={
                    "contact_mode": mode,
                    "trace_component": node_id,
                    "trace_pads_touching": len(touched_pads),
                    "trace_pad_activation_limit": limit_pads,
                    "trace_pads_touching_nodes": touched_pads,
                    "trace_potential_pin_pairs": _count_pairs(len(touched_pads)),
                    "risky_large_trace_component": risky_large,
                },
            ))
    edges.extend(_bridge_nearby_components(side, labels, statistics, active_labels))
    return nodes, edges


def _remove_pads_from_mask_traces(mask: np.ndarray, pads: list[Pad], paired_nodes: set[str]) -> np.ndarray:
    """Remove local pad areas so paired pads do not merge separate traces into one component."""

    result = (mask > 0).astype(np.uint8) * 255
    for pad in pads:
        if pad.node not in paired_nodes or pad.type in {"ignore", "mounting_hole"}:
            continue
        radius = int(max(3, round(pad.radius * 0.92)))
        cv2.circle(result, (int(round(pad.x)), int(round(pad.y))), radius, 0, thickness=-1, lineType=cv2.LINE_AA)
    return result


def _graph_plane(
    side: str,
    pads: list[Pad],
    plane: np.ndarray,
    pads_from_contact_traces: set[str],
    counterpart_by_pad: dict[str, str],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Add copper-plane nodes and pad-contact edges for one PCB side."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    plane_bin = (plane > 0).astype(np.uint8)
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(plane_bin, 8)
    if count <= 1:
        return nodes, edges
    min_area = max(400, int(plane.size * 0.006))
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label]
        nodes.append(GraphNode(
            id=f"{side}:PLANE:{label}",
            type="plane_region",
            side=side,
            x=round(float(cx), 2),
            y=round(float(cy), 2),
            bbox=_bbox_stat(statistics, label),
            area=area,
            attrs={"min_area": min_area},
        ))
    for pad in pads:
        label = _label_plane_under_pad(labels, statistics, pad, min_area)
        if label:
            active = not _physical_pad_has_trace_contact(pad.node, pads_from_contact_traces, counterpart_by_pad)
            edges.append(GraphEdge(
                source=pad.node,
                target=f"{side}:PLANE:{label}",
                type="plane_contact",
                source_kind="auto",
                confidence=0.7,
                active=active,
                reason=(
                    "pad touches a detected copper-plane region"
                    if active
                    else "plane contact inactive: physical pad already has trace contact"
                ),
                geometry={"pad": [round(pad.x, 2), round(pad.y, 2)], "component_bbox": _bbox_stat(statistics, label)},
            ))
    return nodes, edges


def _map_counterparts(pairs: list[HolePair]) -> dict[str, str]:
    """Map each paired TOP/BOTTOM pad node to its opposite-side counterpart."""

    counterparts: dict[str, str] = {}
    for pair in pairs:
        counterparts[pair.pad_top] = pair.pad_bottom
        counterparts[pair.pad_bottom] = pair.pad_top
    return counterparts


def _physical_pad_has_trace_contact(
    node: str,
    pads_from_contact_traces: set[str],
    counterpart_by_pad: dict[str, str],
) -> bool:
    """Check whether a pad or its paired counterpart already has an active trace contact."""

    return node in pads_from_contact_traces or counterpart_by_pad.get(node, "") in pads_from_contact_traces


def _bbox_pad(pad: Pad) -> list[float]:
    """Return a square bounding box around a pad."""

    r = max(1.0, pad.radius)
    return [round(pad.x - r, 2), round(pad.y - r, 2), round(pad.x + r, 2), round(pad.y + r, 2)]


def _bbox_element(element: Element) -> list[float]:
    """Estimate a component bounding box from its center and pin count."""

    pin_count = max(1, len(element.pins))
    w = 12.0 if pin_count <= 2 else 18.0
    h = max(10.0, pin_count * 2.54 + 4.0)
    return [round(element.x - w / 2, 2), round(element.y - h / 2, 2), round(element.x + w / 2, 2), round(element.y + h / 2, 2)]


def _bbox_stat(statistics: np.ndarray, label: int) -> list[float]:
    """Return a bounding box from OpenCV connected-component statistics."""

    x = float(statistics[label, cv2.CC_STAT_LEFT])
    y = float(statistics[label, cv2.CC_STAT_TOP])
    w = float(statistics[label, cv2.CC_STAT_WIDTH])
    h = float(statistics[label, cv2.CC_STAT_HEIGHT])
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def _contacts_traces_pad(
    labels: np.ndarray,
    statistics: np.ndarray,
    pad: Pad,
    paired: bool,
    expand_contacts_non_green: bool = False,
) -> list[tuple[int, str]]:
    """Find trace-component labels touched by a pad."""

    labels_strict = _labels_under_pad(labels, pad)
    contacts = [(label, "strict") for label in labels_strict]
    if not paired or pad.status == "inferred":
        return contacts

    labels_relaxed = _labels_under_pad(labels, pad, relaxed=True, limit=8)
    if not labels_strict:
        return [(label, "relaxed_paired_pad") for label in labels_relaxed]

    seen = set(labels_strict)
    min_trace_area = max(36, int(round(pad.radius * pad.radius * 0.8)))
    for label in labels_relaxed:
        if label in seen or label <= 0 or label >= statistics.shape[0]:
            continue
        area = int(statistics[label, cv2.CC_STAT_AREA])
        has_local_samples = bool(_samples_component(labels, label, pad))
        has_horizontal_approach = expand_contacts_non_green and _horizontal_trace_component_near_pad(statistics, label, pad)
        if area >= min_trace_area and (has_local_samples or has_horizontal_approach):
            contacts.append((label, "relaxed_paired_pad"))
            seen.add(label)
    return contacts


def _horizontal_trace_component_near_pad(statistics: np.ndarray, label: int, pad: Pad) -> bool:
    """Detect relaxed horizontal trace approaches near paired THT pads."""

    x = float(statistics[label, cv2.CC_STAT_LEFT])
    y = float(statistics[label, cv2.CC_STAT_TOP])
    width = float(statistics[label, cv2.CC_STAT_WIDTH])
    height = float(statistics[label, cv2.CC_STAT_HEIGHT])
    if width < max(16.0, height * 2.0):
        return False
    y_margin = max(2.0, pad.radius * 0.55)
    if not (y <= pad.y + y_margin and y + height >= pad.y - y_margin):
        return False
    distance_x = 0.0
    if x > pad.x:
        distance_x = x - (pad.x + pad.radius)
    elif x + width < pad.x:
        distance_x = (pad.x - pad.radius) - (x + width)
    return 0.0 <= distance_x <= max(9.0, pad.radius * 1.85)


def _labels_under_pad(
    labels: np.ndarray,
    pad: Pad,
    relaxed: bool = False,
    limit: int = 4,
) -> list[int]:
    """Return trace labels located under or near a pad."""

    return labels_traces_near_pad(labels, pad, limit=limit, relaxed=relaxed)


def _label_plane_under_pad(labels: np.ndarray, statistics: np.ndarray, pad: Pad, min_area: int) -> int:
    """Return the copper-plane label surrounding a pad, or zero when none is credible."""

    x = int(round(pad.x))
    y = int(round(pad.y))
    radius_inner = int(max(3, round(pad.radius * 0.8)))
    radius_outer = int(max(10, round(pad.radius * 2.4)))
    x1, x2 = max(0, x - radius_outer), min(labels.shape[1], x + radius_outer + 1)
    y1, y2 = max(0, y - radius_outer), min(labels.shape[0], y + radius_outer + 1)
    if x1 >= x2 or y1 >= y2:
        return 0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    ring = (distance_squared >= radius_inner * radius_inner) & (distance_squared <= radius_outer * radius_outer)
    slice_labels = labels[y1:y2, x1:x2]
    values, counters = np.unique(slice_labels[ring & (slice_labels > 0)], return_counts=True)
    if len(values) == 0:
        return 0
    best = int(values[int(np.argmax(counters))])
    if statistics[best, cv2.CC_STAT_AREA] < min_area:
        return 0
    coverage = int(np.max(counters)) / max(1, int(np.count_nonzero(ring)))
    return best if coverage >= 0.08 else 0


def _component_traces_ok(statistics: np.ndarray, shape: tuple[int, int], label: int) -> bool:
    """Reject trace components that are too large to represent local connectivity."""

    height, width = shape
    x = statistics[label, cv2.CC_STAT_LEFT]
    y = statistics[label, cv2.CC_STAT_TOP]
    area = statistics[label, cv2.CC_STAT_AREA]
    if area > height * width * 0.16:
        return False
    return x >= 0 and y >= 0


def _bridge_nearby_components(
    side: str,
    labels: np.ndarray,
    statistics: np.ndarray,
    active_labels: set[int],
) -> list[GraphEdge]:
    """Create bridge edges between nearby active trace components separated by small pixel gaps."""

    if len(active_labels) < 2:
        return []
    max_gap_px = 6
    kernel = np.ones((max_gap_px * 2 + 1, max_gap_px * 2 + 1), np.uint8)
    pairs: set[tuple[int, int]] = set()
    for label in active_labels:
        if int(statistics[label, cv2.CC_STAT_AREA]) < 24:
            continue
        component = (labels == label).astype(np.uint8)
        if int(statistics[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        expanded = cv2.dilate(component, kernel, iterations=1) > 0
        neighbors = np.unique(labels[expanded])
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor <= 0 or neighbor == label or neighbor not in active_labels:
                continue
            if int(statistics[neighbor, cv2.CC_STAT_AREA]) < 24:
                continue
            a, b = sorted((label, neighbor))
            pairs.add((a, b))

    edges: list[GraphEdge] = []
    for a, b in sorted(pairs):
        edges.append(GraphEdge(
            source=f"{side}:TRACE:{a}",
            target=f"{side}:TRACE:{b}",
            type="mask_bridge",
            source_kind="auto",
            confidence=0.45,
            active=True,
            reason="trace-mask components are separated by a small pixel gap",
            geometry={
                "source_bbox": _bbox_stat(statistics, a),
                "target_bbox": _bbox_stat(statistics, b),
                "dilation_px": max_gap_px,
            },
        ))
    return edges


def _net_explanations(
    nets: list[Net],
    pairs: list[HolePair],
    edges: list[GraphEdge],
    pads_by_node: dict[str, Pad],
) -> list[dict[str, Any]]:
    """Build per-net evidence summaries from active graph edges."""

    active_types = {"trace_contact", "plane_contact", "via_pair", "manual_connection", "mask_bridge"}
    neighborhood: dict[str, list[GraphEdge]] = {}
    for edge in edges:
        if not edge.active or edge.type not in active_types:
            continue
        neighborhood.setdefault(edge.source, []).append(edge)
        neighborhood.setdefault(edge.target, []).append(edge)

    physical_counterpart = _map_physical_pads(pairs)
    result: list[dict[str, Any]] = []
    for net in nets:
        visited: set[str] = set()
        queue = [node for node in net.pads if node in pads_by_node]
        net_edges: list[GraphEdge] = []
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for edge in neighborhood.get(node, []):
                net_edges.append(edge)
                second = edge.target if edge.source == node else edge.source
                if second not in visited:
                    queue.append(second)

        type_counters: dict[str, int] = {}
        for edge in net_edges:
            type_counters[edge.type] = type_counters.get(edge.type, 0) + 1
        risky_edges = [
            edge for edge in net_edges
            if bool(edge.attrs.get("risky_large_trace_component"))
        ]
        physical = sorted({physical_counterpart.get(node, node) for node in net.pads})
        positions = [
            {
                "pad": node,
                "side": pads_by_node[node].side,
                "x": round(pads_by_node[node].x, 2),
                "y": round(pads_by_node[node].y, 2),
            }
            for node in net.pads
            if node in pads_by_node
        ]
        result.append({
            "net": net.name,
            "pad_count": len(net.pads),
            "physical_pad_count": len(physical),
            "pads": list(net.pads),
            "physical_pads": physical,
            "active_electrical_edges_by_type": dict(sorted(type_counters.items())),
            "edge_count": len(net_edges),
            "evidence_edges": [_edge_description(edge) for edge in net_edges],
            "risky_large_trace_component_count": len(risky_edges),
            "risky_large_trace_components": _risky_components_traces(risky_edges),
            "positions": positions,
            "diagnostic_flags": _net_diagnostic_flags(net, physical, type_counters, bool(risky_edges)),
        })
    return result


def _map_physical_pads(pairs: list[HolePair]) -> dict[str, str]:
    """Map each paired pad node to a canonical physical pad id."""

    physical_map: dict[str, str] = {}
    for pair in pairs:
        physical_map[pair.pad_top] = pair.pad_top
        physical_map[pair.pad_bottom] = pair.pad_top
    return physical_map


def _net_diagnostic_flags(
    net: Net,
    physical: list[str],
    type_counters: dict[str, int],
    has_risky_component_traces: bool = False,
) -> list[str]:
    """Return diagnostic flags for suspicious net evidence patterns."""

    flags: list[str] = []
    if len(physical) == 1:
        flags.append("single_physical_pad")
    if len(physical) >= 8 and not net.name.upper().startswith("GND"):
        flags.append("large_non_plane_candidate")
    if not type_counters.get("trace_contact"):
        flags.append("no_trace_contact")
    if type_counters.get("plane_contact") and type_counters.get("trace_contact"):
        flags.append("mixed_trace_and_plane_evidence")
    if has_risky_component_traces:
        flags.append("risky_large_trace_component")
    return flags


def _edge_description(edge: GraphEdge) -> dict[str, Any]:
    """Serialize the important diagnostic details of one graph edge."""

    description = {
        "source": edge.source,
        "target": edge.target,
        "type": edge.type,
        "confidence": round(edge.confidence, 3),
        "reason": edge.reason,
    }
    if edge.attrs:
        description["attrs"] = dict(edge.attrs)
    return description


def _risky_components_traces(edges: list[GraphEdge]) -> list[dict[str, Any]]:
    """Group risky trace-contact evidence by trace component id."""

    by_component: dict[str, dict[str, Any]] = {}
    for edge in edges:
        component = str(edge.attrs.get("trace_component", edge.target if ":TRACE:" in edge.target else edge.source))
        entry = by_component.setdefault(component, {
            "trace_component": component,
            "pads_touching": int(edge.attrs.get("trace_pads_touching", 0) or 0),
            "pad_activation_limit": int(edge.attrs.get("trace_pad_activation_limit", 0) or 0),
            "potential_pin_pairs": int(edge.attrs.get("trace_potential_pin_pairs", 0) or 0),
            "pads": sorted(str(pad) for pad in edge.attrs.get("trace_pads_touching_nodes", []) or []),
        })
        if not entry["pads"]:
            entry["pads"] = sorted(str(pad) for pad in edge.attrs.get("trace_pads_touching_nodes", []) or [])
    return sorted(by_component.values(), key=lambda item: item["trace_component"])


def _count_pairs(count: int) -> int:
    """Return the number of unique unordered pairs for a count."""

    return max(0, count * (count - 1) // 2)


def _trace_rejection_reason(component_ok: bool, pad_count: int, limit_pads: int) -> str:
    """Explain why a trace component was excluded from netlist connectivity."""

    if not component_ok:
        return "component is too large or looks like background"
    if pad_count > limit_pads:
        return f"component touches {pad_count} pads, limit {limit_pads}"
    return ""


def _contact_confidence(pad: Pad, statistics: np.ndarray, label: int) -> float:
    """Estimate confidence for a pad-to-trace-component contact."""

    area = max(1, int(statistics[label, cv2.CC_STAT_AREA]))
    radius_factor = min(1.0, max(0.25, pad.radius / 8.0))
    area_factor = min(1.0, math.log10(area + 10.0) / 4.0)
    return round(0.35 + 0.45 * radius_factor + 0.2 * area_factor, 3)


def _samples_component(labels: np.ndarray, label: int, pad: Pad) -> list[list[int]]:
    """Sample a few nearby pixels that belong to a trace component."""

    x = int(round(pad.x))
    y = int(round(pad.y))
    radius = int(max(8, round(pad.radius * 2.0)))
    x1, x2 = max(0, x - radius), min(labels.shape[1], x + radius + 1)
    y1, y2 = max(0, y - radius), min(labels.shape[0], y + radius + 1)
    slice_mask = labels[y1:y2, x1:x2] == label
    points = np.argwhere(slice_mask)
    if len(points) == 0:
        return []
    step = max(1, len(points) // 8)
    return [[int(x1 + px), int(y1 + py)] for py, px in points[::step][:8]]


def _summary(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict[str, Any]:
    """Summarize graph nodes and edges by type."""

    node_counter = {}
    edge_counter = {}
    active_edge_counter = {}
    for node in nodes:
        node_counter[node.type] = node_counter.get(node.type, 0) + 1
    for edge in edges:
        edge_counter[edge.type] = edge_counter.get(edge.type, 0) + 1
        if edge.active and edge.type in {"trace_contact", "plane_contact", "via_pair", "manual_connection", "mask_bridge"}:
            active_edge_counter[edge.type] = active_edge_counter.get(edge.type, 0) + 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "active_electrical_edges": sum(1 for e in edges if e.active and e.type in {"trace_contact", "plane_contact", "via_pair", "manual_connection", "mask_bridge"}),
        "nodes_by_type": node_counter,
        "edges_by_type": edge_counter,
        "active_electrical_edges_by_type": dict(sorted(active_edge_counter.items())),
    }


def _summary_data(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize serialized graph nodes and edges by type."""

    node_counter: dict[str, int] = {}
    edge_counter: dict[str, int] = {}
    active_edge_counter: dict[str, int] = {}
    for node in nodes:
        type = str(node.get("type", ""))
        node_counter[type] = node_counter.get(type, 0) + 1
    for edge in edges:
        type = str(edge.get("type", ""))
        edge_counter[type] = edge_counter.get(type, 0) + 1
        if bool(edge.get("active", True)) and type in {"trace_contact", "plane_contact", "via_pair", "manual_connection", "mask_bridge"}:
            active_edge_counter[type] = active_edge_counter.get(type, 0) + 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "active_electrical_edges": sum(active_edge_counter.values()),
        "nodes_by_type": node_counter,
        "edges_by_type": edge_counter,
        "active_electrical_edges_by_type": dict(sorted(active_edge_counter.items())),
    }


def _edge_from_data(edge: dict[str, Any]) -> GraphEdge:
    """Create a GraphEdge from serialized graph data."""

    return GraphEdge(
        source=str(edge.get("source", "")),
        target=str(edge.get("target", "")),
        type=str(edge.get("type", "")),
        source_kind=str(edge.get("source_kind", "")),
        confidence=float(edge.get("confidence", 0.0) or 0.0),
        active=bool(edge.get("active", True)),
        reason=str(edge.get("reason", "")),
        geometry=dict(edge.get("geometry", {}) or {}),
        attrs=dict(edge.get("attrs", {}) or {}),
    )
