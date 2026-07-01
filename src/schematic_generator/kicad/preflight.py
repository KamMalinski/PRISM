from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from schematic_generator.kicad.layout import _overlap
from schematic_generator.kicad.models import PinRef, SymbolLayout, WireSegment
from schematic_generator.kicad.pins import _element_pin_point


def _preflight_wires(
    net_groups: dict[str, list[PinRef]],
    layout: dict[str, SymbolLayout],
    dense_nets: set[str],
    segments: list[WireSegment],
) -> dict[str, object]:
    """Compare intended net groups with routed wires and report connectivity risks."""
    expected_by_net = _preflight_expected_groups(net_groups, layout, dense_nets)
    conflicts = _preflight_segment_conflicts(segments)
    conflicts.extend(_preflight_conflicts_pins(net_groups, layout, dense_nets, segments))
    actual_groups = _preflight_actual_groups(expected_by_net, conflicts)
    expected_groups = sorted(sorted(pins) for pins in expected_by_net.values() if len(pins) > 1)
    expected_set = {tuple(pins) for pins in expected_groups}
    actual_set = {tuple(pins) for pins in actual_groups}
    missing = sorted([list(group) for group in expected_set - actual_set])
    extra = sorted([list(group) for group in actual_set - expected_set])
    return {
        "status": "pass" if not conflicts and not missing and not extra else "fail",
        "expected_groups": expected_groups,
        "actual_groups": actual_groups,
        "missing_groups": missing,
        "extra_groups": extra,
        "wire_conflict_count": len(conflicts),
        "wire_conflicts": conflicts,
    }

def _preflight_expected_groups(
    net_groups: dict[str, list[PinRef]],
    layout: dict[str, SymbolLayout],
    dense_nets: set[str],
) -> dict[str, list[str]]:
    """Build expected connected pin groups for explicitly routed nets."""
    expected: dict[str, list[str]] = {}
    for net, pins in net_groups.items():
        if net in dense_nets:
            continue
        refs = [f"{element.ref}:{pin}" for element, pin in pins if element.ref in layout]
        if len(refs) > 1:
            expected[net] = sorted(refs)
    return expected

def _preflight_actual_groups(
    expected_by_net: dict[str, list[str]],
    conflicts: list[dict[str, object]],
) -> list[list[str]]:
    """Merge expected groups that are accidentally connected by detected wire conflicts."""
    parent = {net: net for net in expected_by_net}

    def find(net: str) -> str:
        """Find the current union-find representative for a net."""
        while parent[net] != net:
            parent[net] = parent[parent[net]]
            net = parent[net]
        return net

    def union(a: str, b: str) -> None:
        """Merge two union-find groups when a conflict connects their nets."""
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for conflict in conflicts:
        net_a = str(conflict.get("net_a", ""))
        net_b = str(conflict.get("net_b", ""))
        union(net_a, net_b)

    groups: dict[str, list[str]] = {}
    for net, pins in expected_by_net.items():
        groups.setdefault(find(net), []).extend(pins)
    return sorted(sorted(pins) for pins in groups.values() if len(pins) > 1)

def _preflight_segment_conflicts(segments: list[WireSegment]) -> list[dict[str, object]]:
    """Detect crossings and overlaps between routed wire segments from different nets."""
    conflicts: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for index, (net_a, a1, a2) in enumerate(segments):
        for net_b, b1, b2 in segments[index + 1:]:
            if net_a == net_b:
                continue
            conflict = _segments_conflict(a1, a2, b1, b2)
            if conflict is None:
                continue
            dedupe_key = (
                tuple(sorted((net_a, net_b))),
                tuple(sorted((a1, a2))),
                tuple(sorted((b1, b2))),
                _hashable_key(conflict.get("point", [])),
                _hashable_key(conflict.get("overlap", [])),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            conflicts.append({
                "type": conflict["type"],
                "net_a": net_a,
                "net_b": net_b,
                "segment_a": [[a1[0], a1[1]], [a2[0], a2[1]]],
                "segment_b": [[b1[0], b1[1]], [b2[0], b2[1]]],
                **{k: v for k, v in conflict.items() if k != "type"},
            })
    return conflicts

def _hashable_key(value: object) -> object:
    """Convert nested diagnostic values into a stable deduplication key."""
    if isinstance(value, list):
        return tuple(_hashable_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable_key(value)) for key, value in value.items()))
    return value

def _preflight_conflicts_pins(
    net_groups: dict[str, list[PinRef]],
    layout: dict[str, SymbolLayout],
    dense_nets: set[str],
    segments: list[WireSegment],
) -> list[dict[str, object]]:
    """Detect foreign wire segments that pass through pins from other nets."""
    conflicts: list[dict[str, object]] = []
    for pin_net, pins in net_groups.items():
        if pin_net in dense_nets:
            continue
        for element, pin in pins:
            if element.ref not in layout:
                continue
            symbol = layout[element.ref]
            point = _element_pin_point(element, pin, symbol.x, symbol.y)
            point = (round(point[0], 2), round(point[1], 2))
            for wire_net, start, end in segments:
                if wire_net == pin_net:
                    continue
                if _point_on_segment(point, start, end):
                    conflicts.append({
                        "type": "foreign_wire_on_pin",
                        "net_a": pin_net,
                        "net_b": wire_net,
                        "pin": f"{element.ref}:{pin}",
                        "point": [point[0], point[1]],
                        "segment_b": [[start[0], start[1]], [end[0], end[1]]],
                    })
    return conflicts

def _segments_conflict(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> dict[str, object] | None:
    """Classify whether two orthogonal wire segments cross or overlap."""
    a_is_vertical = abs(a1[0] - a2[0]) < 1e-6
    b_is_vertical = abs(b1[0] - b2[0]) < 1e-6
    if a_is_vertical and b_is_vertical:
        if abs(a1[0] - b1[0]) > 1e-6:
            return None
        overlap = _overlap_range(a1[1], a2[1], b1[1], b2[1])
        if overlap is None:
            return None
        return {"type": "wire_overlap", "overlap": [[round(a1[0], 2), overlap[0]], [round(a1[0], 2), overlap[1]]]}
    if not a_is_vertical and not b_is_vertical:
        if abs(a1[1] - b1[1]) > 1e-6:
            return None
        overlap = _overlap_range(a1[0], a2[0], b1[0], b2[0])
        if overlap is None:
            return None
        return {"type": "wire_overlap", "overlap": [[overlap[0], round(a1[1], 2)], [overlap[1], round(a1[1], 2)]]}
    vertical_segment = (a1, a2) if a_is_vertical else (b1, b2)
    horizontal_segment = (b1, b2) if a_is_vertical else (a1, a2)
    point = (round(vertical_segment[0][0], 2), round(horizontal_segment[0][1], 2))
    if _point_on_segment(point, *vertical_segment) and _point_on_segment(point, *horizontal_segment):
        return {"type": "wire_crossing", "point": [point[0], point[1]]}
    return None

def _overlap_range(a1: float, a2: float, b1: float, b2: float) -> tuple[float, float] | None:
    """Return the shared one-dimensional range between two collinear segments."""
    lo = round(max(min(a1, a2), min(b1, b2)), 2)
    hi = round(min(max(a1, a2), max(b1, b2)), 2)
    if lo > hi:
        return None
    return lo, hi

def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Check whether a point lies on an orthogonal segment."""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    if abs(x1 - x2) < 1e-6:
        return abs(px - x1) < 1e-6 and min(y1, y2) - 1e-6 <= py <= max(y1, y2) + 1e-6
    if abs(y1 - y2) < 1e-6:
        return abs(py - y1) < 1e-6 and min(x1, x2) - 1e-6 <= px <= max(x1, x2) + 1e-6
    return False

def _save_report_layout(
    schematic_path: Path,
    layout: dict[str, SymbolLayout],
    net_groups: dict[str, list[PinRef]],
    dense_nets: set[str],
    router = None,
    preflight: dict[str, object] | None = None,
) -> None:
    """Persist layout, routing, and preflight diagnostics next to the schematic."""
    collisions = _layout_collisions(layout)
    report = {
        "symbol_count": len(layout),
        "symbol_collisions": len(collisions),
        "collisions": collisions,
        "routing_conflict_count": len(router.conflicts) if router else 0,
        "routing_conflicts": router.conflicts if router else [],
        "wire_preflight": preflight or {},
        "label_nets": sorted(dense_nets),
        "wired_nets": sorted(net for net in net_groups if net not in dense_nets),
        "symbols": [asdict(symbol) | {"bbox": [round(v, 2) for v in symbol.bbox]} for symbol in layout.values()],
    }
    schematic_path.with_suffix(".layout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _layout_collisions(layout: dict[str, SymbolLayout]) -> list[dict[str, str]]:
    """List symbol pairs that still overlap after layout placement."""
    symbols = list(layout.values())
    result: list[dict[str, str]] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            if _overlap(a, b, margin=1.0):
                result.append({"a": a.ref, "b": b.ref})
    return result
