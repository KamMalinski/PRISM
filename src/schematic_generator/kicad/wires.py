from __future__ import annotations

from schematic_generator.kicad.layout import _snap
from schematic_generator.kicad.models import WireSegment
from schematic_generator.kicad.pins import _anchor_wire_pin_element, _element_pin_anchor_segment
from schematic_generator.kicad.sexpr import _junction, _label_net, _orthogonalize_points, _wire
from schematic_generator.models import Element


def _wires_net(
    net: str,
    net_pins: list[tuple[Element, str, float, float]],
    router,
    segments_wires: list[WireSegment] | None = None,
) -> list[str]:
    """Route all pins belonging to a single net."""
    net_pins = sorted(net_pins, key=lambda p: (p[3], p[2]))
    if len(net_pins) == 2:
        return _two_point_net_wires(net, net_pins, router, segments_wires)

    anchors = [_anchor_wire_pin_element(element, pin, x, y) for element, pin, x, y in net_pins]
    xs = [x for x, _ in anchors]
    ys = [y for _, y in anchors]
    hub_x = round(_snap((min(xs) + max(xs)) / 2), 2)
    hub_y = round(_snap(_select_hub_y(ys)), 2)
    hub_x, hub_y = router.nearest_free_point((hub_x, hub_y), net)
    lines: list[str] = []
    stubs = [_element_pin_anchor_segment(element, pin, x, y) for element, pin, x, y in net_pins]
    for stub in stubs:
        router.occupy(stub, net)
        _add_wire(lines, stub, net, segments_wires)
    for anchor in anchors:
        _add_wire(lines, router.route(anchor, (hub_x, hub_y), net), net, segments_wires)
    lines.extend(_junction(hub_x, hub_y))
    lines.extend(_label_net(net, hub_x, hub_y))
    return lines

def _two_point_net_wires(
    net: str,
    net_pins: list[tuple[Element, str, float, float]],
    router,
    segments_wires: list[WireSegment] | None = None,
) -> list[str]:
    """Route a simple two-pin net with pin stubs and one main segment."""
    (element1, pin1, x1, y1), (element2, pin2, x2, y2) = net_pins
    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
    anchor1 = _anchor_wire_pin_element(element1, pin1, x1, y1)
    anchor2 = _anchor_wire_pin_element(element2, pin2, x2, y2)
    stub1 = _element_pin_anchor_segment(element1, pin1, x1, y1)
    stub2 = _element_pin_anchor_segment(element2, pin2, x2, y2)
    router.occupy(stub1, net)
    router.occupy(stub2, net)
    lines: list[str] = []
    _add_wire(lines, stub1, net, segments_wires)
    _add_wire(lines, router.route(anchor1, anchor2, net), net, segments_wires)
    _add_wire(lines, stub2, net, segments_wires)
    lines.extend(_label_net(net, *anchor1))
    return lines

def _add_wire(
    lines: list[str],
    points: list[tuple[float, float]],
    net: str,
    segments_wires: list[WireSegment] | None = None,
) -> None:
    """Append a KiCad wire and optionally collect its diagnostic segments."""
    lines.extend(_wire(points))
    if segments_wires is None:
        return
    segments_wires.extend((net, start, end) for start, end in _segments_from_points(points))

def _segments_from_points(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Convert a point list into rounded orthogonal wire segments."""
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    points = _orthogonalize_points(points)
    for start, end in zip(points, points[1:], strict=False):
        start = (round(start[0], 2), round(start[1], 2))
        end = (round(end[0], 2), round(end[1], 2))
        if start != end:
            segments.append((start, end))
    return segments

def _select_hub_y(ys: list[float]) -> float:
    """Choose a hub y-coordinate for a multi-pin net."""
    min_y = min(ys)
    max_y = max(ys)
    if min_y > 18.0:
        return min_y - 5.08
    return max_y + 5.08
