from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from schematic_generator.kicad.layout import _build_layout, _group_pins_by_net, _select_nets_labeled, _should_skip_schematic_element, _snap
from schematic_generator.kicad.models import SymbolLayout, TWO_PIN_TYPES
from schematic_generator.kicad.pins import _element_pin_anchor_segment, _element_pin_point, _anchor_wire_pin_element, _reserve_pin_stubs
from schematic_generator.kicad.routing import _Router
from schematic_generator.kicad.sexpr import natural_pin_sort_key
from schematic_generator.kicad.wires import _select_hub_y
from schematic_generator.models import Element


def save_schematic_preview_png(path: str | Path, elements: list[Element]) -> None:
    """Render a lightweight PNG preview from the same layout logic used by KiCad export."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = [element for element in elements if not _should_skip_schematic_element(element)]
    if not elements:
        Image.new("RGB", (640, 360), "white").save(path)
        return

    net_groups = _group_pins_by_net(elements)
    layout = _build_layout(elements, net_groups)
    dense_nets = _select_nets_labeled(net_groups, layout)
    router = _Router(layout)
    _reserve_pin_stubs(router, net_groups, layout)
    wires: list[tuple[str, list[tuple[float, float]]]] = []
    labels: list[tuple[str, float, float]] = []
    junctions: list[tuple[float, float]] = []
    no_connects: list[tuple[float, float]] = []

    for net, net_pins in sorted(net_groups.items()):
        positions = [
            (element, pin, *_element_pin_point(element, pin, layout[element.ref].x, layout[element.ref].y))
            for element, pin in net_pins
            if element.ref in layout
        ]
        if len(positions) <= 1:
            if positions:
                _element, _pin, x, y = positions[0]
                no_connects.append((x, y))
            continue
        if net in dense_nets:
            for _element, _pin, x, y in positions:
                labels.append((net, x - 9.0, y - 2.2))
        elif len(positions) == 2:
            (_e1, _p1, x1, y1), (_e2, _p2, x2, y2) = positions
            anchor1 = _anchor_wire_pin_element(_e1, _p1, x1, y1)
            anchor2 = _anchor_wire_pin_element(_e2, _p2, x2, y2)
            stub1 = _element_pin_anchor_segment(_e1, _p1, x1, y1)
            stub2 = _element_pin_anchor_segment(_e2, _p2, x2, y2)
            router.occupy(stub1, net)
            router.occupy(stub2, net)
            wires.append((net, stub1))
            wires.append((net, router.route(anchor1, anchor2, net)))
            wires.append((net, stub2))
        else:
            points = [(element, pin, x, y) for element, pin, x, y in positions]
            anchors = [_anchor_wire_pin_element(element, pin, x, y) for element, pin, x, y in points]
            xs = [x for x, _y in anchors]
            ys = [y for _x, y in anchors]
            hub = router.nearest_free_point(
                (round(_snap((min(xs) + max(xs)) / 2), 2), round(_snap(_select_hub_y(ys)), 2)),
                net,
            )
            junctions.append(hub)
            stubs = [_element_pin_anchor_segment(element, pin, x, y) for element, pin, x, y in points]
            for stub in stubs:
                router.occupy(stub, net)
                wires.append((net, stub))
            for anchor in anchors:
                wires.append((net, router.route(anchor, hub, net)))

    boundary_points: list[tuple[float, float]] = []
    for symbol in layout.values():
        x1, y1, x2, y2 = symbol.bbox
        boundary_points.extend([(x1, y1), (x2, y2)])
    for _net, points in wires:
        boundary_points.extend(points)
    boundary_points.extend((x, y) for _text, x, y in labels)
    boundary_points.extend(no_connects)
    min_x = min(x for x, _y in boundary_points) - 12.0
    max_x = max(x for x, _y in boundary_points) + 16.0
    min_y = min(y for _x, y in boundary_points) - 12.0
    max_y = max(y for _x, y in boundary_points) + 16.0
    width_mm = max(80.0, max_x - min_x)
    height_mm = max(50.0, max_y - min_y)
    scale = min(9.0, max(4.0, 2800.0 / max(width_mm, height_mm)))
    width = max(640, int(math.ceil(width_mm * scale)))
    height = max(360, int(math.ceil(height_mm * scale)))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def p(x: float, y: float) -> tuple[int, int]:
        """Map schematic millimeters to preview pixel coordinates."""
        return int(round((x - min_x) * scale)), int(round((y - min_y) * scale))

    wire_width = max(2, int(round(scale * 0.22)))
    for _net, points in wires:
        for a, b in zip(points, points[1:], strict=False):
            draw.line([p(*a), p(*b)], fill=(48, 72, 103), width=wire_width)
    for x, y in junctions:
        px, py = p(x, y)
        r = max(2, int(round(scale * 0.35)))
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(48, 72, 103))
    for x, y in no_connects:
        px, py = p(x, y)
        r = max(4, int(round(scale * 0.6)))
        draw.line((px - r, py - r, px + r, py + r), fill=(170, 70, 70), width=wire_width)
        draw.line((px - r, py + r, px + r, py - r), fill=(170, 70, 70), width=wire_width)

    elements_by_ref = {element.ref: element for element in elements}
    for symbol in sorted(layout.values(), key=lambda s: (s.y, s.x, s.ref)):
        element = elements_by_ref[symbol.ref]
        _draw_symbol_png(draw, p, font, element, symbol, scale)

    for text, x, y in labels:
        px, py = p(x, y)
        draw.rectangle((px - 2, py - 1, px + 6 * len(text) + 4, py + 10), fill=(255, 255, 255))
        draw.text((px, py), text[:18], fill=(34, 112, 78), font=font)

    image.save(path)

def _draw_symbol_png(
    draw: ImageDraw.ImageDraw,
    map_point,
    font: ImageFont.ImageFont,
    element: Element,
    symbol: SymbolLayout,
    scale: float,
) -> None:
    """Draw one schematic symbol into the PNG preview canvas."""
    line_width = max(2, int(round(scale * 0.18)))
    outline = (31, 45, 64)
    fill = (252, 252, 248)
    ref_color = (22, 31, 45)
    value_color = (78, 88, 101)

    def line(a: tuple[float, float], b: tuple[float, float], color=outline) -> None:
        """Draw one preview line between schematic-space points."""
        draw.line([map_point(*a), map_point(*b)], fill=color, width=line_width)

    def rect(x1: float, y1: float, x2: float, y2: float, body_fill=fill) -> None:
        """Draw one preview rectangle from schematic-space corners."""
        draw.rectangle((*map_point(x1, y1), *map_point(x2, y2)), fill=body_fill, outline=outline, width=line_width)

    def centered_text(text: str, x: float, y: float, color=ref_color) -> None:
        """Draw centered preview text near a symbol body."""
        text = text[:22]
        px, py = map_point(x, y)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((px - (bbox[2] - bbox[0]) // 2, py), text, fill=color, font=font)

    if symbol.kind == "discrete_2pin" and len(element.pins) == 2:
        y = symbol.y
        left = _element_pin_point(element, "1", symbol.x, symbol.y)
        right = _element_pin_point(element, "2", symbol.x, symbol.y)
        line(left, (symbol.x - 4.0, y))
        line((symbol.x + 4.0, y), right)
        if element.type == "Capacitor":
            line((symbol.x - 1.3, y - 3.0), (symbol.x - 1.3, y + 3.0))
            line((symbol.x + 1.3, y - 3.0), (symbol.x + 1.3, y + 3.0))
        elif element.type == "Diode":
            a = map_point(symbol.x - 2.8, y - 3.0)
            b = map_point(symbol.x - 2.8, y + 3.0)
            c = map_point(symbol.x + 2.5, y)
            draw.polygon([a, b, c], outline=outline, fill=(255, 255, 255))
            line((symbol.x + 2.7, y - 3.1), (symbol.x + 2.7, y + 3.1))
        elif element.type == "Inductor":
            for offset in (-2.4, -0.8, 0.8, 2.4):
                x0, y0 = map_point(symbol.x + offset - 0.9, y - 1.7)
                x1, y1 = map_point(symbol.x + offset + 0.9, y + 1.7)
                draw.arc((x0, y0, x1, y1), 180, 360, fill=outline, width=line_width)
        else:
            rect(symbol.x - 3.8, y - 2.4, symbol.x + 3.8, y + 2.4)
        centered_text(element.ref, symbol.x, symbol.y - 8.0)
        if element.value and element.value != element.ref:
            centered_text(element.value, symbol.x, symbol.y + 5.2, value_color)
        return

    if symbol.kind == "testpoint":
        pin = _element_pin_point(element, "1", symbol.x, symbol.y)
        line(pin, (symbol.x, symbol.y))
        px, py = map_point(symbol.x, symbol.y)
        r = max(4, int(round(scale * 1.3)))
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 255, 255), outline=outline, width=line_width)
        centered_text(element.ref, symbol.x, symbol.y - 8.0)
        return

    x1, y1, x2, y2 = symbol.bbox
    rect(x1, y1, x2, y2)
    centered_text(element.ref, symbol.x, y1 - 4.0)
    if element.value and element.value != element.ref:
        centered_text(element.value, symbol.x, y2 + 1.6, value_color)
    for pin in sorted(element.pins, key=lambda p: natural_pin_sort_key(p)):
        px, py = _element_pin_point(element, pin, symbol.x, symbol.y)
        line((px, py), (x1, py))
        mx, my = map_point(px, py)
        r = max(2, int(round(scale * 0.35)))
        draw.ellipse((mx - r, my - r, mx + r, my + r), fill=(255, 255, 255), outline=outline, width=1)
        tx, ty = map_point(x1 + 1.0, py - 1.4)
        draw.text((tx, ty), str(pin)[:4], fill=value_color, font=font)
