from __future__ import annotations

from schematic_generator.kicad.models import PinRef, SymbolLayout, TWO_PIN_TYPES
from schematic_generator.models import Element


def _place_layers(
    layers: dict[int, list[str]],
    elements_by_ref: dict[str, Element],
    y_start: float,
) -> tuple[dict[str, SymbolLayout], float]:
    """Place ordered graph layers on a grid and return their symbol layouts."""
    layout: dict[str, SymbolLayout] = {}
    x_start = 25.0
    x_gap = 34.0
    max_height = 0.0
    for index_layers, layer in enumerate(sorted(layers)):
        refs = layers[layer]
        heights = [_symbol_size(elements_by_ref[ref])[1] for ref in refs]
        height_layers = sum(heights) + max(0, len(refs) - 1) * 14.0
        max_height = max(max_height, height_layers)
        y = y_start
        x = x_start + index_layers * x_gap
        for ref, height in zip(refs, heights, strict=False):
            element = elements_by_ref[ref]
            width, height, kind = _symbol_size(element)
            y += height / 2
            y_symbol = _snap_symbol_y(element, y)
            layout[ref] = SymbolLayout(
                ref=ref,
                x=round(_snap(x), 2),
                y=round(y_symbol, 2),
                width=width,
                height=height,
                source_x=element.x,
                source_y=element.y,
                kind=kind,
            )
            y += height / 2 + 14.0
    return layout, max_height

def _snap_symbol_y(element: Element, y: float) -> float:
    """Snap vertical symbol position while keeping pin rows aligned to the KiCad grid."""
    if element.type not in TWO_PIN_TYPES and len(element.pins) > 1 and len(element.pins) % 2 == 0:
        return _snap(y - 1.27) + 1.27
    return _snap(y)

def _symbol_size(element: Element) -> tuple[float, float, str]:
    """Choose a symbol footprint size and layout kind from element type and pin count."""
    pin_count = max(1, len(element.pins))
    if element.type in TWO_PIN_TYPES and pin_count == 2:
        return 18.0, 12.0, "discrete_2pin"
    if pin_count <= 1:
        return 12.0, 12.0, "testpoint"
    return 18.0, max(14.0, pin_count * 2.54 + 7.0), "pinrow"

def _pull_connected(layout: dict[str, SymbolLayout], net_groups: dict[str, list[PinRef]]) -> None:
    """Apply a light relaxation pass that pulls connected symbols closer together."""
    for _ in range(16):
        moves = {ref: [0.0, 0.0, 0] for ref in layout}
        for pins in net_groups.values():
            refs = [element.ref for element, _pin in pins if element.ref in layout]
            if len(refs) < 2 or len(refs) > 6:
                continue
            cx = sum(layout[ref].x for ref in refs) / len(refs)
            cy = sum(layout[ref].y for ref in refs) / len(refs)
            for ref in refs:
                moves[ref][0] += (cx - layout[ref].x) * 0.035
                moves[ref][1] += (cy - layout[ref].y) * 0.035
                moves[ref][2] += 1
        for ref, (dx, dy, count) in moves.items():
            if count:
                layout[ref].x += dx
                layout[ref].y += dy

def _separate_collisions(layout: dict[str, SymbolLayout]) -> None:
    """Iteratively move overlapping symbols apart while preserving rough layout order."""
    symbols = list(layout.values())
    for _ in range(180):
        changed = False
        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                overlap = _overlap(a, b, margin=4.0)
                if not overlap:
                    continue
                ox, oy = overlap
                dx = a.x - b.x
                dy = a.y - b.y
                if abs(dx) < 0.01 and abs(dy) < 0.01:
                    dx, dy = 1.0, 0.0
                dl = max(0.01, math.hypot(dx, dy))
                if ox < oy:
                    push = (ox / 2.0 + 0.8) * (1 if dx >= 0 else -1)
                    a.x += push
                    b.x -= push
                else:
                    push = (oy / 2.0 + 0.8) * (1 if dy >= 0 else -1)
                    a.y += push
                    b.y -= push
                changed = True
        if not changed:
            break

def _pack_fallback(layout: dict[str, SymbolLayout]) -> None:
    """Pack any remaining unplaced or invalid symbols into a fallback grid."""
    placed: list[SymbolLayout] = []
    for symbol in sorted(layout.values(), key=lambda s: (s.y, s.x, s.ref)):
        attempts = 0
        while any(_overlap(symbol, other, margin=4.0) for other in placed):
            symbol.x += max(22.0, symbol.width + 8.0)
            attempts += 1
            if symbol.x > 275.0 or attempts > 18:
                symbol.x = 25.0
                symbol.y += max(18.0, symbol.height + 8.0)
                attempts = 0
        symbol.x = round(_snap(symbol.x), 2)
        symbol.y = round(_snap(symbol.y), 2)
        placed.append(symbol)

def _move_to_positive_coordinates(layout: dict[str, SymbolLayout]) -> None:
    """Shift all symbols so exported coordinates stay in the positive quadrant."""
    if not layout:
        return
    min_x = min(symbol.bbox[0] for symbol in layout.values())
    min_y = min(symbol.bbox[1] for symbol in layout.values())
    dx = max(0.0, 12.0 - min_x)
    dy = max(0.0, 12.0 - min_y)
    for symbol in layout.values():
        symbol.x = round(symbol.x + dx, 2)
        symbol.y = round(symbol.y + dy, 2)

def _overlap(a: SymbolLayout, b: SymbolLayout, margin: float = 0.0) -> tuple[float, float] | None:
    """Return overlap distance for two symbol boxes, including optional margin."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ox = min(ax2 + margin, bx2 + margin) - max(ax1 - margin, bx1 - margin)
    oy = min(ay2 + margin, by2 + margin) - max(ay1 - margin, by1 - margin)
    return (ox, oy) if ox > 0 and oy > 0 else None

def _snap(value: float, grid: float = 2.54) -> float:
    """Round a coordinate to the configured KiCad grid."""
    return round(value / grid) * grid

def _is_power_net(net: str) -> bool:
    """Identify power-like net names that should use labels instead of full routing pressure."""
    return net.upper().startswith(("GND", "VCC", "VDD", "VSS", "+", "3V", "5V"))

def _select_nets_labeled(net_groups: dict[str, list[PinRef]], layout: dict[str, SymbolLayout]) -> set[str]:
    """Choose dense or power nets that are clearer as labels than explicit wires."""
    # A usable schematic must show physical connections instead of only labels
    # scattered around pins. Keep this function as an explicit extension point
    # for a possible simplified mode, but route all multi-pin nets by default.
    return set()

def _points_on_one_axis(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Check whether two points are horizontally or vertically aligned."""
    return abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6

def _is_short_straight_connection(a: tuple[float, float], b: tuple[float, float], limit: float = 20.0) -> bool:
    """Detect short direct connections that should remain as wires."""
    if not _points_on_one_axis(a, b):
        return False
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= limit

def _length_wire_estimated(points: list[tuple[float, float]]) -> float:
    """Estimate Manhattan span of a routed point list."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

def _group_pins_by_net(elements: list[Element]) -> dict[str, list[PinRef]]:
    """Group element pins by their assigned electrical net name."""
    groups: dict[str, list[PinRef]] = {}
    for element in elements:
        for pin, net in element.pins.items():
            if net and net != "NET?":
                groups.setdefault(net, []).append((element, pin))
    return groups

def _should_skip_schematic_element(element: Element) -> bool:
    """Skip mechanical mounting-hole elements that should not appear in the schematic."""
    type = element.type.lower().replace(" ", "_")
    value = element.value.lower().replace(" ", "_")
    footprint = element.footprint.lower()
    ref = element.ref.upper()
    return (
        type in {"mounting_hole", "mountinghole"}
        or value.startswith("mounting_hole")
        or "mountinghole" in footprint
        or ref.startswith("MH")
    )
