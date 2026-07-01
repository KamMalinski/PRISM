from __future__ import annotations

from uuid import uuid4

from schematic_generator.kicad.layout import _snap
from schematic_generator.kicad.models import TWO_PIN_TYPES
from schematic_generator.kicad.pins import _element_pin_point
from schematic_generator.models import Element


def natural_pin_sort_key(pin: str) -> tuple[int, int | str]:
    """Sort numeric pin names before non-numeric pin names."""
    text = str(pin)
    return (0, int(text)) if text.isdigit() else (1, text)

def _label_near_pin(net: str, x: float, y: float) -> list[str]:
    """Create a short stub and net label beside a pin."""
    x2 = round(_snap(x - 5.08), 2)
    return [*_wire([(x, y), (x2, y)]), *_label_net(net, x2, y)]

def _symbol_testpoint() -> list[str]:
    """Return local KiCad symbol definitions for one-pin test points."""
    return [
        '    (symbol "GeneratedSymbols:TP"',
        "      (pin_numbers hide)",
        "      (pin_names (offset 0.508))",
        "      (exclude_from_sim no)",
        "      (in_bom yes)",
        "      (on_board yes)",
        '      (property "Reference" "TP" (at 0 3.81 0)',
        "        (effects (font (size 1.27 1.27))))",
        '      (property "Value" "TP" (at 0 -3.81 0)',
        "        (effects (font (size 1.27 1.27))))",
        '      (property "Footprint" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        '      (property "Datasheet" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        '      (symbol "TP_0_1"',
        "        (circle (center 0 0) (radius 1.27)",
        "          (stroke (width 0.254) (type default))",
        "          (fill (type none)))",
        '        (pin passive line (at 0 2.54 270) (length 1.27)',
        '          (name "1" (effects (font (size 1.27 1.27))))',
        '          (number "1" (effects (font (size 1.27 1.27)))))',
        "      )",
        "    )",
    ]

def _symbol_pinrow(pin_count: int) -> list[str]:
    """Return a local KiCad symbol definition for a generic pin row."""
    height = max(5.08, (pin_count + 1) * 2.54)
    y0 = -((pin_count - 1) * 1.27)
    lines = [
        f'    (symbol "GeneratedSymbols:PinRow_{pin_count}"',
        "      (pin_numbers hide)",
        "      (pin_names (offset 0.508))",
        "      (exclude_from_sim no)",
        "      (in_bom yes)",
        "      (on_board yes)",
        f'      (property "Reference" "J" (at 0 {-height / 2 - 1.5:.2f} 0)',
        "        (effects (font (size 1.27 1.27))))",
        f'      (property "Value" "{pin_count} pin" (at 0 {height / 2 + 1.5:.2f} 0)',
        "        (effects (font (size 1.27 1.27))))",
        '      (property "Footprint" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        '      (property "Datasheet" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        f'      (symbol "PinRow_{pin_count}_0_1"',
        f"        (rectangle (start -2.54 {-height / 2:.2f}) (end 2.54 {height / 2:.2f})",
        "          (stroke (width 0.254) (type default))",
        "          (fill (type none)))",
    ]
    for nr in range(1, pin_count + 1):
        y = y0 + (nr - 1) * 2.54
        lines.extend([
            f'        (pin passive line (at -5.08 {y:.2f} 0) (length 2.54)',
            f'          (name "{nr}" (effects (font (size 1.27 1.27))))',
            f'          (number "{nr}" (effects (font (size 1.27 1.27)))))',
        ])
    lines.extend(["      )", "    )"])
    return lines

def _symbol_two_pin(type: str) -> list[str]:
    """Return a local KiCad symbol definition for a two-pin component type."""
    ref = {"Resistor": "R", "Capacitor": "C", "Diode": "D", "Inductor": "L"}[type]
    value = {"Resistor": "R", "Capacitor": "C", "Diode": "D", "Inductor": "L"}[type]
    body = {
        "Resistor": [
            "        (polyline (pts (xy -2.54 0) (xy -1.9 -1.0) (xy -1.1 1.0) (xy -0.3 -1.0) (xy 0.5 1.0) (xy 1.3 -1.0) (xy 2.1 1.0) (xy 2.54 0))",
            "          (stroke (width 0.254) (type default))",
            "          (fill (type none)))",
        ],
        "Capacitor": [
            "        (polyline (pts (xy -0.8 -2.0) (xy -0.8 2.0))",
            "          (stroke (width 0.254) (type default))",
            "          (fill (type none)))",
            "        (polyline (pts (xy 0.8 -2.0) (xy 0.8 2.0))",
            "          (stroke (width 0.254) (type default))",
            "          (fill (type none)))",
        ],
        "Diode": [
            "        (polyline (pts (xy -1.8 -1.8) (xy -1.8 1.8) (xy 1.2 0) (xy -1.8 -1.8))",
            "          (stroke (width 0.254) (type default))",
            "          (fill (type none)))",
            "        (polyline (pts (xy 1.2 -1.8) (xy 1.2 1.8))",
            "          (stroke (width 0.254) (type default))",
            "          (fill (type none)))",
        ],
        "Inductor": [
            "        (arc (start -2.4 0) (mid -1.8 -1.0) (end -1.2 0)",
            "          (stroke (width 0.254) (type default)) (fill (type none)))",
            "        (arc (start -1.2 0) (mid -0.6 -1.0) (end 0 0)",
            "          (stroke (width 0.254) (type default)) (fill (type none)))",
            "        (arc (start 0 0) (mid 0.6 -1.0) (end 1.2 0)",
            "          (stroke (width 0.254) (type default)) (fill (type none)))",
            "        (arc (start 1.2 0) (mid 1.8 -1.0) (end 2.4 0)",
            "          (stroke (width 0.254) (type default)) (fill (type none)))",
        ],
    }[type]
    return [
        f'    (symbol "GeneratedSymbols:{type}"',
        "      (pin_numbers hide)",
        "      (pin_names (offset 0.508))",
        "      (exclude_from_sim no)",
        "      (in_bom yes)",
        "      (on_board yes)",
        f'      (property "Reference" "{ref}" (at 0 -3.81 0)',
        "        (effects (font (size 1.27 1.27))))",
        f'      (property "Value" "{value}" (at 0 3.81 0)',
        "        (effects (font (size 1.27 1.27))))",
        '      (property "Footprint" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        '      (property "Datasheet" "" (at 0 0 0)',
        "        (effects (font (size 1.27 1.27)) hide))",
        f'      (symbol "{type}_0_1"',
        *body,
        '        (pin passive line (at -5.08 0 0) (length 2.54)',
        '          (name "1" (effects (font (size 1.27 1.27))))',
        '          (number "1" (effects (font (size 1.27 1.27)))))',
        '        (pin passive line (at 5.08 0 180) (length 2.54)',
        '          (name "2" (effects (font (size 1.27 1.27))))',
        '          (number "2" (effects (font (size 1.27 1.27)))))',
        "      )",
        "    )",
    ]

def _symbol_instance(element: Element, x: float, y: float) -> list[str]:
    """Create a KiCad symbol instance for one generated element."""
    uuid_symbol = _uuid()
    pin_count = len(element.pins)
    if element.type in TWO_PIN_TYPES and pin_count == 2:
        lib_id = f"GeneratedSymbols:{element.type}"
    else:
        lib_id = "GeneratedSymbols:TP" if pin_count <= 1 else f"GeneratedSymbols:PinRow_{pin_count}"
    lines = [
        f'  (symbol (lib_id "{lib_id}")',
        f"    (at {x:.2f} {y:.2f} 0)",
        "    (unit 1)",
        "    (exclude_from_sim no)",
        "    (in_bom yes)",
        "    (on_board yes)",
        f'    (uuid "{uuid_symbol}")',
        f'    (property "Reference" "{_txt(element.ref)}" (at {x:.2f} {y - 4:.2f} 0)',
        "      (effects (font (size 1.27 1.27))))",
        f'    (property "Value" "{_txt(element.value)}" (at {x:.2f} {y + 6:.2f} 0)',
        "      (effects (font (size 1.27 1.27))))",
        f'    (property "Footprint" "{_txt(element.footprint)}" (at {x:.2f} {y:.2f} 0)',
        "      (effects (font (size 1.27 1.27)) hide))",
    ]
    for pin in element.pins:
        lines.append(f'    (pin "{_txt(pin)}" (uuid "{_uuid()}"))')
    lines.append("  )")
    return lines

def _pin_descriptions(element: Element, x: float, y: float) -> list[str]:
    """Create small local pin labels near a generated element symbol."""
    lines: list[str] = []
    for pin, description in element.pin_descriptions.items():
        if not description:
            continue
        px, py = _element_pin_point(element, pin, x, y)
        lines.extend(_text_local(description, px, py - 1.8))
    return lines

def _text_local(text: str, x: float, y: float) -> list[str]:
    """Create a KiCad text item at a schematic coordinate."""
    return [
        f'  (text "{_txt(text)}"',
        f"    (at {x:.2f} {y:.2f} 0)",
        "    (effects (font (size 1.0 1.0)))",
        f'    (uuid "{_uuid()}"))',
    ]

def _label_net(name: str, x: float, y: float) -> list[str]:
    """Create a KiCad global label at a schematic coordinate."""
    return [
        f'  (label "{_txt(name)}"',
        f"    (at {x:.2f} {y:.2f} 0)",
        "    (effects (font (size 1.27 1.27)))",
        f'    (uuid "{_uuid()}"))',
    ]

def _wire(points: list[tuple[float, float]]) -> list[str]:
    """Create a KiCad wire item from an orthogonal point list."""
    lines: list[str] = []
    points = _orthogonalize_points(points)
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        saved_points = " ".join(f"(xy {x:.2f} {y:.2f})" for x, y in (start, end))
        lines.extend([
            "  (wire",
            f"    (pts {saved_points})",
            "    (stroke (width 0) (type default))",
            f'    (uuid "{_uuid()}"))',
        ])
    return lines

def _orthogonalize_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Insert a bend point when two consecutive points are diagonal."""
    if len(points) <= 1:
        return points
    result: list[tuple[float, float]] = [points[0]]
    for x2, y2 in points[1:]:
        x1, y1 = result[-1]
        x2, y2 = round(x2, 2), round(y2, 2)
        if abs(x1 - x2) > 1e-6 and abs(y1 - y2) > 1e-6:
            result.append((x2, y1))
        if result[-1] != (x2, y2):
            result.append((x2, y2))
    return result

def _junction(x: float, y: float) -> list[str]:
    """Create a KiCad junction marker."""
    return [
        "  (junction",
        f"    (at {x:.2f} {y:.2f})",
        "    (diameter 0)",
        "    (color 0 0 0 0)",
        f'    (uuid "{_uuid()}"))',
    ]

def _no_connect(x: float, y: float) -> list[str]:
    """Create a KiCad no-connect marker."""
    return [
        "  (no_connect",
        f"    (at {x:.2f} {y:.2f})",
        f'    (uuid "{_uuid()}"))',
    ]

def _uuid() -> str:
    """Return a KiCad-compatible random UUID string."""
    return str(uuid4())

def _txt(value: str) -> str:
    """Escape text for KiCad s-expression output."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
