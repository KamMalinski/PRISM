from __future__ import annotations

from schematic_generator.kicad.layout import _snap
from schematic_generator.kicad.models import PinRef, SymbolLayout, TWO_PIN_TYPES
from schematic_generator.models import Element


def _element_pin_point(element: Element, pin: str, x: float, y: float) -> tuple[float, float]:
    """Compute the schematic coordinate of one element pin."""
    if element.type in TWO_PIN_TYPES and len(element.pins) == 2:
        return (x - 5.08, y) if str(pin) == "1" else (x + 5.08, y)
    if len(element.pins) <= 1:
        return x, y + 2.54
    try:
        index = max(1, int(pin))
    except ValueError:
        index = 1
    y0 = y + (len(element.pins) - 1) * 1.27
    return x - 5.08, y0 - (index - 1) * 2.54

def _anchor_wire_pin(x: float, y: float) -> tuple[float, float]:
    """Place a default wire anchor to the left of a pin."""
    return round(_snap(x - 5.08), 2), round(y, 2)

def _pin_anchor_segment(x: float, y: float) -> list[tuple[float, float]]:
    """Return the short stub from a pin to its default wire anchor."""
    return [(round(x, 2), round(y, 2)), _anchor_wire_pin(x, y)]

def _anchor_wire_pin_element(element: Element, pin: str, x: float, y: float) -> tuple[float, float]:
    """Choose the correct wire anchor side for a pin on a specific element kind."""
    if element.type in TWO_PIN_TYPES and len(element.pins) == 2 and str(pin) != "1":
        return round(_snap(x + 5.08), 2), round(y, 2)
    return _anchor_wire_pin(x, y)

def _element_pin_anchor_segment(element: Element, pin: str, x: float, y: float) -> list[tuple[float, float]]:
    """Return a pin-to-anchor stub for an element pin."""
    return [(round(x, 2), round(y, 2)), _anchor_wire_pin_element(element, pin, x, y)]

def _reserve_pin_stubs(
    router,
    net_groups: dict[str, list[PinRef]],
    layout: dict[str, SymbolLayout],
) -> None:
    """Reserve all pin stubs in the router before longer net routing starts."""
    for net, pins in net_groups.items():
        positions = [
            (element, pin, *_element_pin_point(element, pin, layout[element.ref].x, layout[element.ref].y))
            for element, pin in pins
            if element.ref in layout
        ]
        if len(positions) <= 1:
            continue
        for element, pin, x, y in positions:
            router.occupy(_element_pin_anchor_segment(element, pin, x, y), net)
