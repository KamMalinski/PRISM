from __future__ import annotations

from pathlib import Path

from schematic_generator.kicad.layout import _build_layout, _group_pins_by_net, _select_nets_labeled, _should_skip_schematic_element
from schematic_generator.kicad.library import _save_local_symbol_library
from schematic_generator.kicad.models import TWO_PIN_TYPES, WireSegment
from schematic_generator.kicad.pins import _element_pin_point, _reserve_pin_stubs
from schematic_generator.kicad.preflight import _preflight_wires, _save_report_layout
from schematic_generator.kicad.routing import _Router
from schematic_generator.kicad.sexpr import _label_near_pin, _no_connect, _pin_descriptions, _symbol_instance, _symbol_pinrow, _symbol_testpoint, _symbol_two_pin
from schematic_generator.kicad.wires import _wires_net
from schematic_generator.models import Element


def save_kicad_schematic(path: str | Path, elements: list[Element]) -> None:
    """Write a complete KiCad schematic and companion layout diagnostics for generated elements."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = [element for element in elements if not _should_skip_schematic_element(element)]
    net_groups = _group_pins_by_net(elements)
    layout = _build_layout(elements, net_groups)
    router = _Router(layout)
    _reserve_pin_stubs(router, net_groups, layout)
    dense_nets = _select_nets_labeled(net_groups, layout)
    pin_counts = sorted({len(e.pins) for e in elements if len(e.pins) > 1 and e.type not in TWO_PIN_TYPES})
    types_two_pin = sorted({e.type for e in elements if e.type in TWO_PIN_TYPES})

    library_symbols = [
        *_symbol_testpoint(),
    ]
    for pin_count in pin_counts:
        library_symbols.extend(_symbol_pinrow(pin_count))
    for type in types_two_pin:
        library_symbols.extend(_symbol_two_pin(type))

    lines = [
        "(kicad_sch",
        "  (version 20230121)",
        '  (generator "schematic_generator")',
        '  (paper "A3")',
        "  (lib_symbols",
        *library_symbols,
    ]
    lines.append("  )")

    for element in elements:
        symbol = layout[element.ref]
        lines.extend(_symbol_instance(element, symbol.x, symbol.y))
        lines.extend(_pin_descriptions(element, symbol.x, symbol.y))

    segments_wires: list[WireSegment] = []
    for net, net_pins in sorted(net_groups.items()):
        pin_positions = [
            (element, pin, *_element_pin_point(element, pin, layout[element.ref].x, layout[element.ref].y))
            for element, pin in net_pins
            if element.ref in layout
        ]
        if len(pin_positions) <= 1:
            if pin_positions:
                _element, _pin, x, y = pin_positions[0]
                lines.extend(_no_connect(x, y))
            continue
        if net in dense_nets:
            for _element, _pin, x, y in pin_positions:
                lines.extend(_label_near_pin(net, x, y))
        else:
            lines.extend(_wires_net(net, pin_positions, router, segments_wires))

    lines.append(")")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _save_local_symbol_library(path, library_symbols)
    preflight = _preflight_wires(net_groups, layout, dense_nets, segments_wires)
    _save_report_layout(path, layout, net_groups, dense_nets, router, preflight)
