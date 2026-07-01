from __future__ import annotations

from schematic_generator.kicad.layout_graph import _build_neighborhood, _component_layers, _connected_components, _order_layers
from schematic_generator.kicad.layout_placement import (
    _group_pins_by_net,
    _is_power_net,
    _is_short_straight_connection,
    _length_wire_estimated,
    _move_to_positive_coordinates,
    _overlap,
    _pack_fallback,
    _place_layers,
    _points_on_one_axis,
    _pull_connected,
    _select_nets_labeled,
    _separate_collisions,
    _should_skip_schematic_element,
    _snap,
    _symbol_size,
)
from schematic_generator.kicad.models import PinRef, SymbolLayout
from schematic_generator.models import Element


def _build_layout(elements: list[Element], net_groups: dict[str, list[PinRef]]) -> dict[str, SymbolLayout]:
    """Build a collision-reduced schematic symbol layout from element connectivity."""
    if not elements:
        return {}
    elements_by_ref = {element.ref: element for element in elements}
    neighbors = _build_neighborhood(net_groups)
    components = _connected_components(elements, neighbors)
    layout: dict[str, SymbolLayout] = {}

    cursor_y = 25.0
    for component in components:
        layers = _component_layers(component, neighbors, elements_by_ref)
        layers = _order_layers(layers, neighbors, elements_by_ref)
        partial, height = _place_layers(layers, elements_by_ref, cursor_y)
        layout.update(partial)
        cursor_y += max(24.0, height + 22.0)

    _separate_collisions(layout)
    _pack_fallback(layout)
    _move_to_positive_coordinates(layout)
    return layout
