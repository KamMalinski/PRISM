from __future__ import annotations

from schematic_generator.kicad.layout_placement import _is_power_net
from schematic_generator.kicad.models import PinRef
from schematic_generator.models import Element


def _build_neighborhood(net_groups: dict[str, list[PinRef]]) -> dict[str, dict[str, float]]:
    """Convert shared non-power nets into weighted element adjacency data."""
    neighbors: dict[str, dict[str, float]] = {}
    for net, pins in net_groups.items():
        if _is_power_net(net):
            continue
        refs = sorted({element.ref for element, _pin in pins})
        if len(refs) < 2:
            continue
        weight = 3.0 if len(refs) == 2 else max(0.6, 2.0 / (len(refs) - 1))
        for i, a in enumerate(refs):
            neighbors.setdefault(a, {})
            for b in refs[i + 1:]:
                neighbors.setdefault(b, {})
                neighbors[a][b] = neighbors[a].get(b, 0.0) + weight
                neighbors[b][a] = neighbors[b].get(a, 0.0) + weight
    return neighbors

def _connected_components(elements: list[Element], neighbors: dict[str, dict[str, float]]) -> list[list[str]]:
    """Group element references into connected components using the adjacency graph."""
    remaining = {element.ref for element in elements}
    result: list[list[str]] = []
    positions = {element.ref: (element.y, element.x) for element in elements}
    while remaining:
        start = min(remaining, key=lambda ref: positions.get(ref, (0.0, 0.0)))
        stack = [start]
        component: list[str] = []
        remaining.remove(start)
        while stack:
            ref = stack.pop()
            component.append(ref)
            for neighbor in sorted(neighbors.get(ref, {}), key=lambda r: positions.get(r, (0.0, 0.0))):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(sorted(component, key=lambda ref: positions.get(ref, (0.0, 0.0))))
    return sorted(result, key=lambda component: min(positions.get(ref, (0.0, 0.0)) for ref in component))

def _component_layers(
    component: list[str],
    neighbors: dict[str, dict[str, float]],
    elements_by_ref: dict[str, Element],
) -> dict[int, list[str]]:
    """Assign graph layers for one connected component using farthest-end traversal."""
    if len(component) == 1:
        return {0: component}
    start_a, start_b = _component_diameter(component, neighbors, elements_by_ref)
    if elements_by_ref[start_b].x < elements_by_ref[start_a].x:
        start_a = start_b
    distances = _bfs_distances(start_a, neighbors, set(component))
    layers: dict[int, list[str]] = {}
    for ref in component:
        layer = distances.get(ref, 0)
        layers.setdefault(layer, []).append(ref)
    return layers

def _component_diameter(
    component: list[str],
    neighbors: dict[str, dict[str, float]],
    elements_by_ref: dict[str, Element],
) -> tuple[str, str]:
    """Measure the longest graph distance inside one connected component."""
    start = min(component, key=lambda ref: (elements_by_ref[ref].x, elements_by_ref[ref].y))
    a = _farthest_ref(start, neighbors, set(component), elements_by_ref)
    b = _farthest_ref(a, neighbors, set(component), elements_by_ref)
    return a, b

def _farthest_ref(
    start: str,
    neighbors: dict[str, dict[str, float]],
    allowed: set[str],
    elements_by_ref: dict[str, Element],
) -> str:
    """Return the reference farthest from the selected starting element."""
    distances = _bfs_distances(start, neighbors, allowed)
    return max(
        allowed,
        key=lambda ref: (
            distances.get(ref, 0),
            abs(elements_by_ref[ref].x - elements_by_ref[start].x) + abs(elements_by_ref[ref].y - elements_by_ref[start].y),
        ),
    )

def _bfs_distances(start: str, neighbors: dict[str, dict[str, float]], allowed: set[str]) -> dict[str, int]:
    """Run breadth-first traversal over allowed graph nodes and return hop distances."""
    distances = {start: 0}
    queue = [start]
    for ref in queue:
        for neighbor in neighbors.get(ref, {}):
            if neighbor in allowed and neighbor not in distances:
                distances[neighbor] = distances[ref] + 1
                queue.append(neighbor)
    return distances

def _order_layers(
    layers: dict[int, list[str]],
    neighbors: dict[str, dict[str, float]],
    elements_by_ref: dict[str, Element],
) -> dict[int, list[str]]:
    """Sort graph layers to keep strongly connected neighbors visually close."""
    indices = sorted(layers)
    for layer in indices:
        layers[layer].sort(key=lambda ref: (elements_by_ref[ref].y, elements_by_ref[ref].x, ref))
    position = {ref: i for layer in indices for i, ref in enumerate(layers[layer])}
    for _ in range(10):
        for direction in (indices, list(reversed(indices))):
            for layer in direction:
                def layer_sort_key(ref: str) -> tuple[float, float, str]:
                    """Rank one reference inside a layer by neighbor pull and original position."""
                    values = []
                    for neighbor, weight in neighbors.get(ref, {}).items():
                        for delta in (-1, 1):
                            if neighbor in layers.get(layer + delta, []):
                                values.append((position.get(neighbor, 0), weight))
                    if not values:
                        return (position.get(ref, 0), elements_by_ref[ref].y, ref)
                    suma_wag = sum(w for _poz, w in values) or 1.0
                    barycentrum = sum(poz * w for poz, w in values) / suma_wag
                    return (barycentrum, elements_by_ref[ref].y, ref)

                layers[layer].sort(key=layer_sort_key)
                for i, ref in enumerate(layers[layer]):
                    position[ref] = i
    return layers
