from __future__ import annotations

from schematic_generator.schematic import optimization_autorouter as autorouter
from schematic_generator.schematic import optimization_connectivity as connectivity
from schematic_generator.schematic import optimization_drawing as drawing
from schematic_generator.schematic import optimization_geometry as geometry
from schematic_generator.schematic import optimization_models as models
from schematic_generator.schematic import optimization_output as output
from schematic_generator.schematic import optimization_parsing as parsing
from schematic_generator.schematic import optimization_placement as placement
from schematic_generator.schematic import optimization_public_api as public_api
from schematic_generator.schematic import optimization_routing as routing
from schematic_generator.schematic import optimization_routing_smoothing as routing_smoothing
from schematic_generator.schematic import optimization_routing_tree as routing_tree
from schematic_generator.schematic import optimization_serialization as serialization

_MODULES = [
    models,
    public_api,
    parsing,
    placement,
    connectivity,
    routing,
    routing_tree,
    routing_smoothing,
    autorouter,
    serialization,
    output,
    drawing,
    geometry,
]


def _patch_module_globals() -> None:
    """Share private optimization helpers across modules split from the original file."""

    merged = {}
    for module in _MODULES:
        merged.update({name: value for name, value in vars(module).items() if not name.startswith("__")})
    for module in _MODULES:
        vars(module).update(merged)


_patch_module_globals()

OptimizationConfig = models.OptimizationConfig
OptimizationProgress = models.OptimizationProgress
OptimizationResult = models.OptimizationResult
SchematicMetrics = models.SchematicMetrics
SchematicSymbol = models.SchematicSymbol
SchematicWire = models.SchematicWire
analyze_kicad_schematic = public_api.analyze_kicad_schematic
optimize_kicad_schematic = public_api.optimize_kicad_schematic
optimize_schematic_automatically = public_api.optimize_schematic_automatically

__all__ = [
    "OptimizationConfig",
    "OptimizationProgress",
    "OptimizationResult",
    "SchematicMetrics",
    "SchematicSymbol",
    "SchematicWire",
    "analyze_kicad_schematic",
    "optimize_kicad_schematic",
    "optimize_schematic_automatically",
]
