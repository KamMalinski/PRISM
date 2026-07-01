from __future__ import annotations

from schematic_generator.schematic.optimization import (
    OptimizationConfig,
    OptimizationProgress,
    OptimizationResult,
    analyze_kicad_schematic,
    optimize_kicad_schematic,
    optimize_schematic_automatically,
)
from schematic_generator.schematic.validation import find_kicad_cli, validate_kicad_schematic

__all__ = [
    "OptimizationConfig",
    "OptimizationProgress",
    "OptimizationResult",
    "analyze_kicad_schematic",
    "find_kicad_cli",
    "optimize_kicad_schematic",
    "optimize_schematic_automatically",
    "validate_kicad_schematic",
]
