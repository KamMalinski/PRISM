from __future__ import annotations

from dataclasses import dataclass

from schematic_generator.models import Element

PinRef = tuple[Element, str]
WireSegment = tuple[str, tuple[float, float], tuple[float, float]]
TWO_PIN_TYPES = {"Resistor", "Capacitor", "Diode", "Inductor"}


@dataclass(slots=True)
class SymbolLayout:
    """Store schematic symbol position, size, and source metadata used by routing and reports."""
    ref: str
    x: float
    y: float
    width: float
    height: float
    source_x: float
    source_y: float
    kind: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return the symbol bounding box as left, top, right, bottom coordinates."""
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )
