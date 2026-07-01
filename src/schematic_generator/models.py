from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Pad:
    """Solder pad, hole, or via detected on one side of the PCB."""

    identifier: str
    side: str
    x: float
    y: float
    radius: float
    confidence: float
    net: str = ""
    type: str = "pad"
    status: str = "auto"
    name: str = ""

    @property
    def node(self) -> str:
        """Return the side-qualified node name used in graph and netlist data."""

        return f"{self.side}:{self.identifier}"


@dataclass(slots=True)
class HolePair:
    """Candidate match for the same physical hole seen from both board sides."""

    pad_top: str
    pad_bottom: str
    distance: float
    confidence: float


@dataclass(slots=True)
class Net:
    """Logical electrical connection built from pads and vias."""

    name: str
    pads: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Element:
    """Detected or manually assigned schematic component with pin-to-net mapping."""

    ref: str
    type: str
    value: str
    footprint: str
    x: float
    y: float
    rotation: float
    pins: dict[str, str] = field(default_factory=dict)
    pin_descriptions: dict[str, str] = field(default_factory=dict)
    pin_pad_nodes: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    decision_source: str = "unknown"
    decision_score: float = 0.0
    decision_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManualCorrection:
    """One user correction stored independently from automatic reconstruction output."""

    identifier: str
    type: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    created: str = ""


@dataclass(slots=True)
class Problem:
    """Issue that should be reviewed or resolved by the user."""

    identifier: str
    type: str
    category: str
    severity: str
    description: str
    status: str = "open"
    related: dict[str, list[str]] = field(default_factory=dict)
    positions: dict[str, list[float]] = field(default_factory=dict)
    suggestion: str = ""


@dataclass(slots=True)
class AnalysisReport:
    """Paths and summary statistics returned after a reconstruction run."""

    output_folder: str
    count_pads_top: int
    count_pads_bottom: int
    count_pairs_holes: int
    count_nets: int
    count_elements: int
    file_netlist: str
    file_devices: str
    file_kicad: str
    file_preview: str
