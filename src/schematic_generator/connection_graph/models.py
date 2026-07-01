from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphNode:
    """Node stored in the diagnostic connection graph."""

    id: str
    type: str
    side: str = ""
    x: float | None = None
    y: float | None = None
    bbox: list[float] = field(default_factory=list)
    area: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    """Edge stored in the diagnostic connection graph."""

    source: str
    target: str
    type: str
    source_kind: str
    confidence: float
    active: bool = True
    reason: str = ""
    geometry: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
