from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from schematic_generator.models import Pad

LogFn = Callable[[str], None]


@dataclass(slots=True)
class RectificationResult:
    """Store the output image and transform metadata from perspective rectification."""

    image: np.ndarray
    matrix: np.ndarray
    description: str
    confidence: float


@dataclass(slots=True)
class AlignmentResult:
    """Store BOTTOM-to-TOP alignment output together with quality measurements."""

    bottom_image: np.ndarray
    bottom_pads: list[Pad]
    description: str
    mean_error: float
    count_matches: int
    percent_alignment: float = 0.0
