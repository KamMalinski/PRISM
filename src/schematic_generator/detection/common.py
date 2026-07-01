from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from schematic_generator.models import Pad

LogFn = Callable[[str], None]
ColorProfile = dict[str, tuple[float, float, float] | float | int]

def _distance(first: Pad, second: Pad) -> float:
    """Return Euclidean distance between two pads in image coordinates."""
    return math.hypot(first.x - second.x, first.y - second.y)

def _fmt3(values: np.ndarray) -> str:
    """Format the first three numeric values for compact log messages."""
    return "(" + ", ".join(f"{float(x):.1f}" for x in values[:3]) + ")"

def _convert_sample_bgr(sample: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert one BGR color sample into HSV and Lab vectors used by segmentation."""
    pixel = np.array([[sample]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return hsv, lab

def _pad_identifier(pad: Pad) -> str:
    """Read the side-local pad identifier."""
    return str(pad.identifier)

def _pad_confidence(pad: Pad) -> float:
    """Read pad confidence."""
    return float(pad.confidence)

def _log(log: LogFn | None, text: str) -> None:
    """Send a progress message to the optional logger when one is provided."""
    if log:
        log(text)
