from __future__ import annotations

import math

import numpy as np

from schematic_generator.geometry.models import LogFn
from schematic_generator.models import Pad


def _transform_pads(pads: list[Pad], matrix: np.ndarray) -> list[Pad]:
    """Apply an affine transform to pad centers and scale radii by the transform area scale."""

    points = np.array([[p.x, p.y] for p in pads], dtype=np.float32)
    transformed_points = _transform_points(points, matrix)
    result: list[Pad] = []
    scale = math.sqrt(abs(float(np.linalg.det(matrix[:, :2])))) if matrix.shape == (2, 3) else 1.0
    for pad, (x, y) in zip(pads, transformed_points, strict=False):
        result.append(Pad(
            _pad_identifier(pad),
            pad.side,
            float(x),
            float(y),
            pad.radius * scale,
            _pad_confidence(pad),
            pad.net,
            pad.type,
            pad.status,
            pad.name,
        ))
    return result


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to an array of two-dimensional points."""

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.hstack([points, ones]) @ matrix.T


def _rescale_pads(pads: list[Pad], old_w: int, old_h: int, new_w: int, new_h: int) -> None:
    """Scale pad coordinates in place after resizing the image they belong to."""

    sx = new_w / max(1, old_w)
    sy = new_h / max(1, old_h)
    for pad in pads:
        pad.x *= sx
        pad.y *= sy
        pad.radius *= (sx + sy) / 2.0


def _order_points(points: np.ndarray) -> np.ndarray:
    """Return contour corners ordered as top-left, top-right, bottom-right, bottom-left."""

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(point_sum)],
            points[np.argmin(point_diff)],
            points[np.argmax(point_sum)],
            points[np.argmax(point_diff)],
        ],
        dtype=np.float32,
    )


def _pad_identifier(pad: Pad) -> str:
    """Read the side-local pad identifier."""

    return str(pad.identifier)


def _pad_confidence(pad: Pad) -> float:
    """Read pad confidence."""

    return float(pad.confidence)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Measure Euclidean distance between two image-space points."""

    return float(np.linalg.norm(a - b))


def _log(log: LogFn | None, text: str) -> None:
    """Forward a diagnostic message only when the caller provided a logger callback."""

    if log:
        log(text)
