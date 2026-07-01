from __future__ import annotations

import math

import cv2
import numpy as np

from schematic_generator.models import Pad


def components_traces(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Label trace mask components conservatively for electrical connectivity."""

    return cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=4)


def label_traces_near_pad(labels: np.ndarray, pad: Pad) -> int:
    """Return the nearest trace component touching the pad."""

    labels_pad = labels_traces_near_pad(labels, pad, limit=1)
    return labels_pad[0] if labels_pad else 0


def labels_traces_near_pad(
    labels: np.ndarray,
    pad: Pad,
    limit: int = 4,
    relaxed: bool = False,
) -> list[int]:
    """Return trace components touching the pad annulus.

    A through-hole pad can electrically join several separate mask components:
    one segment may enter the annular pad from the left and another may leave
    from the right, while the pad itself is not part of the trace mask. Returning
    all close components avoids splitting such a node into artificial nets.
    """

    x = int(round(pad.x))
    y = int(round(pad.y))
    if relaxed:
        radius = int(max(9, min(38, round(pad.radius * 2.25 + 4.0))))
    else:
        radius = int(max(6, min(24, round(pad.radius * 1.45 + 2.0))))
    x1, x2 = max(0, x - radius), min(labels.shape[1], x + radius + 1)
    y1, y2 = max(0, y - radius), min(labels.shape[0], y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return []

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance2 = (xx - x) ** 2 + (yy - y) ** 2
    disc = distance2 <= radius * radius
    cutout = labels[y1:y2, x1:x2]
    values, counters = np.unique(cutout[disc & (cutout > 0)], return_counts=True)
    if len(values) == 0:
        return []

    if relaxed:
        min_pixel = max(3, int(np.count_nonzero(disc) * 0.003))
        touch_border = max(7.0, min(34.0, pad.radius * 2.15 + 4.0))
    else:
        min_pixel = max(3, int(np.count_nonzero(disc) * 0.006))
        touch_border = max(5.0, min(22.0, pad.radius * 1.25 + 2.0))
    candidates: list[tuple[float, int, int]] = []
    for value, counter in zip(values, counters):
        label = int(value)
        pixels = disc & (cutout == label)
        pixel_count = int(counter)
        if pixel_count < min_pixel:
            continue
        min_distance = math.sqrt(float(np.min(distance2[pixels])))
        if min_distance <= touch_border:
            candidates.append((min_distance, -pixel_count, label))
    if not candidates:
        return []
    candidates.sort()
    maks_labels = limit if (relaxed or pad.radius >= 6.0) else 1
    return [label for _distance, _pixels, label in candidates[:maks_labels]]
