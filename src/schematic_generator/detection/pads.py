from __future__ import annotations

import math

import cv2
import numpy as np

from schematic_generator.detection.colors import _has_non_green_saturated_soldermask
from schematic_generator.detection.common import LogFn, _log
from schematic_generator.detection.pad_sources import _pads_from_circles, _pads_from_contours, _pads_from_dark_holes, _remove_duplicates
from schematic_generator.models import Pad

def detect_pads(image: np.ndarray, side: str, log: LogFn | None = None) -> list[Pad]:
    """Detects candidate pads and holes with Hough circles and contour fallback."""

    _log(log, f"{side}: method cv2.cvtColor(BGR2GRAY) + cv2.medianBlur(ksize=5) for pad detection.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_dimension = min(gray.shape[:2])
    min_radius = max(3, min_dimension // 300)
    max_radius = max(8, min_dimension // 35)
    if (
        max(image.shape[:2]) / max(1, min_dimension) >= 2.5
        and min_dimension <= 320
        and _has_non_green_saturated_soldermask(image)
    ):
        max_radius = max(max_radius, min(16, int(round(min_dimension / 18))))
    min_dist = max_radius * 2
    _log(log, f"{side}: method cv2.HoughCircles dp=1.2, minDist={min_dist}, radius={min_radius}-{max_radius}.")

    pads = _pads_from_circles(gray, side, min_radius, max_radius, min_dist)
    _log(log, f"{side}: HoughCircles found {len(pads)} candidates.")
    if len(pads) < 4:
        _log(log, f"{side}: fallback: adaptiveThreshold + findContours + minEnclosingCircle.")
        contours = _pads_from_contours(gray, side, min_radius, max_radius)
        _log(log, f"{side}: fallback contourow dodal {len(contours)} candidates.")
        pads.extend(contours)
    if 4 <= len(pads) < 8 and _has_non_green_saturated_soldermask(image):
        _log(log, f"{side}: fallback: dark holes + circularity contour.")
        holes = _pads_from_dark_holes(gray, side, min_radius, max_radius)
        _log(log, f"{side}: dark-hole fallback added {len(holes)} candidates.")
        pads.extend(holes)

    pads = _remove_duplicates(pads)
    before_color_filter = len(pads)
    max_radius = max((pad.radius for pad in pads), default=0.0)
    if before_color_filter <= 6 or (before_color_filter <= 8 and max_radius >= 24.0):
        pads = _filter_pads_by_color(image, pads)
        _log(
            log,
            f"{side}: pad color filter rejected {before_color_filter - len(pads)} "
            "candidates silkscreen/soldermask candidates.",
        )
    else:
        _log(log, f"{side}: pad color filter skipped for {before_color_filter} candidates.")
    before_flat_filter = len(pads)
    pads = _filter_flat_non_copper_circles(image, pads)
    if len(pads) != before_flat_filter:
        _log(
            log,
            f"{side}: flat non-copper circle filter rejected "
            f"{before_flat_filter - len(pads)} candidates.",
        )
    pads.sort(key=lambda pad: (pad.y, pad.x))
    _log(log, f"{side}: radius-distance duplicate removal kept {len(pads)} pads.")

    for index, pad in enumerate(pads, start=1):
        pad.identifier = f"P{index:04d}"
    return pads[:500]

def _filter_pads_by_color(image: np.ndarray, pads: list[Pad]) -> list[Pad]:
    """Reject sparse pad candidates that look like silkscreen or soldermask rather than copper or a hole."""
    if not pads:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result: list[Pad] = []
    for pad in pads:
        if _is_local_pad_copper(hsv, pad):
            result.append(pad)
    return result

def _filter_flat_non_copper_circles(image: np.ndarray, pads: list[Pad]) -> list[Pad]:
    """Remove repeated flat circular artifacts when enough real copper pads are present for comparison."""
    if len(pads) < 9:
        return pads
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = [_local_pad_features(hsv, pad) for pad in pads]
    copper = sum(1 for feature in features if float(feature.get("copper_share", 0.0)) >= 0.08)
    if copper < max(4, int(math.ceil(len(pads) * 0.55))):
        return pads
    flat_count = sum(1 for feature in features if _is_flat_non_copper_circle(feature))
    if flat_count < max(3, int(math.ceil(len(pads) * 0.30))):
        return pads
    radii_copper = [
        pad.radius
        for pad, feature in zip(pads, features)
        if float(feature.get("copper_share", 0.0)) >= 0.08
    ]
    copper_radius_median = float(np.median(radii_copper)) if radii_copper else 0.0

    result = [
        pad
        for pad, feature in zip(pads, features)
        if not _is_small_flat_non_copper_circle(pad, feature, copper_radius_median)
    ]
    if len(result) < max(4, int(math.ceil(len(pads) * 0.45))):
        return pads
    return result

def _local_pad_features(hsv: np.ndarray, pad: Pad) -> dict[str, float | bool]:
    """Measure ring color, copper share, and center contrast around one pad candidate."""
    h, w = hsv.shape[:2]
    x = int(round(pad.x))
    y = int(round(pad.y))
    r = max(3, int(round(pad.radius)))
    margin = max(3, int(round(r * 0.35)))
    x1 = max(0, x - r - margin)
    x2 = min(w, x + r + margin + 1)
    y1 = max(0, y - r - margin)
    y2 = min(h, y + r + margin + 1)
    if x2 <= x1 or y2 <= y1:
        return {"has_ring": False, "copper_share": 0.0, "ring_contrast": 0.0}

    patch = hsv[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = np.sqrt((xx - pad.x) ** 2 + (yy - pad.y) ** 2)
    ring = (distance >= max(1.5, r * 0.48)) & (distance <= max(3.0, r * 1.18))
    center = distance <= max(2.0, r * 0.45)
    if not np.any(ring):
        return {"has_ring": False, "copper_share": 0.0, "ring_contrast": 0.0}

    h_ring = patch[:, :, 0][ring].astype(np.float32)
    s_ring = patch[:, :, 1][ring].astype(np.float32)
    v_ring = patch[:, :, 2][ring].astype(np.float32)
    v_center = patch[:, :, 2][center].astype(np.float32) if np.any(center) else np.array([], dtype=np.float32)
    yellow_copper = (
        (h_ring >= 8.0)
        & (h_ring <= 55.0)
        & (s_ring >= 55.0)
        & (v_ring >= 70.0)
    )
    copper_share = float(np.count_nonzero(yellow_copper)) / float(len(h_ring))
    contrast = (
        float(np.percentile(v_ring, 75)) - float(np.percentile(v_center, 25))
        if v_center.size
        else 0.0
    )
    return {
        "has_ring": True,
        "copper_share": copper_share,
        "ring_contrast": contrast,
    }

def _is_flat_non_copper_circle(feature: dict[str, float | bool]) -> bool:
    """Classify a local pad feature set as a flat circle without a copper ring."""
    if not bool(feature.get("has_ring", False)):
        return False
    return (
        float(feature.get("copper_share", 0.0)) < 0.02
        and float(feature.get("ring_contrast", 0.0)) < 20.0
    )

def _is_small_flat_non_copper_circle(
    pad: Pad,
    feature: dict[str, float | bool],
    copper_radius_median: float,
) -> bool:
    """Reject small non-copper circles that are much smaller than typical copper pads."""
    if copper_radius_median <= 0.0 or not _is_flat_non_copper_circle(feature):
        return False
    return pad.radius < copper_radius_median * 0.72

def _is_local_pad_copper(hsv: np.ndarray, pad: Pad) -> bool:
    """Check whether local HSV evidence around a candidate resembles copper, a hole, or a valid pad ring."""
    h, w = hsv.shape[:2]
    x = int(round(pad.x))
    y = int(round(pad.y))
    r = max(3, int(round(pad.radius)))
    margin = max(3, int(round(r * 0.35)))
    x1 = max(0, x - r - margin)
    x2 = min(w, x + r + margin + 1)
    y1 = max(0, y - r - margin)
    y2 = min(h, y + r + margin + 1)
    if x2 <= x1 or y2 <= y1:
        return False

    patch = hsv[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = np.sqrt((xx - pad.x) ** 2 + (yy - pad.y) ** 2)
    ring = (distance >= max(1.5, r * 0.48)) & (distance <= max(3.0, r * 1.18))
    center = distance <= max(2.0, r * 0.45)
    if not np.any(ring):
        return False

    h_ring = patch[:, :, 0][ring].astype(np.float32)
    s_ring = patch[:, :, 1][ring].astype(np.float32)
    v_ring = patch[:, :, 2][ring].astype(np.float32)
    s_center = patch[:, :, 1][center].astype(np.float32) if np.any(center) else np.array([], dtype=np.float32)
    v_center = patch[:, :, 2][center].astype(np.float32) if np.any(center) else np.array([], dtype=np.float32)

    yellow_copper = (
        (h_ring >= 8.0)
        & (h_ring <= 55.0)
        & (s_ring >= 55.0)
        & (v_ring >= 70.0)
    )
    copper_share = float(np.count_nonzero(yellow_copper)) / float(len(h_ring))
    bright_silkscreen = (
        float(np.median(s_ring)) < 55.0
        and float(np.percentile(v_ring, 75)) > 145.0
        and copper_share < 0.08
    )
    green_soldermask = (
        float(np.median(h_ring)) >= 55.0
        and float(np.median(h_ring)) <= 105.0
        and float(np.median(s_ring)) >= 45.0
        and copper_share < 0.10
    )
    dark_hole = bool(v_center.size and float(np.percentile(v_center, 25)) < 75.0)
    prawdopodobny_hole = bool(
        v_center.size
        and s_center.size
        and float(np.percentile(v_center, 35)) < 115.0
        and float(np.median(s_center)) < 85.0
    )
    contrast_ringia = (
        float(np.percentile(v_ring, 75)) - float(np.percentile(v_center, 25))
        if v_center.size
        else 0.0
    )
    if dark_hole or prawdopodobny_hole:
        return True
    if bright_silkscreen or green_soldermask:
        return False
    return copper_share >= 0.08
