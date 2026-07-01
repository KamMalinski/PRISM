from __future__ import annotations

import math

import cv2
import numpy as np

from schematic_generator.detection.common import _distance, _pad_confidence
from schematic_generator.models import Pad

def _pads_from_circles(gray: np.ndarray, side: str, min_r: int, max_r: int, min_dist: int) -> list[Pad]:
    """Create pad candidates from Hough circle detections."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    return [Pad("", side, float(x), float(y), float(radius), 0.75) for x, y, radius in np.round(circles[0]).astype(int)]

def _pads_from_contours(gray: np.ndarray, side: str, min_r: int, max_r: int) -> list[Pad]:
    """Create pad candidates from thresholded contours when circle detection is too sparse."""
    threshold_local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    threshold_dark = _threshold_by_percentile(gray, 25, invert=True)
    threshold = cv2.bitwise_or(threshold_local, threshold_dark)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pads: list[Pad] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < math.pi * min_r * min_r or area > math.pi * max_r * max_r * 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if min_r <= radius <= max_r:
            pads.append(Pad("", side, x, y, radius, min(0.7, circularity)))
    return pads

def _pads_from_dark_holes(gray: np.ndarray, side: str, min_r: int, max_r: int) -> list[Pad]:
    """Create pad candidates from dark circular holes on non-green boards."""
    threshold = min(78.0, max(24.0, float(np.percentile(gray, 6))))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pads: list[Pad] = []
    min_area = max(5.0, math.pi * (min_r * 0.55) ** 2)
    max_area = math.pi * (max_r * 0.85) ** 2
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.42:
            continue
        (x, y), hole_radius = cv2.minEnclosingCircle(contour)
        if hole_radius < max(1.5, min_r * 0.45) or hole_radius > max_r * 0.95:
            continue
        pad_radius = min(float(max_r), max(float(min_r), hole_radius * 2.15))
        pads.append(Pad("", side, x, y, pad_radius, min(0.68, max(0.42, circularity))))
    return pads

def _remove_duplicates(pads: list[Pad]) -> list[Pad]:
    """Keep the best pad candidate when multiple detections overlap the same physical pad."""
    result: list[Pad] = []
    for pad in sorted(pads, key=lambda entry: entry.confidence, reverse=True):
        is_duplicate = any(_distance(pad, other) < max(pad.radius, other.radius) for other in result)
        if not is_duplicate:
            result.append(pad)
    return result

def _threshold_by_percentile(image: np.ndarray, percentile: int, invert: bool) -> np.ndarray:
    """Create a binary mask by thresholding an image at a percentile value."""
    threshold = np.percentile(image, percentile)
    type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(image, threshold, 255, type)
    return mask.astype(np.uint8)
