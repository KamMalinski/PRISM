from __future__ import annotations

import cv2
import numpy as np

from schematic_generator.geometry.helpers import _distance, _log, _order_points
from schematic_generator.geometry.models import LogFn, RectificationResult


def rectify_board(image: np.ndarray, side: str, log: LogFn | None = None) -> RectificationResult:
    """Detect the visible PCB contour and warp it to a front-facing rectangular image."""

    _log(log, f"{side}: looking for the board contour for perspective correction.")
    points = _find_board_corners(image)
    if points is None:
        _log(log, f"{side}: no reliable board contour found; keeping original geometry.")
        return RectificationResult(image, np.eye(3, dtype=np.float32), "no contour correction", 0.0)

    points = _order_points(points)
    width_top = _distance(points[0], points[1])
    width_bottom = _distance(points[3], points[2])
    height_left = _distance(points[0], points[3])
    height_right = _distance(points[1], points[2])
    width = max(64, int(round(max(width_top, width_bottom))))
    height = max(64, int(round(max(height_left, height_right))))

    target_points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target_points)
    result = cv2.warpPerspective(image, matrix, (width, height))

    area = cv2.contourArea(points.astype(np.float32))
    confidence = min(1.0, max(0.0, area / float(image.shape[0] * image.shape[1])))
    _log(
        log,
        f"{side}: rectifying perspective to rectangle {width}x{height}, "
        f"contour coverage={confidence:.2f}.",
    )
    return RectificationResult(result, matrix, "perspective correction from contour", confidence)


def _find_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Approximate the largest board-colored region as four ordered perspective corners."""

    mask = _find_board_mask(image)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    if contour_area < image_area * 0.05:
        return None

    for factor in (0.015, 0.025, 0.04, 0.06):
        epsilon = factor * cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximation) == 4 and cv2.contourArea(approximation) >= contour_area * 0.65:
            return approximation.reshape(4, 2).astype(np.float32)

    rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
    if cv2.contourArea(rectangle.astype(np.float32)) >= image_area * 0.05:
        return rectangle.astype(np.float32)
    return None


def _find_board_mask(image: np.ndarray) -> np.ndarray | None:
    """Build a coarse mask for the PCB body using saturation, green hue, and LAB fallback cues."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width = image.shape[:2]
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    threshold_s = max(35, int(np.percentile(saturation, 55)))
    threshold_v_min = max(15, int(np.percentile(brightness, 4)))
    threshold_v_max = min(250, int(np.percentile(brightness, 99)))
    saturated_mask = (
        (saturation >= threshold_s)
        & (brightness >= threshold_v_min)
        & (brightness <= threshold_v_max)
    ).astype(np.uint8) * 255

    green_mask = (
        (hue >= 28)
        & (hue <= 100)
        & (saturation >= max(25, int(np.percentile(saturation, 35))))
        & (brightness >= threshold_v_min)
    ).astype(np.uint8) * 255

    channel_a = lab[:, :, 1]
    color_mask = cv2.bitwise_or(saturated_mask, green_mask)
    if np.count_nonzero(color_mask) < height * width * 0.03:
        # Fallback for boards with unusual colors: look for an area that differs from the background.
        distance_from_median = cv2.absdiff(channel_a, np.full_like(channel_a, int(np.median(channel_a))))
        _, color_mask = cv2.threshold(distance_from_median, 8, 255, cv2.THRESH_BINARY)

    kernel_size = max(9, min(height, width) // 45)
    if kernel_size % 2 == 0:
        kernel_size += 1
    large_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    if stats[index, cv2.CC_STAT_AREA] < height * width * 0.05:
        return None

    result = np.zeros((height, width), dtype=np.uint8)
    result[labels == index] = 255
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, large_kernel, iterations=1)
    return result
