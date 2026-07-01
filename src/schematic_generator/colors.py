from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ColorCandidate:
    """One dominant BGR color candidate and its estimated image share."""

    bgr: tuple[int, int, int]
    share: float


@dataclass(frozen=True, slots=True)
class ColorSuggestions:
    """Suggested BGR samples for trace copper, board background, and dominant colors."""

    trace_bgr: tuple[int, int, int] | None
    background_bgr: tuple[int, int, int] | None
    dominant: list[ColorCandidate]


def find_suggestions_colors(image_bgr: np.ndarray, color_count: int = 5) -> ColorSuggestions:
    """Find dominant board colors and propose trace/copper and soldermask samples."""

    dominant_colors = dominant_colors_bgr(image_bgr, color_count)
    if not dominant_colors:
        return ColorSuggestions(None, None, [])

    hsv = _colors_to_hsv([candidate.bgr for candidate in dominant_colors])
    lab = _colors_to_lab([candidate.bgr for candidate in dominant_colors])
    shares = np.array([candidate.share for candidate in dominant_colors], dtype=np.float32)

    mask_candidates = [
        index
        for index, color_hsv in enumerate(hsv)
        if color_hsv[1] >= 30.0 and color_hsv[2] >= 35.0 and shares[index] >= 0.03
    ]
    mask_index = max(mask_candidates or range(len(dominant_colors)), key=lambda index: (shares[index], hsv[index][1]))
    solder_lab = lab[mask_index]
    solder_hsv = hsv[mask_index]

    def copper_score(index: int) -> float:
        """Score one dominant color as exposed copper or trace material."""

        if index == mask_index:
            return -1.0
        hue, saturation, value = hsv[index]
        lab_distance = float(np.linalg.norm(lab[index] - solder_lab))
        hue_distance_from_mask = min(abs(float(hue - solder_hsv[0])), 180.0 - abs(float(hue - solder_hsv[0])))
        under_mask_bonus = 28.0 if hue_distance_from_mask <= 24.0 and abs(float(value - solder_hsv[2])) >= 5.0 else 0.0
        yellow_copper_bonus = 38.0 if 10.0 <= hue <= 48.0 else 0.0
        brightness_bonus = max(0.0, float(value - solder_hsv[2])) * 0.35
        saturation_bonus = float(saturation) * 0.08
        white_silkscreen_penalty = 30.0 if saturation < 35.0 and value > 170.0 else 0.0
        dark_hole_penalty = 80.0 if value < 45.0 else 0.0
        rare_color_penalty = 10.0 if shares[index] < 0.01 else 0.0
        return (
            lab_distance
            + under_mask_bonus
            + yellow_copper_bonus
            + brightness_bonus
            + saturation_bonus
            - white_silkscreen_penalty
            - dark_hole_penalty
            - rare_color_penalty
        )

    remaining_indices = [index for index in range(len(dominant_colors)) if index != mask_index]
    copper_index = max(remaining_indices, key=copper_score) if remaining_indices else mask_index
    return ColorSuggestions(
        trace_bgr=dominant_colors[copper_index].bgr if dominant_colors else None,
        background_bgr=dominant_colors[mask_index].bgr if dominant_colors else None,
        dominant=dominant_colors,
    )


def dominant_colors_bgr(image_bgr: np.ndarray, color_count: int = 5) -> list[ColorCandidate]:
    """Return the most visible BGR colors, sorted by estimated image share."""

    if image_bgr.size == 0:
        return []
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("Image must be a three-channel BGR matrix.")

    pixels = _sample_board_pixels(image_bgr)
    if len(pixels) == 0:
        return []

    count = max(1, min(int(color_count), len(pixels), 8))
    cv2.setRNGSeed(12345)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.3)
    _compactness, labels, centers = cv2.kmeans(
        pixels.astype(np.float32),
        count,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(-1)
    raw_candidates: list[ColorCandidate] = []
    for index in range(count):
        pixel_count = int(np.count_nonzero(labels == index))
        if pixel_count == 0:
            continue
        bgr = tuple(int(round(float(x))) for x in centers[index][:3])
        bgr = tuple(max(0, min(255, x)) for x in bgr)
        raw_candidates.append(ColorCandidate(bgr=bgr, share=pixel_count / len(pixels)))
    raw_candidates.sort(key=lambda candidate: candidate.share, reverse=True)

    merged_candidates: list[ColorCandidate] = []
    for color in raw_candidates:
        for index, existing in enumerate(merged_candidates):
            if _distance_bgr(color.bgr, existing.bgr) <= 8.0:
                combined_share = color.share + existing.share
                bgr = tuple(
                    int(round((existing.bgr[i] * existing.share + color.bgr[i] * color.share) / combined_share))
                    for i in range(3)
                )
                merged_candidates[index] = ColorCandidate(bgr=bgr, share=combined_share)
                break
        else:
            merged_candidates.append(color)
    merged_candidates.sort(key=lambda candidate: candidate.share, reverse=True)
    return merged_candidates[:color_count]


def _sample_board_pixels(image_bgr: np.ndarray) -> np.ndarray:
    """Downsample likely board pixels and reject highlights, shadows, and silkscreen-heavy pixels."""

    height, width = image_bgr.shape[:2]
    step = max(1, int(math.sqrt((height * width) / 45000)))
    image_sample = image_bgr[::step, ::step, :3]
    board_mask = _green_board_mask(image_sample)
    if board_mask is not None:
        sample = image_sample[board_mask > 0]
    else:
        sample = image_sample.reshape(-1, 3)
    if len(sample) > 50000:
        indices = np.linspace(0, len(sample) - 1, 50000, dtype=int)
        sample = sample[indices]

    hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    pixel_filter = (value > 18) & (value < 248) & ((saturation > 18) | (value < 215))
    if int(np.count_nonzero(pixel_filter)) >= min(800, max(40, len(sample) // 12)):
        sample = sample[pixel_filter]
    return sample.astype(np.float32)


def _green_board_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """Build a rough mask for the main green soldermask region, when such a region is present."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    candidate = (
        (hue >= 32)
        & (hue <= 105)
        & (saturation >= max(28, int(np.percentile(saturation, 42))))
        & (value >= max(20, int(np.percentile(value, 3))))
        & (value <= min(245, int(np.percentile(value, 99))))
    ).astype(np.uint8)
    if int(np.count_nonzero(candidate)) < max(80, image_bgr.shape[0] * image_bgr.shape[1] // 25):
        return None

    kernel = np.ones((5, 5), np.uint8)
    candidate = cv2.morphologyEx(candidate * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    if count <= 1:
        return None
    areas = statistics[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    if int(statistics[index, cv2.CC_STAT_AREA]) < max(80, image_bgr.shape[0] * image_bgr.shape[1] // 30):
        return None
    result = np.zeros(candidate.shape, dtype=np.uint8)
    result[labels == index] = 255
    return cv2.dilate(result, np.ones((5, 5), np.uint8), iterations=1)


def _colors_to_hsv(colors_bgr: list[tuple[int, int, int]]) -> np.ndarray:
    """Convert a list of BGR colors into HSV float rows."""

    pixels = np.array(colors_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)


def _colors_to_lab(colors_bgr: list[tuple[int, int, int]]) -> np.ndarray:
    """Convert a list of BGR colors into CIE Lab float rows."""

    pixels = np.array(colors_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _distance_bgr(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Return Euclidean distance between two BGR colors."""

    return float(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)) ** 0.5)
