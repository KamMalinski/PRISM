from __future__ import annotations

import cv2
import numpy as np

from schematic_generator.detection.common import ColorProfile, LogFn, _log

def _mask_color_copper(
    hsv: np.ndarray,
    lab: np.ndarray,
    profile: ColorProfile,
    cfg: dict[str, int | float],
    log: LogFn | None,
    prefix: str,
) -> np.ndarray:
    """Build a trace mask from pixels close to the exposed copper color profile."""
    copper_lab = np.array(profile["copper_lab"], dtype=np.float32)
    solder_lab = np.array(profile["solder_lab"], dtype=np.float32)
    threshold_lab = float(profile["lab_threshold"]) * (0.85 + 0.03 * (int(cfg["bright"]) - 52))
    copper_distance = np.linalg.norm(lab.astype(np.float32) - copper_lab, axis=2)
    mask_distance = np.linalg.norm(lab.astype(np.float32) - solder_lab, axis=2)
    similar_to_copper = (copper_distance <= threshold_lab) & (copper_distance < mask_distance * 0.92)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    copper_h = float(tuple(profile["copper_hsv"])[0])
    distance_h = np.minimum(np.abs(h.astype(np.float32) - copper_h), 180.0 - np.abs(h.astype(np.float32) - copper_h))
    threshold_h = max(10.0, float(profile["hsv_threshold"]) * 0.35)
    color_hsv = (distance_h <= threshold_h) & (s >= np.percentile(s, max(10, int(cfg["sat"]) - 8)))
    brightness_ok = v >= np.percentile(v, max(35, int(cfg["bright"]) - 12))
    mask = np.where(similar_to_copper | (color_hsv & brightness_ok), 255, 0).astype(np.uint8)
    _log(log, f"{prefix}  copper color mask method: Lab threshold={threshold_lab:.1f}, Hue threshold={threshold_h:.1f}, morphology open 3x3.")
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

def _mask_traces_under_soldermask(
    hsv: np.ndarray,
    lab: np.ndarray,
    profile: ColorProfile,
    cfg: dict[str, int | float],
    log: LogFn | None,
    prefix: str,
) -> np.ndarray:
    """Build a trace mask for tracks visible under soldermask by comparing Lab, hue, saturation, and value shifts."""
    solder_lab = np.array(profile["solder_lab"], dtype=np.float32)
    solder_hsv = np.array(profile["solder_hsv"], dtype=np.float32)
    mask_distance = np.linalg.norm(lab.astype(np.float32) - solder_lab, axis=2)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    distance_h = np.minimum(np.abs(h - solder_hsv[0]), 180.0 - np.abs(h - solder_hsv[0]))
    threshold_lab = max(5.5, float(profile["lab_threshold"]) * 0.30)
    threshold_h = max(12.0, min(34.0, float(profile["hsv_threshold"]) * 0.55))
    threshold_s = max(18.0, float(solder_hsv[1]) * 0.42)
    value_difference = np.abs(v - float(solder_hsv[2]))

    # In KiCad renders traces are often covered by soldermask and have a similar
    # hue to the board background, but different brightness and Lab values. They do not look like
    # exposed copper, so these green under-mask bands are detected separately.
    under_soldermask_mask = (
        (mask_distance >= threshold_lab)
        & (mask_distance <= max(24.0, float(profile["lab_threshold"]) * 1.8))
        & (distance_h <= threshold_h)
        & (s >= threshold_s)
        & ((value_difference >= 4.0) | (mask_distance >= threshold_lab * 1.35))
    )

    dark_holes = (
        (v <= min(55.0, np.percentile(v, max(4, int(cfg["dark"]) // 2))))
        & (s >= max(70.0, np.percentile(s, 55)))
    )
    bright_silkscreen = (
        (v >= max(170.0, np.percentile(v, 86)))
        & (s <= max(38.0, np.percentile(s, 18)))
    )
    mask = np.where((under_soldermask_mask | dark_holes) & ~bright_silkscreen, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    _log(
        log,
        f"{prefix}  under-soldermask trace mask method: Lab threshold={threshold_lab:.1f}, "
        f"Hue threshold={threshold_h:.1f}, morphology close/open.",
    )
    return mask

def _mask_dominant_color_traces(
    image: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    profile: ColorProfile,
    log: LogFn | None,
    prefix: str,
) -> np.ndarray:
    """Recover traces on difficult boards by combining dominant trace-color and under-soldermask evidence."""
    solder_lab = np.array(profile["solder_lab"], dtype=np.float32)
    copper_lab = np.array(profile["copper_lab"], dtype=np.float32)
    solder_hsv = np.array(profile["solder_hsv"], dtype=np.float32)
    copper_hsv = np.array(profile["copper_hsv"], dtype=np.float32)
    solder_distance = np.linalg.norm(lab.astype(np.float32) - solder_lab, axis=2)
    copper_distance = np.linalg.norm(lab.astype(np.float32) - copper_lab, axis=2)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    distance_h_solder = np.minimum(np.abs(h - solder_hsv[0]), 180.0 - np.abs(h - solder_hsv[0]))
    distance_h_copper = np.minimum(np.abs(h - copper_hsv[0]), 180.0 - np.abs(h - copper_hsv[0]))
    lab_difference = float(np.linalg.norm(copper_lab - solder_lab))

    threshold_copper = max(24.0, min(72.0, lab_difference * 0.95))
    threshold_h_copper = max(12.0, min(38.0, float(profile["hsv_threshold"]) * 0.55))
    threshold_h_solder = max(14.0, min(42.0, float(profile["hsv_threshold"]) * 0.70))
    threshold_s = max(18.0, float(solder_hsv[1]) * 0.25)

    near_samples_traces = (
        (copper_distance <= threshold_copper)
        & ((copper_distance <= solder_distance * 1.25) | (distance_h_copper <= threshold_h_copper))
        & (s >= max(12.0, float(copper_hsv[1]) * 0.35))
    )
    under_soldermask = (
        (distance_h_solder <= threshold_h_solder)
        & (solder_distance >= max(4.0, lab_difference * 0.10))
        & (solder_distance <= max(28.0, lab_difference * 1.35))
        & (s >= threshold_s)
        & (
            (np.abs(v - float(solder_hsv[2])) >= 2.0)
            | (np.abs(s - float(solder_hsv[1])) >= 6.0)
        )
    )
    bright_silkscreen = (
        (v >= max(168.0, np.percentile(v, 86)))
        & (s <= max(42.0, np.percentile(s, 18)))
    )
    outside_board_background = (s <= 18.0) & (v >= np.percentile(v, 82))
    mask = np.where((near_samples_traces | under_soldermask) & ~bright_silkscreen & ~outside_board_background, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = _filter_components_traces_by_shape(mask, image.shape[:2])
    _log(
        log,
        f"{prefix}  dominant trace-color method: Lab threshold={threshold_copper:.1f}, "
        f"trace Hue={threshold_h_copper:.1f}, soldermask Hue={threshold_h_solder:.1f}.",
    )
    return mask

def _filter_components_traces_by_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Suppress connected components whose shape is too small or too solid to be a likely trace."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    result = np.zeros_like(mask)
    area_image = max(1, shape[0] * shape[1])
    min_area = max(10, area_image // 30000)
    max_area = int(area_image * 0.30)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if width <= 1 or height <= 1:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        elongation = long_side / short_side
        fill_ratio = area / max(1, width * height)
        if elongation >= 1.7 or area >= 55 or fill_ratio <= 0.58:
            result[labels == index] = 255
    return result

def _clean_trace_mask(mask: np.ndarray) -> np.ndarray:
    """Clean trace masks with morphology and connected-component area filtering."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    min_area = max(12, mask.size // 20000)
    max_area = int(mask.size * 0.45)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            result[labels == index] = 255
    return result
