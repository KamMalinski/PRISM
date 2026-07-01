from __future__ import annotations

import cv2
import numpy as np

from schematic_generator.detection.colors import analyze_colors_pcb
from schematic_generator.detection.common import LogFn, _log

def detect_plane_copper(
    image: np.ndarray,
    mask_traces: np.ndarray,
    log: LogFn | None = None,
    label: str = "",
    samples_colors: dict[str, tuple[int, int, int]] | None = None,
    has_groundplane: bool = True,
) -> np.ndarray:
    """Detects large copper/ground-plane regions distinct from soldermask."""

    prefix = f"{label}: " if label else ""
    if not has_groundplane:
        _log(log, f"{prefix}plane: detection disabled by user.")
        return np.zeros(mask_traces.shape, dtype=np.uint8)
    _log(log, f"{prefix}plane: method: color analysis + Lab distance from soldermask + connected components.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    profile = analyze_colors_pcb(image, log, label, samples_colors)
    solder_lab = np.array(profile["solder_lab"], dtype=np.float32)
    copper_lab = np.array(profile["copper_lab"], dtype=np.float32)
    solder_distance = np.linalg.norm(lab.astype(np.float32) - solder_lab, axis=2)
    copper_distance = np.linalg.norm(lab.astype(np.float32) - copper_lab, axis=2)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    solder_hsv = np.array(profile["solder_hsv"], dtype=np.float32)
    distance_h = np.minimum(np.abs(h - solder_hsv[0]), 180.0 - np.abs(h - solder_hsv[0]))

    threshold_solder = max(7.0, min(24.0, float(profile["lab_threshold"]) * 0.42))
    threshold_h = max(12.0, min(36.0, float(profile["hsv_threshold"]) * 0.60))
    threshold_s = max(18.0, float(solder_hsv[1]) * 0.40)
    _log(
        log,
        f"{prefix}plane: Lab threshold from soldermask={threshold_solder:.1f}, Hue={threshold_h:.1f}; "
        "rejecting bright silkscreen and outside-board background.",
    )
    under_soldermask_mask = (
        (solder_distance >= threshold_solder)
        & (distance_h <= threshold_h)
        & (s >= threshold_s)
    )
    exposed_copper = (copper_distance < solder_distance * 0.82) & (s >= np.percentile(s, 45))
    bright_silkscreen = (
        (v >= max(170.0, np.percentile(v, 86)))
        & (s <= max(38.0, np.percentile(s, 18)))
    )
    candidate = np.where((under_soldermask_mask | exposed_copper) & ~bright_silkscreen, 255, 0).astype(np.uint8)

    # Ground planes can be nearly uniform with only borders visible.
    # Close contours and fill large compact regions instead of relying only on edges.
    min_dimension = min(candidate.shape[:2])
    kernel_size = max(7, min(11, min_dimension // 85))
    if kernel_size % 2 == 0:
        kernel_size += 1
    _log(log, f"{prefix}plane: morphology close kernel={kernel_size}x{kernel_size}, open 5x5.")
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    # Trace mask helps seed thin borders; after closing, large regions become plane candidates.
    expanded_traces = cv2.dilate((mask_traces > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=1)
    candidate = cv2.bitwise_or(candidate, cv2.bitwise_and(expanded_traces, candidate))
    return _clean_plane_mask(candidate, log, prefix)

def _clean_plane_mask(mask: np.ndarray, log: LogFn | None, prefix: str) -> np.ndarray:
    """Keep large connected copper-plane candidates while rejecting full-image background components."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    result = np.zeros_like(mask)
    area_image = mask.size
    min_area = max(1200, int(area_image * 0.022))
    max_area = int(area_image * 0.90)
    accepted_count = 0
    _log(log, f"{prefix}plane: connectedComponentsWithStats min_area={min_area}, max_area={max_area}.")
    height, width = mask.shape[:2]
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        fill_ratio = area / max(1, width * height)
        touches_edges = x <= 1 and y <= 1 and x + width >= width - 1 and y + height >= height - 1
        image_coverage = area / max(1, area_image)
        if touches_edges and image_coverage >= 0.55:
            _log(
                log,
                f"{prefix}plane: rejecting component {index}, because it covers almost the whole image "
                f"({image_coverage:.1%}).",
            )
            continue
        if min_area <= area <= max_area and (width > 25 or height > 25) and fill_ratio >= 0.18:
            result[labels == index] = 255
            accepted_count += 1
    _log(log, f"{prefix}plane: accepted {accepted_count} large copper/plane regions.")
    return result
