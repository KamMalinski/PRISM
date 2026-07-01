from __future__ import annotations

import math

import cv2
import numpy as np

from schematic_generator.detection.colors import analyze_colors_pcb, _profile_needs_robust_color_segmentation
from schematic_generator.detection.common import ColorProfile, LogFn, _log
from schematic_generator.detection.trace_sources import _clean_trace_mask, _mask_color_copper, _mask_dominant_color_traces, _mask_traces_under_soldermask

def refine_trace_mask(
    image: np.ndarray,
    combination_count: int = 3,
    log: LogFn | None = None,
    label: str = "",
    samples_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Builds a trace mask by voting over color and classical filter configurations."""

    combination_count = max(1, min(100, int(combination_count)))
    prefix = f"{label}: " if label else ""
    _log(log, f"{prefix}building trace mask by voting over {combination_count} filter combinations.")
    _log(log, f"{prefix}method cv2.cvtColor: BGR->GRAY, BGR->HSV, BGR->Lab.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    profile = analyze_colors_pcb(image, log, label, samples_colors)

    votes = np.zeros(gray.shape, dtype=np.uint16)
    configurations = _filter_configurations(combination_count)
    for index, cfg in enumerate(configurations, start=1):
        _log(
            log,
            f"{prefix}filter {index}/{combination_count}: "
            f"CLAHE={cfg['clahe']:.1f}, Canny={cfg['canny1']}/{cfg['canny2']}, "
            f"adaptive={cfg['block']}, C={cfg['c']}, percentilee V={cfg['bright']}/{cfg['dark']} S={cfg['sat']}.",
        )
        variant = _mask_for_configuration(gray, hsv, lab, profile, cfg, log, prefix)
        votes += (variant > 0).astype(np.uint16)

    threshold_voting = max(1, int(math.ceil(combination_count * 0.35)))
    _log(log, f"{prefix}mask voting threshold: minimum {threshold_voting} votes.")
    if _profile_needs_robust_color_segmentation(profile):
        mask_dominant = _mask_dominant_color_traces(image, hsv, lab, profile, log, prefix)
    else:
        mask_dominant = np.zeros(gray.shape, dtype=np.uint8)
    if int(np.count_nonzero(mask_dominant)) > 0:
        _log(
            log,
            f"{prefix}dominant trace-color variant added "
            f"{int(np.count_nonzero(mask_dominant))} pixels przed czyszczeniem.",
        )
    mask_voted = np.where(votes >= threshold_voting, 255, 0).astype(np.uint8)
    _log(log, f"{prefix}cleanup method: morphology close/open + connectedComponentsWithStats.")
    mask_voted = _clean_trace_mask(mask_voted)
    if int(np.count_nonzero(mask_dominant)) == 0:
        return mask_voted
    mask = cv2.bitwise_or(mask_voted, mask_dominant)
    return _clean_trace_mask(mask)

def _filter_configurations(count: int) -> list[dict[str, int | float]]:
    """Generate deterministic threshold configurations used for voting-based trace segmentation."""
    clahe = [1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 5.0]
    canny = [(22, 75), (30, 100), (38, 115), (45, 135), (55, 160), (65, 185), (75, 210), (90, 240)]
    blocks = [21, 31, 41, 51, 61, 75, 91]
    constants = [-2, 0, 2, 4, 6, 8, 10, 12]
    bright_values = [52, 58, 62, 66, 70, 74, 78, 82]
    dark = [10, 14, 18, 22, 26, 30, 34, 38]
    sat = [18, 25, 32, 40, 48, 56, 65, 75]
    configurations: list[dict[str, int | float]] = []
    for i in range(count):
        configurations.append(
            {
                "clahe": clahe[i % len(clahe)],
                "canny1": canny[i % len(canny)][0],
                "canny2": canny[i % len(canny)][1],
                "block": blocks[(i // 2) % len(blocks)],
                "c": constants[(i // 3) % len(constants)],
                "bright": bright_values[(i // 4) % len(bright_values)],
                "dark": dark[(i // 5) % len(dark)],
                "sat": sat[(i // 6) % len(sat)],
            }
        )
    return configurations

def _mask_for_configuration(
    gray: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    profile: ColorProfile,
    cfg: dict[str, int | float],
    log: LogFn | None,
    prefix: str,
) -> np.ndarray:
    """Build one trace-mask vote from classical edges, adaptive thresholding, and color masks."""
    _log(log, f"{prefix}  CLAHE method: clipLimit={float(cfg['clahe']):.1f}, tileGridSize=(8,8).")
    clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe"]), tileGridSize=(8, 8)).apply(gray)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    mask_color = _mask_color_copper(hsv, lab, profile, cfg, log, prefix)
    mask_under_soldermask_mask = _mask_traces_under_soldermask(hsv, lab, profile, cfg, log, prefix)

    _log(log, f"{prefix}  HSV threshold method: bright/dark saturated areas by percentiles.")
    bright_saturated = (
        (brightness > np.percentile(brightness, int(cfg["bright"])))
        & (saturation > np.percentile(saturation, int(cfg["sat"])))
    ).astype(np.uint8) * 255

    dark_saturated = (
        (brightness < np.percentile(brightness, int(cfg["dark"])))
        & (saturation > np.percentile(saturation, int(cfg["sat"])))
    ).astype(np.uint8) * 255

    _log(log, f"{prefix}  cv2.Canny method: thresholds=({int(cfg['canny1'])},{int(cfg['canny2'])}) + dilate kernel=3x3.")
    edges = cv2.Canny(clahe, int(cfg["canny1"]), int(cfg["canny2"]))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    _log(log, f"{prefix}  adaptiveThreshold method: Gaussian block={int(cfg['block'])}, C={int(cfg['c'])}.")
    local_threshold = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        int(cfg["block"]),
        int(cfg["c"]),
    )
    local_threshold_edges = cv2.bitwise_and(local_threshold, cv2.dilate(edges, np.ones((5, 5), np.uint8)))
    classical_mask = cv2.bitwise_or(cv2.bitwise_or(bright_saturated, dark_saturated), local_threshold_edges)
    color_mask = cv2.bitwise_or(mask_color, mask_under_soldermask_mask)
    neighborhood_color = cv2.dilate(color_mask, np.ones((7, 7), np.uint8))
    return cv2.bitwise_or(color_mask, cv2.bitwise_and(classical_mask, neighborhood_color))
