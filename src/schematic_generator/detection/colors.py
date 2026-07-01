from __future__ import annotations

import math

import cv2
import numpy as np

from schematic_generator.detection.common import ColorProfile, LogFn, _convert_sample_bgr, _fmt3, _log

def analyze_colors_pcb(
    image: np.ndarray,
    log: LogFn | None = None,
    label: str = "",
    samples_colors: dict[str, tuple[int, int, int]] | None = None,
) -> ColorProfile:
    """Estimate soldermask and exposed copper colors from sampled image pixels."""

    prefix = f"{label}: " if label else ""
    _log(log, f"{prefix}PCB colors: cv2.cvtColor BGR->HSV/Lab, pixel sampling, cv2.kmeans(K=4).")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = image.shape[:2]
    sample_step = max(1, int(math.sqrt((h * w) / 25000)))
    sample_hsv = hsv[::sample_step, ::sample_step].reshape(-1, 3).astype(np.float32)
    sample_lab = lab[::sample_step, ::sample_step].reshape(-1, 3).astype(np.float32)
    if len(sample_hsv) > 30000:
        indices = np.linspace(0, len(sample_hsv) - 1, 30000, dtype=int)
        sample_hsv = sample_hsv[indices]
        sample_lab = sample_lab[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _compactness, labels, hsv_centers = cv2.kmeans(sample_hsv, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    lab_centers = np.array([sample_lab[labels == i].mean(axis=0) for i in range(4)], dtype=np.float32)
    counters = np.array([np.count_nonzero(labels == i) for i in range(4)], dtype=np.int32)

    candidates_mask = [
        i for i, c in enumerate(hsv_centers)
        if c[1] > 35 and c[2] > 35 and counters[i] > len(sample_hsv) * 0.05
    ]
    idx_mask = max(candidates_mask or range(4), key=lambda i: (counters[i], hsv_centers[i][1]))

    def score_copper(i: int) -> float:
        """Score one k-means cluster as exposed copper using color distance, hue, brightness, and saturation."""
        h_val, s_val, v_val = hsv_centers[i]
        distance_from_mask = float(np.linalg.norm(lab_centers[i] - lab_centers[idx_mask]))
        yellow_bonus = 40.0 if 12 <= h_val <= 45 else 0.0
        brightness_bonus = max(0.0, v_val - hsv_centers[idx_mask][2])
        return distance_from_mask + yellow_bonus + brightness_bonus * 0.5 + s_val * 0.1

    remaining = [i for i in range(4) if i != idx_mask]
    idx_copper = max(remaining, key=score_copper) if remaining else idx_mask
    solder_hsv = hsv_centers[idx_mask].astype(np.float32)
    solder_lab = lab_centers[idx_mask].astype(np.float32)
    copper_hsv = hsv_centers[idx_copper].astype(np.float32)
    copper_lab = lab_centers[idx_copper].astype(np.float32)

    if samples_colors:
        manual_trace_sample = samples_colors.get("trace_bgr")
        manual_background_sample = samples_colors.get("background_bgr")
        if manual_trace_sample:
            copper_hsv, copper_lab = _convert_sample_bgr(manual_trace_sample)
            _log(log, f"{prefix}colors PCB: using manual trace/copper sample BGR={tuple(int(x) for x in manual_trace_sample)}.")
        if manual_background_sample:
            solder_hsv, solder_lab = _convert_sample_bgr(manual_background_sample)
            _log(log, f"{prefix}colors PCB: using manual soldermask background sample BGR={tuple(int(x) for x in manual_background_sample)}.")

    threshold_lab = max(12.0, min(55.0, float(np.linalg.norm(copper_lab - solder_lab)) * 0.55))
    threshold_hsv = max(18.0, min(80.0, float(np.linalg.norm(copper_hsv - solder_hsv)) * 0.70))
    _log(
        log,
        f"{prefix}colors PCB: soldermask HSV={_fmt3(solder_hsv)}, "
        f"copper/pads HSV={_fmt3(copper_hsv)}, lab_threshold={threshold_lab:.1f}, hsv_threshold={threshold_hsv:.1f}, samples={len(sample_hsv)}.",
    )
    return {
        "solder_hsv": tuple(float(x) for x in solder_hsv),
        "solder_lab": tuple(float(x) for x in solder_lab),
        "copper_hsv": tuple(float(x) for x in copper_hsv),
        "copper_lab": tuple(float(x) for x in copper_lab),
        "lab_threshold": threshold_lab,
        "hsv_threshold": threshold_hsv,
        "samples": int(len(sample_hsv)),
    }

def _has_non_green_saturated_soldermask(image: np.ndarray) -> bool:
    """Detect whether the board background is saturated but outside the usual green soldermask hue range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    candidate = (s >= max(30.0, np.percentile(s, 45))) & (v >= 30.0) & (v <= 245.0)
    if int(np.count_nonzero(candidate)) < max(80, image.shape[0] * image.shape[1] // 30):
        return False
    h_med = float(np.median(hsv[:, :, 0][candidate]))
    s_med = float(np.median(hsv[:, :, 1][candidate]))
    return s_med >= 35.0 and not (35.0 <= h_med <= 105.0)

def _profile_needs_robust_color_segmentation(profile: ColorProfile) -> bool:
    """Decide whether color-based trace recovery should run for difficult non-green soldermask profiles."""
    solder_hsv = tuple(float(x) for x in profile["solder_hsv"])
    copper_hsv = tuple(float(x) for x in profile["copper_hsv"])
    solder_h, solder_s = solder_hsv[0], solder_hsv[1]
    non_green = solder_s >= 35.0 and not (35.0 <= solder_h <= 105.0)
    close_trace_hue = _hue_distance(solder_h, copper_hsv[0]) <= 42.0
    darker_trace = copper_hsv[2] <= solder_hsv[2] - 18.0 and copper_hsv[1] >= 55.0
    return non_green and (close_trace_hue or darker_trace)

def _hue_distance(a: float, b: float) -> float:
    """Measure circular HSV hue distance on OpenCV 0..180 hue scale."""
    difference = abs(float(a) - float(b))
    return min(difference, 180.0 - difference)
