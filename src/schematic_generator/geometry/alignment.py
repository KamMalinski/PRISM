from __future__ import annotations

import math
import time

import cv2
import numpy as np

from schematic_generator.geometry.helpers import _distance, _log, _rescale_pads, _transform_pads, _transform_points
from schematic_generator.geometry.models import AlignmentResult, LogFn
from schematic_generator.images import align_size
from schematic_generator.models import Pad


def align_bottom_to_top(
    top_image: np.ndarray,
    bottom_image: np.ndarray,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    alignment_threshold: float = 0.5,
    time_limit_s: float = 30.0,
    log: LogFn | None = None,
) -> AlignmentResult:
    """Align the BOTTOM image and its pads to TOP coordinates using detected drill holes."""

    h, w = top_image.shape[:2]
    if bottom_image.shape[:2] != top_image.shape[:2]:
        _log(log, f"BOTTOM: scaling after rectification to TOP size {w}x{h}.")
        old_h, old_w = bottom_image.shape[:2]
        bottom_image = align_size(bottom_image, (w, h))
        _rescale_pads(bottom_pads, old_w, old_h, w, h)

    if len(top_pads) < 4 or len(bottom_pads) < 4:
        _log(log, "Geometry: too few holes for reliable TOP/BOTTOM alignment.")
        return AlignmentResult(bottom_image, bottom_pads, "no alignment - too few holes", math.inf, 0)

    top_points = np.array([[p.x, p.y] for p in top_pads], dtype=np.float32)
    bottom_points = np.array([[p.x, p.y] for p in bottom_pads], dtype=np.float32)
    best = _select_hole_transform(
        top_points,
        bottom_points,
        (w, h),
        max(0.01, min(1.0, alignment_threshold)),
        max(1.0, time_limit_s),
        log,
    )
    if best is None:
        _log(log, "Geometry: could not determine a stable affine matrix.")
        return AlignmentResult(bottom_image, bottom_pads, "no stable matrix", math.inf, 0)

    name, matrix, error, count, percent = best
    _log(
        log,
        f"Geometry: best BOTTOM -> TOP alignment: {name}, "
        f"pairs={count}, coverage={percent:.0%}, error={error:.1f}px.",
    )
    image_result = cv2.warpAffine(bottom_image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    pads_result = _transform_pads(bottom_pads, matrix)
    return AlignmentResult(image_result, pads_result, name, error, count, percent)


def _select_hole_transform(
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    size: tuple[int, int],
    alignment_threshold: float,
    time_limit_s: float,
    log: LogFn | None,
) -> tuple[str, np.ndarray, float, int, float] | None:
    """Try coarse orientation transforms first, then search hole-pair affine candidates."""

    w, h = size
    start = time.monotonic()
    denominator = max(1, min(len(top_points), len(bottom_points)))
    candidates = [
        ("no mirror", np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)),
        ("horizontal mirror", np.array([[-1, 0, w - 1], [0, 1, 0]], dtype=np.float32)),
        ("vertical mirror", np.array([[1, 0, 0], [0, -1, h - 1]], dtype=np.float32)),
        ("rotate 180", np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=np.float32)),
        ("rotate 90", cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 90, 1.0).astype(np.float32)),
        ("rotate -90", cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -90, 1.0).astype(np.float32)),
    ]

    best: tuple[str, np.ndarray, float, int, float] | None = None
    tolerance = max(18.0, min(w, h) * 0.035)
    for name, base_matrix in candidates:
        best = _check_candidate(name, base_matrix, top_points, bottom_points, tolerance, denominator, best)
        if best and best[4] >= alignment_threshold:
            _log(log, f"Geometry: threshold {alignment_threshold:.0%} reached by '{name}'.")
            return best

    top_sample = _sample_points(top_points, 70)
    bottom_sample = _sample_points(bottom_points, 70)
    top_pairs = _point_pairs(top_sample, min(w, h), 900)
    bottom_pairs = _point_pairs(bottom_sample, min(w, h), 900)
    _log(
        log,
        f"Geometry: searching transform from hole pairs, "
        f"threshold={alignment_threshold:.0%}, limit={time_limit_s:.0f}s.",
    )

    counter = 0
    for bottom_first, bottom_second, bottom_distance in bottom_pairs:
        if time.monotonic() - start > time_limit_s:
            break
        for top_first, top_second, top_distance in top_pairs:
            if time.monotonic() - start > time_limit_s:
                break
            scale = top_distance / max(1.0, bottom_distance)
            if not 0.45 <= scale <= 2.2:
                continue
            for swapped in (False, True):
                target_first, target_second = (top_second, top_first) if swapped else (top_first, top_second)
                for mirrored in (False, True):
                    matrix = _matrix_from_pairs(
                        bottom_first,
                        bottom_second,
                        target_first,
                        target_second,
                        mirrored,
                    )
                    if matrix is None:
                        continue
                    counter += 1
                    best = _check_candidate(
                        "hole-pair search",
                        matrix,
                        top_points,
                        bottom_points,
                        tolerance,
                        denominator,
                        best,
                    )
                    if best and best[4] >= alignment_threshold:
                        _log(
                            log,
                            f"Geometry: threshold {alignment_threshold:.0%} reached after {counter} attempts.",
                        )
                        return best

    if best:
        _log(
            log,
            f"Geometry: threshold not reached {alignment_threshold:.0%}; "
            f"best coverage={best[4]:.0%}.",
        )
    return best


def _check_candidate(
    name: str,
    matrix: np.ndarray,
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    tolerance: float,
    denominator: int,
    best: tuple[str, np.ndarray, float, int, float] | None,
) -> tuple[str, np.ndarray, float, int, float] | None:
    """Score one transform and optionally refine it with RANSAC over matched holes."""

    alignment = _score_transform(top_points, bottom_points, matrix, tolerance)
    if alignment is None:
        return best

    pairs_src, pairs_dst, error, count = alignment
    candidate = matrix
    if count >= 4:
        refined_matrix, inliers = cv2.estimateAffine2D(
            pairs_src,
            pairs_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=tolerance,
            maxIters=1500,
            confidence=0.98,
            refineIters=10,
        )
        if refined_matrix is not None and inliers is not None and int(inliers.sum()) >= 4:
            refined_alignment = _score_transform(top_points, bottom_points, refined_matrix.astype(np.float32), tolerance)
            if refined_alignment is not None and (refined_alignment[3], -refined_alignment[2]) >= (count, -error):
                pairs_src, pairs_dst, error, count = refined_alignment
                candidate = refined_matrix.astype(np.float32)

    percent = count / denominator
    result = (name, candidate.astype(np.float32), error, count, percent)
    if best is None or (result[3], -result[2]) > (best[3], -best[2]):
        return result
    return best


def _score_transform(
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    matrix: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Match transformed BOTTOM holes to nearest TOP holes and return inlier statistics."""

    transformed_bottom_points = _transform_points(bottom_points, matrix)
    indices, distances = _distances_to_nearest(transformed_bottom_points, top_points)
    order = np.argsort(distances)
    used_top_indices: set[int] = set()
    src = []
    dst = []
    errors = []
    for bottom_index in order:
        distance = float(distances[bottom_index])
        top_index = int(indices[bottom_index])
        if distance > tolerance or top_index in used_top_indices:
            continue
        used_top_indices.add(top_index)
        src.append(bottom_points[int(bottom_index)])
        dst.append(top_points[top_index])
        errors.append(distance)

    if len(src) < 3:
        return None
    return (
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        float(np.mean(errors)),
        len(src),
    )


def _sample_points(points: np.ndarray, limit: int) -> np.ndarray:
    """Keep a bounded set of central and outer points so pair search stays fast."""

    if len(points) <= limit:
        return points
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    distances = np.linalg.norm(points - center, axis=1)
    angle_indices = np.argsort(angles)
    selected = set(np.linspace(0, len(angle_indices) - 1, limit // 2, dtype=int))
    indices = {int(angle_indices[i]) for i in selected}
    indices.update(int(i) for i in np.argsort(distances)[-(limit // 2):])
    return points[sorted(indices)[:limit]]


def _point_pairs(points: np.ndarray, min_dimension: int, limit: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Build the longest useful point pairs used as anchors for affine candidates."""

    min_distance = max(12.0, min_dimension * 0.035)
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = _distance(points[i], points[j])
            if distance >= min_distance:
                pairs.append((points[i], points[j], distance))
    pairs.sort(key=lambda p: p[2], reverse=True)
    if len(pairs) <= limit:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, limit, dtype=int)
    return [pairs[int(i)] for i in indices]


def _matrix_from_pairs(
    src1: np.ndarray,
    src2: np.ndarray,
    dst1: np.ndarray,
    dst2: np.ndarray,
    mirrored: bool,
) -> np.ndarray | None:
    """Create an affine matrix from two matched hole pairs and optional perpendicular mirroring."""

    source_vector = src2 - src1
    target_vector = dst2 - dst1
    if np.linalg.norm(source_vector) < 1.0 or np.linalg.norm(target_vector) < 1.0:
        return None

    source_perpendicular = np.array([-source_vector[1], source_vector[0]], dtype=np.float32)
    target_perpendicular = np.array([-target_vector[1], target_vector[0]], dtype=np.float32)
    if mirrored:
        target_perpendicular = -target_perpendicular

    src = np.array([src1, src2, src1 + source_perpendicular], dtype=np.float32)
    dst = np.array([dst1, dst2, dst1 + target_perpendicular], dtype=np.float32)
    return cv2.getAffineTransform(src, dst).astype(np.float32)


def _distances_to_nearest(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest destination index and distance for each source point."""

    differences = src[:, None, :] - dst[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    indices = np.argmin(distances, axis=1)
    return indices, distances[np.arange(len(src)), indices]
