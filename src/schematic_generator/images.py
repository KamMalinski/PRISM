from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


def load_image(path: str | Path) -> np.ndarray:
    """Load an image through a byte buffer so OpenCV handles Unicode filesystem paths."""

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot load image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an OpenCV image through an encoded buffer to preserve Unicode path support."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Cannot save image: {path}")
    buffer.tofile(str(path))


def align_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image to the requested ``(width, height)`` while keeping BGR channel order."""

    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def assess_sharpness(image: np.ndarray) -> float:
    """Estimate focus quality with the variance of the grayscale Laplacian response."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_image_quality(image: np.ndarray) -> dict[str, float | int | str]:
    """Return lightweight input-image quality metrics for logs and project metadata."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = assess_sharpness(image)
    dark_fraction = float(np.mean(gray <= 8))
    bright_fraction = float(np.mean(gray >= 247))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    score = 100.0

    if sharpness < 60:
        score -= 35
    elif sharpness < 120:
        score -= 18
    if dark_fraction > 0.08:
        score -= min(25.0, dark_fraction * 180)
    if bright_fraction > 0.08:
        score -= min(25.0, bright_fraction * 180)
    if contrast < 45:
        score -= 20

    score = max(0.0, min(100.0, score))
    if score >= 75:
        rating = "good"
    elif score >= 45:
        rating = "medium"
    else:
        rating = "poor"

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "sharpness_laplacian": round(sharpness, 2),
        "dark_pixels_percent": round(dark_fraction * 100.0, 2),
        "overexposed_pixels_percent": round(bright_fraction * 100.0, 2),
        "contrast_p5_p95": round(contrast, 2),
        "quality_score": round(score, 1),
        "rating": rating,
    }


def create_tk_thumbnail(path: str | Path, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    """Create a Tkinter-compatible thumbnail; the caller must keep the object alive."""

    image = Image.open(path)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def convert_mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask into a three-channel BGR image for saving and preview rendering."""

    return cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
