from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from schematic_generator.images import save_image

LogFn = Callable[[str], None]
OcrText = dict[str, float | int | str]


def read_texts(
    image: np.ndarray,
    output_folder: str | Path,
    label: str,
    path_tesseract: str | Path | None = None,
    pass_count: int = 1,
    log: LogFn | None = None,
) -> list[OcrText]:
    """Runs optional OCR over a PCB image and returns recognized text boxes."""

    prefix = f"{label}: "
    _log(log, f"{prefix}OCR: importing pytesseract.")
    try:
        import pytesseract
        from pytesseract import Output, TesseractNotFoundError
    except Exception as error:
        _log(log, f"{prefix}OCR: pytesseract unavailable ({error}); skipping OCR.")
        return []
    if path_tesseract:
        pytesseract.pytesseract.tesseract_cmd = str(path_tesseract)
        _log(log, f"{prefix}OCR: setting tesseract_cmd={path_tesseract}.")

    pass_count = max(1, min(20, int(pass_count)))
    variants = _preprocessing_variants(pass_count)
    _log(
        log,
        f"{prefix}OCR: searching for short PCB labels, variants={len(variants)}, "
        "whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.",
    )

    results: list[OcrText] = []
    for variant_index, variant in enumerate(variants, start=1):
        base = _prepare_for_ocr(image, variant, log, prefix, variant_index, len(variants))
        for rotation in (0, 90, 180, 270):
            prepared = _rotate_image(base, rotation)
            path_debug = Path(output_folder) / f"ocr_{label.lower()}_v{variant_index:02d}_{rotation}.png"
            save_image(path_debug, cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR))
            config = _config_tesseract(variant)
            try:
                _log(log, f"{prefix}OCR: pytesseract.image_to_data variant={variant_index}, rotation={rotation}, config='{config}'.")
                data = pytesseract.image_to_data(
                    prepared,
                    lang="eng",
                    config=config,
                    output_type=Output.DICT,
                )
            except TesseractNotFoundError:
                _log(log, f"{prefix}OCR: the Tesseract executable is missing; the Python package is installed but the engine is unavailable.")
                return []
            except Exception as error:
                _log(log, f"{prefix}OCR: engine error for variant {variant_index}, rotation {rotation}: {error}.")
                continue

            results.extend(_results_from_tesseract(data, label, image.shape[1], image.shape[0], rotation, variant, variant_index))

    results = _remove_duplicates_ocr(results)
    _log(log, f"{prefix}OCR: found {len(results)} labels after duplicate removal.")
    for entry in results:
        _log(
            log,
            f"{prefix}OCR text: '{entry['text']}' confidence={entry['confidence']} "
            f"box=({entry['x']},{entry['y']},{entry['w']},{entry['h']}), "
            f"rotation={entry['rotation']}, variant={entry.get('variant', '')}.",
        )
    return results


def save_texts_ocr(path: str | Path, texts: list[OcrText]) -> None:
    """Save OCR boxes as a tab-separated diagnostic file for later inspection."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["side\ttext\tconfidence\tx\ty\tw\th\trotation\tvariant"]
    for entry in texts:
        lines.append(
            f"{entry['side']}\t{entry['text']}\t{entry['confidence']}\t"
            f"{entry['x']}\t{entry['y']}\t{entry['w']}\t{entry['h']}\t{entry['rotation']}\t{entry.get('variant', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_for_ocr(
    image: np.ndarray,
    variant: dict[str, float | int | str | bool],
    log: LogFn | None,
    prefix: str,
    index: int,
    count: int,
) -> np.ndarray:
    """Convert the PCB image into one thresholded grayscale variant for Tesseract."""

    _log(
        log,
        f"{prefix}OCR preprocessing {index}/{count}: "
        f"scale={variant['scale']}, CLAHE={variant['clahe']}, blur={variant['blur']}, "
        f"threshold={variant['threshold']}, invert={variant['invert']}, psm={variant['psm']}.",
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = float(variant["scale"])
    if not math.isclose(scale, 1.0):
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=float(variant["clahe"]), tileGridSize=(8, 8)).apply(gray)
    if int(variant["blur"]) > 0:
        gray = cv2.GaussianBlur(gray, (int(variant["blur"]), int(variant["blur"])), 0)
    threshold = str(variant["threshold"])
    if threshold == "otsu":
        _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold == "otsu_inv":
        _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(variant["block"]), int(variant["c"]))
    if bool(variant["invert"]):
        binary = cv2.bitwise_not(binary)
    return binary


def _preprocessing_variants(count: int) -> list[dict[str, float | int | str | bool]]:
    """Return OCR preprocessing settings, cycling through proven variants when needed."""

    base_variants = [
        {"scale": 2.0, "clahe": 2.2, "blur": 3, "threshold": "otsu", "invert": False, "block": 31, "c": 5, "psm": 11},
        {"scale": 2.5, "clahe": 3.0, "blur": 3, "threshold": "otsu_inv", "invert": False, "block": 31, "c": 5, "psm": 11},
        {"scale": 3.0, "clahe": 2.6, "blur": 0, "threshold": "adaptive", "invert": False, "block": 31, "c": 7, "psm": 11},
        {"scale": 3.0, "clahe": 3.6, "blur": 3, "threshold": "adaptive", "invert": True, "block": 41, "c": 9, "psm": 6},
        {"scale": 2.0, "clahe": 4.0, "blur": 0, "threshold": "otsu", "invert": True, "block": 31, "c": 5, "psm": 13},
        {"scale": 4.0, "clahe": 2.0, "blur": 3, "threshold": "adaptive", "invert": False, "block": 51, "c": 11, "psm": 11},
        {"scale": 2.5, "clahe": 4.5, "blur": 0, "threshold": "otsu_inv", "invert": True, "block": 31, "c": 5, "psm": 6},
        {"scale": 3.5, "clahe": 3.2, "blur": 3, "threshold": "adaptive", "invert": True, "block": 61, "c": 13, "psm": 13},
    ]
    variants = []
    for i in range(count):
        variants.append(dict(base_variants[i % len(base_variants)]))
    return variants


def _config_tesseract(variant: dict[str, float | int | str | bool]) -> str:
    """Build the Tesseract command-line config for short PCB reference labels."""

    return (
        f"--oem 3 --psm {int(variant['psm'])} "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
        "-c classify_bln_numeric_mode=0 "
        "-c load_system_dawg=0 "
        "-c load_freq_dawg=0"
    )


def _results_from_tesseract(
    data: dict[str, list],
    label: str,
    width: int,
    height: int,
    rotation: int,
    variant: dict[str, float | int | str | bool],
    variant_index: int,
) -> list[OcrText]:
    """Filter raw Tesseract boxes and map accepted labels back to original image space."""

    results: list[OcrText] = []
    scale = float(variant["scale"])
    for i, text in enumerate(data.get("text", [])):
        text = _clean_text_ocr(str(text))
        if not _looks_like_pcb_label(text):
            continue
        try:
            confidence = float(data["conf"][i])
        except ValueError:
            confidence = -1.0
        if confidence < 18:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        if w <= 2 or h <= 2:
            continue
        x0, y0, w0, h0 = _undo_box_rotation(x, y, w, h, int(round(width * scale)), int(round(height * scale)), rotation, scale)
        results.append({
            "side": label,
            "text": text,
            "confidence": round(confidence, 1),
            "x": int(x0),
            "y": int(y0),
            "w": int(w0),
            "h": int(h0),
            "rotation": int(rotation),
            "variant": int(variant_index),
        })
    return results


def _clean_text_ocr(text: str) -> str:
    """Normalize OCR text to the compact character set used by PCB markings."""

    return "".join(ch for ch in text.upper().strip() if ch.isalnum() or ch in "_-")[:10]


def _looks_like_pcb_label(text: str) -> bool:
    """Return whether text resembles a reference designator or short PCB label."""

    if not (2 <= len(text) <= 10):
        return False
    if not re.fullmatch(r"[A-Z]{1,4}[0-9][A-Z0-9_-]{0,5}", text):
        return False
    return True


def _rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a preprocessed OCR image by a right-angle amount supported by OpenCV."""

    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _undo_box_rotation(x: int, y: int, w: int, h: int, width: int, height: int, rotation: int, scale: float) -> tuple[int, int, int, int]:
    """Transform a Tesseract box from rotated scaled space back to original pixels."""

    if rotation == 90:
        bx, by, bw, bh = y, height - x - w, h, w
    elif rotation == 180:
        bx, by, bw, bh = width - x - w, height - y - h, w, h
    elif rotation == 270:
        bx, by, bw, bh = width - y - h, x, h, w
    else:
        bx, by, bw, bh = x, y, w, h
    return (
        int(round(bx / scale)),
        int(round(by / scale)),
        max(1, int(round(bw / scale))),
        max(1, int(round(bh / scale))),
    )


def _remove_duplicates_ocr(texts: list[OcrText]) -> list[OcrText]:
    """Keep the highest-confidence box when repeated OCR passes find the same label."""

    result: list[OcrText] = []
    for entry in sorted(texts, key=lambda x: float(x["confidence"]), reverse=True):
        cx = int(entry["x"]) + int(entry["w"]) / 2.0
        cy = int(entry["y"]) + int(entry["h"]) / 2.0
        duplicate = False
        for saved in result:
            if entry["text"] != saved["text"] or entry["side"] != saved["side"]:
                continue
            saved_cx = int(saved["x"]) + int(saved["w"]) / 2.0
            saved_cy = int(saved["y"]) + int(saved["h"]) / 2.0
            if abs(cx - saved_cx) < max(int(entry["w"]), int(saved["w"])) and abs(cy - saved_cy) < max(int(entry["h"]), int(saved["h"])):
                duplicate = True
                break
        if not duplicate:
            result.append(entry)
    return sorted(result, key=lambda x: (str(x["side"]), int(x["y"]), int(x["x"])))


def _log(log: LogFn | None, text: str) -> None:
    """Send an OCR diagnostic message when a logger callback is available."""

    if log:
        log(text)
