from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from schematic_generator.detection_facade import (
    detect_pads,
    detect_plane_copper,
    find_pairs_holes,
    refine_trace_mask,
)
from schematic_generator.diagnostics_facade import save_diagnostics
from schematic_generator.geometry_facade import RectificationResult, align_bottom_to_top, rectify_board
from schematic_generator.images import (
    align_size,
    assess_image_quality,
    convert_mask_to_bgr,
    load_image,
    save_image,
)
from schematic_generator.kicad_facade import save_kicad_schematic, save_schematic_preview_png
from schematic_generator.models import AnalysisReport, HolePair, Pad
from schematic_generator.netlist_facade import (
    build_nets,
    create_elements,
    save_devices,
    save_netlist,
)
from schematic_generator.ocr import read_texts, save_texts_ocr as save_ocr_texts
from schematic_generator.preview import (
    draw_alignment_holes,
    draw_connections_colored,
    draw_preview,
    draw_thick_traces,
    overlay_bottom_on_top,
)
from schematic_generator.schematic_facade import optimize_schematic_automatically
from schematic_generator.schematic_facade import validate_kicad_schematic

LogFn = Callable[[str], None]


def run_analysis(
    top_path: str | Path,
    bottom_path: str | Path,
    base_folder: str | Path,
    filter_combination_count: int = 3,
    alignment_threshold: float = 0.5,
    schematic_name: str = "",
    tesseract_path: str | Path | None = None,
    color_samples: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    has_groundplane: bool = True,
    component_solver_mode: str | None = None,
    contact_solver_mode: str | None = None,
    log: LogFn | None = None,
) -> AnalysisReport:
    """Run the complete reconstruction pipeline for TOP and BOTTOM PCB images."""

    _log(log, "1/8 Creating the output folder.")
    output_folder = _create_output_folder(base_folder, schematic_name)
    work_folder = output_folder / "work"
    work_folder.mkdir(parents=True, exist_ok=True)
    _copy_original_image(top_path, output_folder, "TOP", log)
    _copy_original_image(bottom_path, output_folder, "BOTTOM", log)
    _log(log, f"Output folder: {output_folder}")
    _log(log, f"Work folder: {work_folder}")

    _log(log, "2/8 Loading TOP and BOTTOM images.")
    top_image = load_image(top_path)
    bottom_image = load_image(bottom_path)
    top_quality = _normalize_image_quality(assess_image_quality(top_image))
    bottom_quality = _normalize_image_quality(assess_image_quality(bottom_image))
    _log(log, f"TOP: {top_image.shape[1]}x{top_image.shape[0]} px.")
    _log(log, f"BOTTOM: {bottom_image.shape[1]}x{bottom_image.shape[0]} px.")
    _log(
        log,
        f"TOP quality={top_quality['rating']} ({top_quality['quality_score']}/100), "
        f"BOTTOM quality={bottom_quality['rating']} ({bottom_quality['quality_score']}/100).",
    )

    _log(log, "3/8 Rectifying each image to the board rectangle.")
    top_rectification = _rectify_with_pad_validation(top_image, "TOP", log)
    bottom_rectification = _rectify_with_pad_validation(bottom_image, "BOTTOM", log)
    top_image = top_rectification.image
    bottom_image = bottom_rectification.image

    height, width = top_image.shape[:2]
    if bottom_image.shape[:2] != top_image.shape[:2]:
        _log(log, "BOTTOM: rectified size differs from TOP, applying initial resize.")
        bottom_image = align_size(bottom_image, (width, height))

    _log(log, "4/8 Detecting holes and pads needed for side alignment.")
    top_pads = detect_pads(top_image, "TOP", log)
    bottom_pads = detect_pads(bottom_image, "BOTTOM", log)

    _log(
        log,
        "5/8 Aligning BOTTOM to TOP using hole geometry "
        f"(threshold {alignment_threshold:.0%}, 30s limit).",
    )
    alignment = align_bottom_to_top(
        top_image,
        bottom_image,
        top_pads,
        bottom_pads,
        alignment_threshold=alignment_threshold,
        time_limit_s=30.0,
        log=log,
    )
    bottom_image = alignment.bottom_image
    bottom_pads = alignment.bottom_pads
    hole_pairs = find_pairs_holes(top_pads, bottom_pads, log)
    top_pads, bottom_pads, hole_pairs = _fill_missing_via_pads(top_pads, bottom_pads, hole_pairs, log)

    _log(log, "6/8 Building trace masks with multi-threshold filtering and voting.")
    top_samples = _samples_for_side(color_samples, "TOP", "BOTTOM")
    bottom_samples = _samples_for_side(color_samples, "BOTTOM", "TOP")
    expand_non_green_contacts = (
        _has_non_green_soldermask_sample(top_samples)
        or _has_non_green_soldermask_sample(bottom_samples)
    )
    if expand_non_green_contacts:
        _log(log, "6/8 Non-green soldermask detected: enabling extra horizontal trace contacts near THT pads.")
    top_mask = refine_trace_mask(top_image, filter_combination_count, log, "TOP", top_samples)
    bottom_mask = refine_trace_mask(bottom_image, filter_combination_count, log, "BOTTOM", bottom_samples)
    _log(log, "6/8 Detecting large copper pours/groundplane regions and separating them from soldermask.")
    top_plane = detect_plane_copper(top_image, top_mask, log, "TOP", top_samples, has_groundplane)
    bottom_plane = detect_plane_copper(bottom_image, bottom_mask, log, "BOTTOM", bottom_samples, has_groundplane)

    _log(log, "7/8 Running OCR, building nets, and building components.")
    ocr_pass_count = max(1, int(round(filter_combination_count / 5)))
    _log(log, f"7/8 OCR preprocessing variants per rotation: {ocr_pass_count}.")
    top_ocr_texts = read_texts(top_image, work_folder, "TOP", tesseract_path, ocr_pass_count, log)
    bottom_ocr_texts = read_texts(bottom_image, work_folder, "BOTTOM", tesseract_path, ocr_pass_count, log)
    ocr_texts = [*top_ocr_texts, *bottom_ocr_texts]

    resolved_contact_solver_mode = (
        contact_solver_mode or os.environ.get("SCHEMATIC_GENERATOR_CONTACT_SOLVER") or "off"
    ).strip()
    _log(log, f"7/8 Nets: contact solver mode = {resolved_contact_solver_mode}.")
    nets = build_nets(
        top_pads,
        bottom_pads,
        top_mask,
        bottom_mask,
        hole_pairs,
        top_plane,
        bottom_plane,
        log,
        expand_contacts_non_green=expand_non_green_contacts,
        contact_solver_mode=resolved_contact_solver_mode,
        diagnostics_folder=work_folder,
    )

    resolved_component_solver_mode = (
        component_solver_mode or os.environ.get("SCHEMATIC_GENERATOR_COMPONENT_SOLVER") or "sequential"
    ).strip()
    _log(log, f"7/8 Components: component solver mode = {resolved_component_solver_mode}.")
    elements = create_elements(
        top_pads,
        bottom_pads,
        hole_pairs,
        ocr_texts,
        side_images={"TOP": top_image, "BOTTOM": bottom_image},
        mask_traces={"TOP": top_mask, "BOTTOM": bottom_mask},
        samples_colors={"TOP": top_samples or {}, "BOTTOM": bottom_samples or {}},
        component_solver_mode=resolved_component_solver_mode,
        diagnostics_folder=work_folder,
        log=log,
    )

    netlist_path = work_folder / "netlist.txt"
    devices_path = work_folder / "devices.tsv"
    kicad_path = output_folder / "schematic.kicad_sch"
    ocr_path = work_folder / "ocr_text.tsv"
    top_preview_path = work_folder / "preview_top.png"
    bottom_preview_path = work_folder / "preview_bottom.png"
    top_traces_path = work_folder / "top_traces_thick.png"
    bottom_traces_path = work_folder / "bottom_traces_thick.png"
    top_connections_path = work_folder / "top_connections_color.png"
    bottom_connections_path = work_folder / "bottom_connections_color.png"
    overlay_alignment_path = work_folder / "alignment_top_bottom_50.png"
    hole_alignment_path = work_folder / "alignment_holes.png"

    _log(log, "8/8 Saving netlist, KiCad schematic, masks, previews, and optimized schematic.")
    save_netlist(netlist_path, nets)
    save_devices(devices_path, elements)
    save_ocr_texts(ocr_path, ocr_texts)
    save_kicad_schematic(kicad_path, elements)
    save_schematic_preview_png(output_folder / "schematic.png", elements)
    schematic_validation = validate_kicad_schematic(kicad_path, elements, log)
    schematic_optimization = optimize_schematic_automatically(kicad_path, log)

    save_image(work_folder / "top_normalized.png", top_image)
    save_image(work_folder / "bottom_normalized.png", bottom_image)
    save_image(work_folder / "top_mask.png", convert_mask_to_bgr(top_mask))
    save_image(work_folder / "bottom_mask.png", convert_mask_to_bgr(bottom_mask))
    save_image(work_folder / "top_plane.png", convert_mask_to_bgr(top_plane))
    save_image(work_folder / "bottom_plane.png", convert_mask_to_bgr(bottom_plane))
    save_image(top_traces_path, draw_thick_traces(top_image, top_mask, (0, 0, 255)))
    save_image(bottom_traces_path, draw_thick_traces(bottom_image, bottom_mask, (255, 0, 0)))
    save_image(top_connections_path, draw_connections_colored(top_image, top_mask, top_pads))
    save_image(bottom_connections_path, draw_connections_colored(bottom_image, bottom_mask, bottom_pads))
    save_image(overlay_alignment_path, overlay_bottom_on_top(top_image, bottom_image, 0.5))
    save_image(hole_alignment_path, draw_alignment_holes(top_image, bottom_image, top_pads, bottom_pads, hole_pairs))
    save_image(top_preview_path, draw_preview(top_image, top_mask, top_pads, hole_pairs, "TOP"))
    save_image(
        bottom_preview_path,
        draw_preview(bottom_image, bottom_mask, bottom_pads, hole_pairs, "BOTTOM"),
    )

    parameters = {
        "top_path": str(top_path),
        "bottom_path": str(bottom_path),
        "schematic_name": schematic_name,
        "tesseract_path": str(tesseract_path) if tesseract_path else "",
        "has_groundplane": bool(has_groundplane),
        "color_samples": _samples_to_json(color_samples),
        "filter_combination_count": int(filter_combination_count),
        "alignment_threshold": float(alignment_threshold),
        "component_solver_mode": resolved_component_solver_mode,
        "contact_solver_mode": resolved_contact_solver_mode,
        "top_quality": top_quality,
        "bottom_quality": bottom_quality,
        "top_rectification": {
            "description": top_rectification.description,
            "confidence": top_rectification.confidence,
            "size": [int(top_image.shape[1]), int(top_image.shape[0])],
        },
        "bottom_rectification": {
            "description": bottom_rectification.description,
            "confidence": bottom_rectification.confidence,
            "size": [int(bottom_image.shape[1]), int(bottom_image.shape[0])],
        },
        "alignment": {
            "description": alignment.description,
            "mean_error": alignment.mean_error,
            "match_count": alignment.count_matches,
            "alignment_percent": alignment.percent_alignment,
        },
        "ocr": {
            "text_count": len(ocr_texts),
            "pass_count": ocr_pass_count,
            "file": str(ocr_path),
        },
        "optimized_schematic": schematic_optimization,
    }
    _save_reconstruction(work_folder, top_pads, bottom_pads, hole_pairs, nets, elements, ocr_texts, parameters)
    _save_project(work_folder, parameters)
    save_diagnostics(
        work_folder,
        top_pads,
        bottom_pads,
        hole_pairs,
        nets,
        elements,
        [],
        0,
        top_mask=top_mask,
        bottom_mask=bottom_mask,
        plane_top=top_plane,
        plane_bottom=bottom_plane,
        schematic_validation=schematic_validation,
        expand_contacts_non_green=expand_non_green_contacts,
        contact_solver_mode=resolved_contact_solver_mode,
    )
    _log(log, "Analysis finished and saved.")

    return AnalysisReport(
        output_folder=str(output_folder),
        count_pads_top=len(top_pads),
        count_pads_bottom=len(bottom_pads),
        count_pairs_holes=len(hole_pairs),
        count_nets=len(nets),
        count_elements=len(elements),
        file_netlist=str(netlist_path),
        file_devices=str(devices_path),
        file_kicad=str(kicad_path),
        file_preview=str(top_preview_path),
    )


def _create_output_folder(base_folder: str | Path, schematic_name: str = "") -> Path:
    """Create a timestamped output folder for one analysis run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_suffix = _safe_name(schematic_name)
    folder_name = f"results_{timestamp}_{safe_suffix}" if safe_suffix else f"results_{timestamp}"
    folder = Path(base_folder) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _copy_original_image(path: str | Path, output_folder: Path, side: str, log: LogFn | None) -> None:
    """Copy the input image into the output folder for traceability."""

    source = Path(path)
    extension = source.suffix if source.suffix else ".png"
    target = output_folder / f"{side}_original{extension}"
    shutil.copy2(source, target)
    _log(log, f"{side}: saved original image copy in the output folder: {target.name}.")


def _safe_name(name: str) -> str:
    """Convert a user-provided schematic name into a filesystem-safe suffix."""

    characters = []
    for character in name.strip():
        if character.isalnum() or character in ("-", "_"):
            characters.append(character)
        elif character.isspace():
            characters.append("_")
    return "".join(characters)[:60].strip("_")


def _samples_for_side(
    samples: dict[str, dict[str, tuple[int, int, int]]] | None,
    side: str,
    fallback: str,
) -> dict[str, tuple[int, int, int]] | None:
    """Return color samples for one side, falling back to the opposite side when needed."""

    if not samples:
        return None
    result: dict[str, tuple[int, int, int]] = {}
    side_data = samples.get(side, {})
    fallback_data = samples.get(fallback, {})
    for key in ("trace_bgr", "background_bgr"):
        value = side_data.get(key) or fallback_data.get(key)
        if value:
            result[key] = tuple(int(x) for x in value)
    return result or None


def _has_non_green_soldermask_sample(samples: dict[str, tuple[int, int, int]] | None) -> bool:
    """Detect whether the supplied background sample likely represents a non-green soldermask."""

    if not samples or "background_bgr" not in samples:
        return False
    bgr = np.array([[samples["background_bgr"]]], dtype=np.uint8)
    h, s, _v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(s) >= 35.0 and not (35.0 <= float(h) <= 105.0)


def _samples_to_json(samples: dict[str, dict[str, tuple[int, int, int]]] | None) -> dict[str, dict[str, list[int]]]:
    """Convert tuple-based color samples into JSON-serializable lists."""

    result: dict[str, dict[str, list[int]]] = {}
    for side, data in (samples or {}).items():
        result[side] = {key: [int(x) for x in value] for key, value in data.items()}
    return result


def _normalize_image_quality(raw_quality: dict[str, float | int | str]) -> dict[str, float | int | str]:
    """Translate image-quality metric keys returned by the current image module."""

    return {
        "sharpness": raw_quality.get("sharpness_laplacian", 0.0),
        "dark_fraction": raw_quality.get("dark_pixels_percent", 0.0),
        "bright_fraction": raw_quality.get("overexposed_pixels_percent", 0.0),
        "contrast": raw_quality.get("contrast_p5_p95", 0.0),
        "quality_score": raw_quality.get("quality_score", 0.0),
        "rating": raw_quality.get("rating", ""),
    }


def _rectify_with_pad_validation(image: np.ndarray, side: str, log: LogFn | None) -> RectificationResult:
    """Rectify the board image and reject the result if pad detection degrades too much."""

    pads_before = detect_pads(image, side, None)
    result = rectify_board(image, side, log)
    pads_after = detect_pads(result.image, side, None)
    contrast_after = _p5_p95_contrast(result.image)
    _log(
        log,
        f"{side}: rectification validation: pads before={len(pads_before)}, "
        f"after={len(pads_after)}, contrast_after={contrast_after:.1f}.",
    )

    minimum_expected_pads = max(3, int(round(len(pads_before) * 0.55)))
    loses_pads = len(pads_before) >= 4 and len(pads_after) < minimum_expected_pads
    if loses_pads:
        _log(
            log,
            f"{side}: rejecting perspective correction because it loses board features; "
            "using the input image and hole-based alignment instead.",
        )
        return RectificationResult(
            image.copy(),
            np.eye(3, dtype=np.float32),
            "no correction - pad validation rejected rectification",
            0.0,
        )
    return result


def _p5_p95_contrast(image: np.ndarray) -> float:
    """Measure image contrast as the difference between grayscale p95 and p5."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(gray, 95) - np.percentile(gray, 5))


def _fill_missing_via_pads(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    log: LogFn | None,
) -> tuple[list[Pad], list[Pad], list[HolePair]]:
    """Infer missing opposite-side pads for likely through-hole vias."""

    if len(pairs) < max(3, int(min(len(top_pads), len(bottom_pads)) * 0.55)):
        return top_pads, bottom_pads, pairs

    paired_top_nodes = {pair.pad_top for pair in pairs}
    paired_bottom_nodes = {pair.pad_bottom for pair in pairs}
    median_radius = _median_radius([*top_pads, *bottom_pads])
    added_count = 0

    for pad in list(top_pads):
        if pad.node in paired_top_nodes or pad.radius < median_radius * 0.55:
            continue
        if _has_nearby_pad(pad, bottom_pads):
            continue
        new_pad = _copy_pad_to_side(pad, "BOTTOM", _next_pad_id(bottom_pads), "inferred")
        bottom_pads.append(new_pad)
        pairs.append(HolePair(pad.node, new_pad.node, 0.0, min(0.45, pad.confidence * 0.6)))
        paired_top_nodes.add(pad.node)
        paired_bottom_nodes.add(new_pad.node)
        added_count += 1

    for pad in list(bottom_pads):
        if pad.node in paired_bottom_nodes or pad.radius < median_radius * 0.55:
            continue
        if _has_nearby_pad(pad, top_pads):
            continue
        new_pad = _copy_pad_to_side(pad, "TOP", _next_pad_id(top_pads), "inferred")
        top_pads.append(new_pad)
        pairs.append(HolePair(new_pad.node, pad.node, 0.0, min(0.45, pad.confidence * 0.6)))
        paired_top_nodes.add(new_pad.node)
        paired_bottom_nodes.add(pad.node)
        added_count += 1

    if added_count:
        top_pads.sort(key=lambda pad: (pad.y, pad.x, pad.identifier))
        bottom_pads.sort(key=lambda pad: (pad.y, pad.x, pad.identifier))
        _log(log, f"Hole pairs: inferred {added_count} missing THT pads on the opposite side.")
    return top_pads, bottom_pads, pairs


def _has_nearby_pad(pad: Pad, candidates: list[Pad]) -> bool:
    """Check whether another pad is close enough to represent the same physical location."""

    tolerance = max(8.0, pad.radius * 2.2)
    return any(
        np.hypot(pad.x - candidate.x, pad.y - candidate.y) <= tolerance
        for candidate in candidates
    )


def _copy_pad_to_side(pad: Pad, side: str, identifier: str, status: str) -> Pad:
    """Create an inferred copy of one pad on the opposite PCB side."""

    return Pad(
        identifier=identifier,
        side=side,
        x=pad.x,
        y=pad.y,
        radius=pad.radius,
        confidence=min(0.45, pad.confidence * 0.6),
        type=pad.type,
        status=status,
        name=pad.name,
    )


def _next_pad_id(pads: list[Pad]) -> str:
    """Return the next sequential pad identifier for a pad collection."""

    numbers = []
    for pad in pads:
        text = pad.identifier.upper().removeprefix("P")
        if text.isdigit():
            numbers.append(int(text))
    return f"P{(max(numbers) if numbers else 0) + 1:04d}"


def _median_radius(pads: list[Pad]) -> float:
    """Calculate the median pad radius, using a stable fallback for empty input."""

    if not pads:
        return 1.0
    return float(np.median([pad.radius for pad in pads]))


def _save_reconstruction(
    folder: str | Path,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets,
    elements,
    ocr_texts,
    parameters: dict,
) -> None:
    """Persist the editable reconstruction state used by the correction editor."""

    data = {
        "parameters": parameters,
        "manual_corrections_file": str(Path(folder) / "manual_corrections.json"),
        "top_pads": [asdict(pad) for pad in top_pads],
        "bottom_pads": [asdict(pad) for pad in bottom_pads],
        "hole_pairs": [asdict(pair) for pair in pairs],
        "nets": [asdict(net) for net in nets],
        "elements": [asdict(element) for element in elements],
        "ocr_texts": ocr_texts,
    }
    (Path(folder) / "reconstruction.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    corrections_path = Path(folder) / "manual_corrections.json"
    if not corrections_path.exists():
        corrections_path.write_text("[]\n", encoding="utf-8")


def _save_project(folder: str | Path, parameters: dict) -> None:
    """Save the run parameters as a compact project metadata file."""

    (Path(folder) / "project.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _log(log: LogFn | None, text: str) -> None:
    """Write a message to the optional caller-provided logger."""

    if log:
        log(text)
