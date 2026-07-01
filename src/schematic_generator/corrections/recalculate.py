from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import schematic_generator.preview as preview
from schematic_generator.corrections.io import save_corrections_file, save_pending_reconstruction, save_reconstruction_with_results
from schematic_generator.corrections.manual import filter_auto_elements, manual_elements
from schematic_generator.corrections.models import (
    CORRECTED_BOTTOM_CONNECTIONS_PNG,
    CORRECTED_HOLES_ALIGNMENT_PNG,
    CORRECTED_SCHEMATIC_STEM,
    CORRECTED_TOP_CONNECTIONS_PNG,
    OPTIMIZED_CORRECTED_SCHEMATIC_KEY,
    CorrectionState,
    LogFn,
    RecalculationResult,
)
from schematic_generator.diagnostics_facade import save_diagnostics
from schematic_generator.images import load_image, save_image
from schematic_generator.kicad_facade import save_kicad_schematic, save_schematic_preview_png
from schematic_generator.netlist_facade import build_nets, create_elements, save_devices, save_netlist
from schematic_generator.schematic_facade import optimize_schematic_automatically
from schematic_generator.schematic_facade import validate_kicad_schematic

overlay_bottom_on_top = preview.overlay_bottom_on_top


def save_mask_by_corrections(work_folder: str | Path, side: str, mask: np.ndarray) -> Path:
    """Persist a corrected binary trace mask for one board side."""
    path = Path(work_folder) / f"{side.lower()}_mask_corrected.png"
    save_image(path, cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
    return path


def recalculate_and_save_by_corrections(state: CorrectionState, log: LogFn | None = None) -> RecalculationResult:
    """Rebuild nets, components, KiCad output, previews, diagnostics, and corrected reconstruction files."""
    for pad in [*state.top_pads, *state.bottom_pads]:
        pad.net = ""
    images = _load_images(state.work_folder)
    masks = _load_masks(state.work_folder)
    samples = _samples_from_parameters(state.parameters)
    nets = build_nets(
        state.top_pads,
        state.bottom_pads,
        masks["TOP"],
        masks["BOTTOM"],
        state.pairs,
        masks.get("PLANE_TOP"),
        masks.get("PLANE_BOTTOM"),
        log,
    )
    automatic_elements = create_elements(
        state.top_pads,
        state.bottom_pads,
        state.pairs,
        state.ocr_texts,
        side_images=images,
        mask_traces={"TOP": masks["TOP"], "BOTTOM": masks["BOTTOM"]},
        samples_colors=samples,
        log=log,
    )
    manual_component_elements = manual_elements(state)
    elements = filter_auto_elements(automatic_elements, manual_component_elements)
    elements.extend(manual_component_elements)

    netlist_path = state.work_folder / "netlist_corrected.txt"
    devices_path = state.work_folder / "devices_corrected.tsv"
    kicad_path = state.output_folder / f"{CORRECTED_SCHEMATIC_STEM}.kicad_sch"
    save_netlist(netlist_path, nets)
    save_devices(devices_path, elements)
    save_kicad_schematic(kicad_path, elements)
    save_schematic_preview_png(state.output_folder / f"{CORRECTED_SCHEMATIC_STEM}.png", elements)
    schematic_validation = validate_kicad_schematic(kicad_path, elements, log)
    schematic_optimization = optimize_schematic_automatically(kicad_path, log)

    _save_previews(state, images, masks)
    save_corrections_file(state.work_folder, state.corrections)
    state.pending_recalculation = False
    state.parameters = dict(state.parameters)
    state.parameters[OPTIMIZED_CORRECTED_SCHEMATIC_KEY] = schematic_optimization
    save_reconstruction_with_results(state, nets, elements)
    save_diagnostics(
        state.work_folder,
        state.top_pads,
        state.bottom_pads,
        state.pairs,
        nets,
        elements,
        state.manual_components,
        len(state.corrections),
        top_mask=masks["TOP"],
        bottom_mask=masks["BOTTOM"],
        plane_top=masks.get("PLANE_TOP"),
        plane_bottom=masks.get("PLANE_BOTTOM"),
        schematic_validation=schematic_validation,
    )
    _log(log, f"Corrections: recalculated {len(nets)} nets and saved corrected output.")
    return RecalculationResult(nets, elements, netlist_path, devices_path, kicad_path)


def save_state_corrections_without_recalculation(state: CorrectionState) -> None:
    """Persist editable correction state without rebuilding nets, previews, diagnostics, or KiCad output."""
    state.pending_recalculation = True
    save_corrections_file(state.work_folder, state.corrections)
    save_pending_reconstruction(state)


def _load_images(work_folder: Path) -> dict[str, np.ndarray]:
    """Load normalized board images required by element creation and preview rendering."""
    return {
        "TOP": load_image(work_folder / "top_normalized.png"),
        "BOTTOM": load_image(work_folder / "bottom_normalized.png"),
    }


def _load_masks(work_folder: Path) -> dict[str, np.ndarray]:
    """Load corrected masks when available and fall back to automatic mask filenames."""
    masks = {
        "TOP": _load_binary_mask(_prefer_corrected(work_folder, "top_mask.png", "top_mask_corrected.png")),
        "BOTTOM": _load_binary_mask(_prefer_corrected(work_folder, "bottom_mask.png", "bottom_mask_corrected.png")),
    }
    plane_top = work_folder / "plane_top.png"
    plane_bottom = work_folder / "plane_bottom.png"
    if plane_top.exists():
        masks["PLANE_TOP"] = _load_binary_mask(plane_top)
    if plane_bottom.exists():
        masks["PLANE_BOTTOM"] = _load_binary_mask(plane_bottom)
    return masks


def _prefer_corrected(work_folder: Path, automatic: str, corrected: str) -> Path:
    """Choose the best available mask path with corrected files taking precedence over automatic files."""
    corrected_path = work_folder / corrected
    if corrected_path.exists():
        return corrected_path
    automatic_path = work_folder / automatic
    if automatic_path.exists():
        return automatic_path
    return automatic_path


def _load_binary_mask(path: Path) -> np.ndarray:
    """Load an image as a binary uint8 trace mask."""
    image = load_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.where(gray > 127, 255, 0).astype(np.uint8)


def _samples_from_parameters(parameters: dict[str, Any]) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Extract color samples from reconstruction parameters in a normalized tuple form."""
    result: dict[str, dict[str, tuple[int, int, int]]] = {}
    for side, data in dict(parameters.get("color_samples", {})).items():
        result[side] = {
            key: tuple(int(channel) for channel in value)
            for key, value in dict(data).items()
            if isinstance(value, (list, tuple)) and len(value) == 3
        }
    return result


def _save_previews(state: CorrectionState, images: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> None:
    """Render corrected board previews, colored trace previews, and alignment previews."""
    save_image(state.work_folder / "preview_top_corrected.png", preview.draw_preview(images["TOP"], masks["TOP"], state.top_pads, state.pairs, "TOP"))
    save_image(state.work_folder / "preview_bottom_corrected.png", preview.draw_preview(images["BOTTOM"], masks["BOTTOM"], state.bottom_pads, state.pairs, "BOTTOM"))
    save_image(state.work_folder / CORRECTED_TOP_CONNECTIONS_PNG, preview.draw_connections_colored(images["TOP"], masks["TOP"], state.top_pads))
    save_image(state.work_folder / CORRECTED_BOTTOM_CONNECTIONS_PNG, preview.draw_connections_colored(images["BOTTOM"], masks["BOTTOM"], state.bottom_pads))
    save_image(state.work_folder / "alignment_top_bottom_50_corrected.png", overlay_bottom_on_top(images["TOP"], images["BOTTOM"], 0.5))
    save_image(state.work_folder / CORRECTED_HOLES_ALIGNMENT_PNG, preview.draw_alignment_holes(images["TOP"], images["BOTTOM"], state.top_pads, state.bottom_pads, state.pairs))


def _log(log: LogFn | None, text: str) -> None:
    """Send a progress message to an optional caller-provided logger."""
    if log:
        log(text)
