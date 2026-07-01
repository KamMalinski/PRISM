from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schematic_generator.corrections.models import CorrectionState
from schematic_generator.models import HolePair, ManualCorrection, Pad


def load_state_corrections(output_folder: str | Path) -> CorrectionState:
    """Load the editable correction state, preferring corrected reconstruction data when it exists."""
    output_path = Path(output_folder)
    work_folder = output_path / "work"
    reconstruction_path = work_folder / "reconstruction_corrected.json"
    automatic_path = work_folder / "reconstruction.json"
    source_path = reconstruction_path if reconstruction_path.exists() else automatic_path
    data = json.loads(source_path.read_text(encoding="utf-8"))
    return CorrectionState(
        output_folder=output_path,
        work_folder=work_folder,
        parameters=dict(data.get("parameters", {})),
        top_pads=[_pad_from_data(entry) for entry in data.get("top_pads", [])],
        bottom_pads=[_pad_from_data(entry) for entry in data.get("bottom_pads", [])],
        pairs=[_pair_from_data(entry) for entry in data.get("hole_pairs", [])],
        ocr_texts=list(data.get("ocr_texts", [])),
        corrections=_load_corrections(work_folder),
        manual_components=list(data.get("manual_components", [])),
        pending_recalculation=bool(data.get("pending_recalculation", False)),
    )


def save_pending_reconstruction(state: CorrectionState) -> None:
    """Write editable state to reconstruction_corrected.json without rebuilding nets or KiCad output."""
    target_path = state.work_folder / "reconstruction_corrected.json"
    source_path = target_path if target_path.exists() else state.work_folder / "reconstruction.json"
    data: dict[str, Any] = {}
    if source_path.exists():
        data = json.loads(source_path.read_text(encoding="utf-8"))
    data.update(_reconstruction_payload(state, None, None))
    target_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_reconstruction_with_results(state: CorrectionState, nets: list[Any], elements: list[Any]) -> None:
    """Write corrected reconstruction data together with recalculated nets and elements."""
    payload = _reconstruction_payload(state, nets, elements)
    (state.work_folder / "reconstruction_corrected.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _reconstruction_payload(state: CorrectionState, nets: list[Any] | None, elements: list[Any] | None) -> dict[str, Any]:
    """Create the JSON payload shared by pending saves and completed recalculations."""
    payload: dict[str, Any] = {
        "parameters": state.parameters,
        "manual_corrections_file": str(state.work_folder / "manual_corrections.json"),
        "top_pads": [asdict(pad) for pad in state.top_pads],
        "bottom_pads": [asdict(pad) for pad in state.bottom_pads],
        "hole_pairs": [asdict(pair) for pair in state.pairs],
        "manual_components": state.manual_components,
        "ocr_texts": state.ocr_texts,
        "pending_recalculation": state.pending_recalculation,
    }
    if nets is not None:
        payload["nets"] = [asdict(net) for net in nets]
    if elements is not None:
        payload["elements"] = [asdict(element) for element in elements]
    return payload


def load_corrections_file(work_folder: Path) -> list[ManualCorrection]:
    """Read saved manual corrections from disk."""
    return _load_corrections(work_folder)


def save_corrections_file(work_folder: Path, corrections: list[ManualCorrection]) -> None:
    """Persist the manual corrections list in the model's current JSON shape."""
    (work_folder / "manual_corrections.json").write_text(
        json.dumps([asdict(correction) for correction in corrections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_corrections(work_folder: Path) -> list[ManualCorrection]:
    """Parse manual_corrections.json when it exists, otherwise start with an empty correction list."""
    path = work_folder / "manual_corrections.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        ManualCorrection(
            str(entry.get("identifier", "")),
            str(entry.get("type", "")),
            str(entry.get("description", "")),
            dict(entry.get("data", {})),
            list(entry.get("suggestions", [])),
            str(entry.get("created", "")),
        )
        for entry in data
    ]


def _pad_from_data(data: dict[str, Any]) -> Pad:
    """Create a Pad from reconstruction JSON fields."""
    return Pad(
        str(data.get("identifier", "")),
        str(data.get("side", "")),
        float(data.get("x", 0.0)),
        float(data.get("y", 0.0)),
        float(data.get("radius", 5.0)),
        float(data.get("confidence", 0.0)),
        str(data.get("net", "")),
        str(data.get("type", "pad")),
        str(data.get("status", "auto")),
        str(data.get("name", "")),
    )


def _pair_from_data(data: dict[str, Any]) -> HolePair:
    """Create a HolePair from reconstruction JSON fields."""
    return HolePair(
        str(data.get("pad_top", "")),
        str(data.get("pad_bottom", "")),
        float(data.get("distance", 0.0)),
        float(data.get("confidence", 0.0)),
    )


def component_pads(component: dict[str, Any]) -> list[str]:
    """Return manual component pad node names."""
    return [str(node) for node in component.get("pads", [])]


def component_type(component: dict[str, Any]) -> str:
    """Return the manual component type."""
    return str(component.get("type", "Device"))


def component_value(component: dict[str, Any]) -> str:
    """Return the manual component value."""
    return str(component.get("value", ""))
