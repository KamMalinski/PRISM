from __future__ import annotations

import math
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from schematic_generator.contact_solver_facade import is_contact_solver, apply_contact_solver, save_solver_diagnostics
from schematic_generator.connection_graph_facade import active_electrical_edges, build_connection_graph
from schematic_generator.mask_contacts import components_traces, label_traces_near_pad
from schematic_generator.models import Element, Net, Pad, HolePair
from schematic_generator.component_solver import ComponentCandidate, select_candidates_globally

LogFn = Callable[[str], None]



class UnionFind:
    """Track connected graph nodes with union-find during net construction."""

    def __init__(self) -> None:
        """Initialize an empty union-find parent map."""
        self.parent: dict[str, str] = {}

    def add(self, name: str) -> None:
        """Add a node to the union-find structure if it is missing."""
        self.parent.setdefault(name, name)

    def find(self, name: str) -> str:
        """Return the representative node with path compression."""
        self.add(name)
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def connect(self, a: str, b: str) -> None:
        """Merge two node groups into one connected set."""
        self.parent[self.find(b)] = self.find(a)

def _map_pins_to_pads(pads: list[Pad]) -> dict[str, str]:
    """Map one-based schematic pin numbers to the detected pad node identifiers."""
    return {str(i): pad.node for i, pad in enumerate(pads, start=1)}

def _is_global_solver(mode: str | None) -> bool:
    """Return whether the configured component grouping mode should use global solving."""
    return str(mode or "").strip().lower() in {"global", "global_solver", "hybrid_solver"}

def _create_elements_with_global_solver(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pads_physical: list[Pad],
    paired_bottom: set[str],
    pad_texts: dict[str, str],
    ocr_texts: list[dict[str, float | int | str]],
    side_images: dict[str, np.ndarray],
    mask_traces: dict[str, np.ndarray],
    samples_colors: dict[str, dict[str, tuple[int, int, int]]],
    diagnostics_folder: str | Path | None,
    log: LogFn | None,
) -> list[Element]:
    """Create components by collecting all candidates first and resolving conflicts globally."""
    _log(log, "Elements global_solver: generating candidates from multiple sources before global selection.")
    pads_by_side = {"TOP": top_pads, "BOTTOM": [pad for pad in bottom_pads if pad.node not in paired_bottom]}
    candidates = _build_global_solver_candidates(
        pads_physical,
        pads_by_side,
        pad_texts,
        ocr_texts,
        side_images,
        mask_traces,
        samples_colors,
        log,
    )
    solver_result = select_candidates_globally(candidates)
    pads_by_node = {pad.node: pad for pad in pads_physical}
    type_counters: dict[str, int] = {"R": 1, "C": 1, "D": 1, "L": 1, "U": 1, "J": 1}
    elements = [
        _element_from_candidate(candidate, pads_by_node, pad_texts, ocr_texts, type_counters)
        for candidate in solver_result.selected
        if all(node in pads_by_node for node in candidate.pin_order)
    ]
    _add_testpoints_for_remaining(elements, pads_physical, pad_texts, solver_result.used_pads)
    _deduplicate_refs(elements)
    _save_global_solver_diagnostics(diagnostics_folder, candidates, solver_result)
    _log(
        log,
        f"Elements global_solver: candidates={len(candidates)}, selected={len(solver_result.selected)}, "
        f"used_pads={len(solver_result.used_pads)}, elementy_z_testpointami={len(elements)}.",
    )
    return elements

def _build_global_solver_candidates(
    pads_physical: list[Pad],
    pads_by_side: dict[str, list[Pad]],
    pad_texts: dict[str, str],
    ocr_texts: list[dict[str, float | int | str]],
    side_images: dict[str, np.ndarray],
    mask_traces: dict[str, np.ndarray],
    samples_colors: dict[str, dict[str, tuple[int, int, int]]],
    log: LogFn | None,
) -> list[ComponentCandidate]:
    """Collect component candidates from OCR, silkscreen, rows, net support, and footprints."""
    candidates: list[ComponentCandidate] = []
    seen: set[str] = set()

    for group in _detect_devices_by_ocr_ref(pads_by_side, set(), ocr_texts, log):
        _add_global_candidate(candidates, seen, group, "OCR/refdes", pad_texts, ocr_texts)

    groups_silkscreen = _filter_out_suspicious_multi_pin_groups(
        _detect_devices_by_silkscreen(
            pads_by_side,
            set(),
            ocr_texts,
            side_images,
            mask_traces,
            samples_colors,
            log,
        ),
        "silkscreen",
        log,
    )
    for group in groups_silkscreen:
        _add_global_candidate(candidates, seen, group, "silkscreen", pad_texts, ocr_texts)

    for group in _detect_rows_pads(pads_physical, log):
        _add_global_candidate(candidates, seen, group, "row_geometry", pad_texts, ocr_texts)

    for group in _filter_out_suspicious_multi_pin_groups(
        _detect_nearby_devices(pads_physical, set(), log),
        "footprint_geometry",
        log,
    ):
        _add_global_candidate(candidates, seen, group, "footprint_geometry", pad_texts, ocr_texts)

    for group, score_hint, evidence, features in _candidates_pairs_two_pin_global(pads_physical, log):
        _add_global_candidate(
            candidates,
            seen,
            group,
            "net_pattern",
            pad_texts,
            ocr_texts,
            score_hint=score_hint,
            evidence_extra=evidence,
            features_extra=features,
        )

    for group, score_hint, evidence, features in _candidates_footprint_two_pin_global(pads_physical, log):
        _add_global_candidate(
            candidates,
            seen,
            group,
            "footprint_template_2pin",
            pad_texts,
            ocr_texts,
            score_hint=score_hint,
            evidence_extra=evidence,
            features_extra=features,
        )

    for group, score_hint, evidence, features in _global_three_pin_footprint_candidates(pads_physical, log):
        _add_global_candidate(
            candidates,
            seen,
            group,
            "footprint_template_3pin",
            pad_texts,
            ocr_texts,
            score_hint=score_hint,
            evidence_extra=evidence,
            features_extra=features,
        )

    return candidates

def _add_global_candidate(
    candidates: list[ComponentCandidate],
    seen: set[str],
    group: list[Pad],
    source: str,
    pad_texts: dict[str, str],
    ocr_texts: list[dict[str, float | int | str]],
    *,
    score_hint: float = 0.0,
    evidence_extra: list[str] | None = None,
    features_extra: dict[str, Any] | None = None,
) -> None:
    """Score one candidate group and append it to the global solver input when usable."""
    group = sorted(_unique_pads(group), key=lambda pad: (pad.y, pad.x))
    if len(group) < 2:
        return
    pads_key = tuple(pad.node for pad in group)
    unique_key = f"{source}:{','.join(pads_key)}"
    if unique_key in seen:
        return
    seen.add(unique_key)

    ref = _label_from_ocr(group, ocr_texts) or _label_from_pads(group, pad_texts) or ""
    type, _prefix, value, footprint = _type_devices_from_ocr(ref, len(group), group)
    nets = sorted({pad.net for pad in group if pad.net and pad.net != "NET?"})
    distances = [
        math.hypot(a.x - b.x, a.y - b.y)
        for index, a in enumerate(group)
        for b in group[index + 1:]
    ]
    max_distance = max(distances) if distances else 0.0
    geometry = _candidate_geometry_metrics(group)
    features: dict[str, Any] = {
        "pin_count": len(group),
        "distinct_net_count": len(nets),
        "max_pad_distance": round(max_distance, 2),
        "median_radius": round(_radius_median(group), 2),
        "tht_hint": _looks_like_tht(group),
        "linearity_score": round(geometry["linearity_score"], 4),
        "spacing_score": round(geometry["spacing_score"], 4),
        "radius_similarity_score": round(geometry["radius_similarity_score"], 4),
        "geometry_score": round(geometry["geometry_score"], 4),
    }
    features.update(features_extra or {})
    risks = _risks_candidate_global(group, source, ref, type, footprint, nets, max_distance, features)
    score_breakdown = _score_breakdown_candidate(group, source, ref, type, footprint, nets, score_hint, features, risks)
    score = score_breakdown["combined_score"]
    weight = _weight_candidate_global(score_breakdown, source, len(group))
    evidence = [
        f"source={source}",
        f"pads={','.join(pads_key)}",
        f"distinct_nets={len(nets)}",
        f"geometry={geometry['geometry_score']:.3f}",
    ]
    if ref:
        evidence.append(f"ref={ref}")
    if len(group) >= 4 and source == "row_geometry" and geometry["geometry_score"] >= 0.80:
        evidence.append("connector_like_regular_row")
    evidence.extend(evidence_extra or [])

    candidates.append(ComponentCandidate(
        identifier=f"{source}:{len(candidates) + 1:04d}",
        source=source,
        pads=pads_key,
        pin_order=pads_key,
        proposed_ref=ref,
        proposed_type=type,
        proposed_value=value,
        proposed_footprint=footprint,
        score=round(score, 4),
        weight=round(weight, 4),
        feature_vector=features,
        score_breakdown=score_breakdown,
        evidence=evidence,
        risks=risks,
    ))
