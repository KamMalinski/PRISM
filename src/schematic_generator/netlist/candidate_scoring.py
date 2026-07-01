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



def _candidate_geometry_metrics(group: list[Pad]) -> dict[str, float]:
    """Measure spread, linearity, pitch regularity, and radius similarity for a pad group."""
    if len(group) <= 2:
        return {
            "linearity_score": 1.0,
            "spacing_score": 1.0,
            "radius_similarity_score": _similarity_radii_groups(group),
            "geometry_score": _similarity_radii_groups(group),
        }
    points = np.array([[pad.x, pad.y] for pad in group], dtype=np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _values, wektory = np.linalg.eigh(np.cov(centered, rowvar=False))
        main_axis = wektory[:, int(np.argmax(_values))]
    except Exception:
        main_axis = np.array([1.0, 0.0], dtype=np.float32)
    projekcje = centered @ main_axis
    prostopadle = centered - np.outer(projekcje, main_axis)
    radius = max(1.0, _radius_median(group))
    rozrzut = max(1.0, float(np.max(projekcje) - np.min(projekcje)))
    error_linii = float(np.median(np.linalg.norm(prostopadle, axis=1)))
    liniowosc = 1.0 - min(1.0, error_linii / max(radius * 2.6, rozrzut * 0.16))

    spacings = np.diff(np.sort(projekcje))
    spacings = spacings[spacings > max(0.5, radius * 0.25)]
    if len(spacings) >= 2:
        median = max(1.0, float(np.median(spacings)))
        mad = float(np.median(np.abs(spacings - median)))
        raster = 1.0 - min(1.0, mad / max(radius * 1.8, median * 0.45))
    else:
        raster = 0.55
    similarity_radii = _similarity_radii_groups(group)
    geometry = 0.48 * liniowosc + 0.34 * raster + 0.18 * similarity_radii
    return {
        "linearity_score": round(max(0.0, min(1.0, liniowosc)), 4),
        "spacing_score": round(max(0.0, min(1.0, raster)), 4),
        "radius_similarity_score": round(max(0.0, min(1.0, similarity_radii)), 4),
        "geometry_score": round(max(0.0, min(1.0, geometry)), 4),
    }

def _similarity_radii_groups(group: list[Pad]) -> float:
    """Return a normalized score describing how similar pad radii are inside a group."""
    if not group:
        return 0.0
    radii = np.array([pad.radius for pad in group], dtype=np.float32)
    median = max(1.0, float(np.median(radii)))
    odchylenie = float(np.median(np.abs(radii - median)))
    return max(0.0, min(1.0, 1.0 - odchylenie / max(1.0, median * 0.55)))

def _risks_candidate_global(
    group: list[Pad],
    source: str,
    ref: str,
    type: str,
    footprint: str,
    nets: list[str],
    max_distance: float,
    features: dict[str, Any],
) -> list[str]:
    """List risk flags that make a global component candidate less trustworthy."""
    risks: list[str] = []
    if len(nets) < min(2, len(group)):
        risks.append("same_net_short")
    if len(group) == 2 and max_distance > max(220.0, _radius_median(group) * 28.0):
        risks.append("large_distance")
    if source == "OCR/refdes" and not ref:
        risks.append("weak_ocr")
    if len(group) >= 4 and len(nets) <= 2 and source != "OCR/refdes":
        risks.append("multi_pin_low_net_diversity")
    if source == "silkscreen" and len(group) >= 4 and not ref:
        risks.append("multi_pin_silkscreen_without_ref")
    geometry_score = float(features.get("geometry_score", 0.0) or 0.0)
    if source == "silkscreen" and len(group) >= 3 and not ref and geometry_score < 0.68:
        risks.append("silkscreen_false_merge_risk")
    if source == "silkscreen" and len(group) == 2 and ref and geometry_score < 0.55:
        risks.append("weak_silkscreen_pair_geometry")
    if len(group) >= 4 and not ref and geometry_score < 0.64:
        risks.append("irregular_multi_pin_without_ref")
    if type == "Device" and len(group) >= 3 and not ref and not footprint:
        risks.append("generic_multi_pin_roundtrip_risk")
    if type == "Device" and len(group) >= 4 and len(nets) >= 3 and not ref:
        risks.append("multi_net_generic_symbol_risk")
    if source == "net_pattern" and len(group) == 2 and float(features.get("net_support", 0.0) or 0.0) < 0.12:
        risks.append("weak_net_pattern_support")
    if source.startswith("footprint_template"):
        risks.append("template_candidate_needs_confirmation")
    if source.startswith("footprint_template") and not ref and float(features.get("template_support", 0.0) or 0.0) < 0.55:
        risks.append("weak_template_without_ref")
    return risks

def _score_breakdown_candidate(
    group: list[Pad],
    source: str,
    ref: str,
    type: str,
    footprint: str,
    nets: list[str],
    score_hint: float,
    features: dict[str, Any],
    risks: list[str],
) -> dict[str, float]:
    """Calculate breakdown candidate for ranking component candidates."""
    source_evidence = {
        "OCR/refdes": 0.68,
        "silkscreen": 0.54,
        "row_geometry": 0.58,
        "footprint_geometry": 0.44,
        "net_pattern": 0.48,
        "footprint_template_2pin": 0.36,
        "footprint_template_3pin": 0.34,
    }.get(source, 0.35)
    pad_confidence = min((pad.confidence for pad in group), default=0.0)
    evidence_score = source_evidence + min(0.18, max(0.0, pad_confidence - 0.45) * 0.32)
    if ref:
        evidence_score += 0.14
    if footprint:
        evidence_score += 0.04

    geometry_score = float(features.get("geometry_score", 0.0) or 0.0)
    if type == "PinRow" and len(group) >= 4 and features.get("linearity_score", 0.0) >= 0.80:
        geometry_score = min(1.0, geometry_score + 0.08)

    if not nets:
        net_context_score = 0.12
    elif len(group) == 2:
        net_context_score = 0.82 if len(nets) >= 2 else 0.04
    else:
        net_context_score = min(0.86, 0.22 + 0.66 * (len(nets) / max(1, len(group))))
    if "net_support" in features:
        net_context_score = min(1.0, net_context_score + min(0.14, float(features.get("net_support", 0.0) or 0.0) * 0.10))

    if source == "OCR/refdes" and ref:
        ocr_score = 0.92
    elif ref:
        ocr_score = 0.62
    elif source == "OCR/refdes":
        ocr_score = 0.18
    else:
        ocr_score = 0.02

    risk_penalty, roundtrip_penalty = _global_risk_penalties(risks)
    combined = (
        0.30 * min(1.0, evidence_score)
        + 0.27 * min(1.0, geometry_score)
        + 0.25 * min(1.0, net_context_score)
        + 0.18 * min(1.0, ocr_score)
        + min(0.12, max(0.0, score_hint))
    )
    return {
        "evidence_score": round(min(1.0, evidence_score), 4),
        "geometry_score": round(min(1.0, geometry_score), 4),
        "net_context_score": round(min(1.0, net_context_score), 4),
        "ocr_score": round(min(1.0, ocr_score), 4),
        "risk_penalty": round(risk_penalty, 4),
        "roundtrip_risk_penalty": round(roundtrip_penalty, 4),
        "combined_score": round(max(0.0, min(1.0, combined)), 4),
    }

def _global_risk_penalties(risks: list[str]) -> tuple[float, float]:
    """Convert candidate risk flags into confidence and solver-weight penalties."""
    risk_penalty = 0.0
    roundtrip_penalty = 0.0
    for risk in risks:
        if risk in {"generic_multi_pin_roundtrip_risk", "multi_net_generic_symbol_risk"}:
            roundtrip_penalty += {
                "generic_multi_pin_roundtrip_risk": 0.48,
                "multi_net_generic_symbol_risk": 0.38,
            }[risk]
            continue
        risk_penalty += {
            "same_net_short": 0.35,
            "large_distance": 0.16,
            "weak_ocr": 0.12,
            "multi_pin_low_net_diversity": 0.42,
            "multi_pin_silkscreen_without_ref": 0.85,
            "silkscreen_false_merge_risk": 0.34,
            "weak_silkscreen_pair_geometry": 0.42,
            "irregular_multi_pin_without_ref": 0.28,
            "weak_net_pattern_support": 0.16,
            "template_candidate_needs_confirmation": 0.46,
            "weak_template_without_ref": 0.18,
        }.get(risk, 0.10)
    return risk_penalty, roundtrip_penalty

def _weight_candidate_global(score_breakdown: dict[str, float], source: str, pin_count: int) -> float:
    """Build the final solver weight from source quality, score components, and pin count."""
    score = float(score_breakdown.get("combined_score", 0.0) or 0.0)
    source_bonus = {
        "OCR/refdes": 0.30,
        "silkscreen": 0.18,
        "row_geometry": 0.16,
        "footprint_geometry": 0.05,
        "net_pattern": 0.08,
        "footprint_template_2pin": 0.0,
        "footprint_template_3pin": 0.0,
    }.get(source, 0.0)
    coverage_bonus = min(0.18, max(0, pin_count - 1) * 0.035)
    risk_penalty = float(score_breakdown.get("risk_penalty", 0.0) or 0.0)
    roundtrip_penalty = float(score_breakdown.get("roundtrip_risk_penalty", 0.0) or 0.0)
    return max(-1.0, score + source_bonus + coverage_bonus - risk_penalty - roundtrip_penalty - 0.50)
