from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from schematic_generator.contact_solver.constants import CONTACT_EDGE_TYPES, ELECTRICAL_EDGE_TYPES, SCHEMA_VERSION
from schematic_generator.models import HolePair


@dataclass(slots=True)
class ContactSolverResult:
    """Result bundle produced by the optional contact solver."""

    mode: str
    graph: dict[str, Any]
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    baseline: dict[str, Any]
    predicted: dict[str, Any]

    def summary_json(self) -> dict[str, Any]:
        """Return compact diagnostic JSON for the selected solver decisions."""

        risk_counts = Counter(
            risk
            for item in self.selected
            for risk in item.get("risk_flags", []) or []
        )
        action_counts = Counter(item.get("action", "") for item in self.selected)
        predicted_impact = _predicted_impact(self.baseline, self.predicted, self.selected)
        return {
            "schema": "schematic_generator.contact_solver",
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "algorithm": _algorithm_name(self.mode),
            "candidate_count": len(self.candidates),
            "selected_count": len(self.selected),
            "rejected_count": len(self.rejected),
            "selected_ids": [item.get("id", "") for item in self.selected],
            "action_counts": dict(sorted(action_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
            "baseline": self.baseline,
            "predicted": self.predicted,
            "predicted_impact": predicted_impact,
            "comparison_with_default": {
                "baseline": self.baseline,
                "opt_in_prediction": self.predicted,
                "impact": predicted_impact,
            },
            "selected": self.selected,
            "rejected_sample": self.rejected[:80],
        }

    def candidates_json(self) -> dict[str, Any]:
        """Return full candidate diagnostics for offline inspection."""

        return {
            "schema": "schematic_generator.contact_solver_candidates",
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates,
        }


def is_contact_solver(mode: str | None) -> bool:
    """Return whether a mode string enables the optional contact solver."""

    return _normalize_mode(mode) not in {"", "off", "none", "disabled", "sequential", "baseline"}


def apply_contact_solver(
    graph: dict[str, Any],
    *,
    mode: str | None,
    pairs: list[HolePair] | None = None,
) -> ContactSolverResult:
    """Runs an opt-in contact solver on a connectivity graph.

    The solver does not use fixture names or ground truth. V0.17 risk_averse
    only cuts risky false merges. V0.18 missing_edge additionally allows very
    narrow positive hypotheses for small isolated groups.
    """

    mode = _normalize_mode(mode)
    result = copy.deepcopy(graph)
    baseline_state = _graph_state(result, pairs or [])
    candidates = _build_candidates(result, baseline_state, mode)
    selected_ids = {
        item["id"]
        for item in candidates
        if item.get("decision") == "selected" and item.get("action") == "deactivate"
    }
    for index, edge in enumerate(result.get("edges", []) or []):
        candidate_id = _edge_candidate_id(index, edge)
        if candidate_id not in selected_ids:
            continue
        edge["active"] = False
        edge["reason"] = _append_reason(str(edge.get("reason", "")), "contact_solver:risk_averse_deactivated")
        attrs = edge.setdefault("attrs", {})
        if isinstance(attrs, dict):
            attrs["contact_solver_mode"] = mode
            attrs["contact_solver_decision"] = "deactivate"
            attrs["contact_solver_reason"] = "large_non_plane_false_merge_risk"

    for item in candidates:
        if item.get("decision") != "selected" or item.get("action") != "activate":
            continue
        new_edge = item.get("new_edge")
        if not isinstance(new_edge, dict):
            continue
        edge_copy = copy.deepcopy(new_edge)
        attrs = edge_copy.setdefault("attrs", {})
        if isinstance(attrs, dict):
            attrs["contact_solver_mode"] = mode
            attrs["contact_solver_decision"] = "activate"
            attrs["contact_solver_candidate_id"] = str(item.get("id", ""))
            attrs["contact_solver_reason"] = str(item.get("decision_reason", "missing_edge_hypothesis"))
        result.setdefault("edges", []).append(edge_copy)

    predicted_state = _graph_state(result, pairs or [])
    result["summary"] = _graph_summary(result)
    selected = [item for item in candidates if item.get("decision") == "selected"]
    rejected = [item for item in candidates if item.get("decision") != "selected"]
    return ContactSolverResult(
        mode=mode,
        graph=result,
        candidates=candidates,
        selected=selected,
        rejected=rejected,
        baseline=_state_to_json(baseline_state),
        predicted=_state_to_json(predicted_state),
    )


def _predicted_impact(
    baseline: dict[str, Any],
    predicted: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare baseline and predicted graph states after selected solver actions."""

    removed_pin_pair_risk = sum(
        int((item.get("features", {}) or {}).get("trace_potential_pin_pairs", 0) or 0)
        for item in selected
        if item.get("edge_type") == "trace_contact"
    )
    removed_large_bridges = sum(
        1
        for item in selected
        if item.get("edge_type") == "mask_bridge" and item.get("action") == "deactivate"
    )
    deactivated_count = sum(1 for item in selected if item.get("action") == "deactivate")
    activated_count = sum(1 for item in selected if item.get("action") == "activate")
    activated_pad_gap = sum(
        1
        for item in selected
        if item.get("action") == "activate"
        and (item.get("features", {}) or {}).get("missing_edge_kind") == "pad_gap_bridge"
    )
    activated_pad_trace = sum(
        1
        for item in selected
        if item.get("action") == "activate"
        and (item.get("features", {}) or {}).get("missing_edge_kind") == "pad_trace_proximity"
    )
    added_false_merge_risk = sum(
        max(0, int((item.get("features", {}) or {}).get("merged_physical_pad_count", 0) or 0) - 1)
        for item in selected
        if item.get("action") == "activate"
    )
    return {
        "net_like_group_delta": int(predicted.get("net_like_group_count", 0) or 0) - int(baseline.get("net_like_group_count", 0) or 0),
        "single_physical_pad_group_delta": int(predicted.get("single_physical_pad_group_count", 0) or 0) - int(baseline.get("single_physical_pad_group_count", 0) or 0),
        "large_non_plane_group_delta": int(predicted.get("large_non_plane_group_count", 0) or 0) - int(baseline.get("large_non_plane_group_count", 0) or 0),
        "largest_physical_pad_group_delta": int(predicted.get("largest_physical_pad_group", 0) or 0) - int(baseline.get("largest_physical_pad_group", 0) or 0),
        "pin_false_positive_proxy_delta": int(added_false_merge_risk - removed_pin_pair_risk - removed_large_bridges * 4),
        "pin_false_negative_proxy_delta": int(deactivated_count - activated_count),
        "removed_trace_potential_pin_pairs": int(removed_pin_pair_risk),
        "removed_large_mask_bridges": int(removed_large_bridges),
        "added_missing_edge_candidates": int(activated_count),
        "activated_pad_gap_bridges": int(activated_pad_gap),
        "activated_pad_trace_proximity": int(activated_pad_trace),
    }


def _algorithm_name(mode: str) -> str:
    """Return the diagnostic algorithm label for a normalized solver mode."""

    if _is_missing_edge_mode(mode):
        return "risk_averse_plus_positive_missing_edge_solver"
    return "risk_averse_large_non_plane_contact_solver"


def _normalize_mode(mode: str | None) -> str:
    """Normalize user-facing mode aliases to internal solver mode ids."""

    text = str(mode or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "risk_averse"
    if text in {"probabilistic", "missing_edge", "missing-edge", "v0.18", "v0_18"}:
        return "missing_edge"
    return text


def _is_missing_edge_mode(mode: str) -> bool:
    """Return whether a normalized mode enables positive missing-edge hypotheses."""

    return mode in {"missing_edge"}


def _build_candidates(graph: dict[str, Any], state: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Build deactivate/activate/observe candidates from graph edges and group state."""

    candidates: list[dict[str, Any]] = []
    for index, edge in enumerate(graph.get("edges", []) or []):
        edge_type = str(edge.get("type", ""))
        if edge_type not in {"trace_contact", "mask_bridge", "via_pair"}:
            continue
        group_id = state.get("node_to_group", {}).get(str(edge.get("source", "")), "")
        group = state.get("groups", {}).get(group_id, {})
        attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs", {}), dict) else {}
        features = _features_edge(edge, group)
        risk_flags = _risk_flags(edge, group)
        action = _proposed_action(edge, group, risk_flags)
        score_breakdown = _score_breakdown(edge, group, action, risk_flags)
        score = round(sum(float(value) for value in score_breakdown.values()), 3)
        selected = action == "deactivate" and score >= 2.0
        if edge_type == "via_pair":
            action = "observe"
            selected = False
        candidate = {
            "id": _edge_candidate_id(index, edge),
            "edge_index": index,
            "edge_type": edge_type,
            "source": str(edge.get("source", "")),
            "target": str(edge.get("target", "")),
            "current_active": bool(edge.get("active", True)),
            "proposed_active": False if selected else bool(edge.get("active", True)),
            "action": action,
            "decision": "selected" if selected else "rejected",
            "reject_reason": "" if selected else _reject_reason(edge, group, action, risk_flags, score),
            "score": score,
            "score_breakdown": score_breakdown,
            "confidence": round(float(edge.get("confidence", 0.0) or 0.0), 3),
            "risk_flags": risk_flags,
            "features": features,
            "reason": str(edge.get("reason", "")),
        }
        if selected:
            candidate["decision_reason"] = "large_non_plane_false_merge_risk"
        if edge_type in CONTACT_EDGE_TYPES or edge_type == "via_pair":
            candidates.append(candidate)
    if _is_missing_edge_mode(mode):
        candidates.extend(_build_candidates_missing_edge(graph, state))
    candidates.sort(key=lambda item: (item.get("decision") != "selected", -float(item.get("score", 0.0)), item.get("id", "")))
    selected_trace_components: set[str] = set()
    for item in candidates:
        if (
            item.get("decision") != "selected"
            or item.get("edge_type") != "trace_contact"
            or item.get("action") != "deactivate"
        ):
            continue
        trace_component = str((item.get("features", {}) or {}).get("trace_component", "") or item.get("target", ""))
        if trace_component in selected_trace_components:
            item["decision"] = "rejected"
            item["proposed_active"] = item.get("current_active", True)
            item["reject_reason"] = "per_trace_component_deactivation_limit"
            item.pop("decision_reason", None)
            continue
        selected_trace_components.add(trace_component)
    return candidates


def _build_candidates_missing_edge(graph: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build positive missing-edge candidates for modes that allow activation."""

    candidates: list[dict[str, Any]] = []
    candidates.extend(_candidates_pad_gap_bridge(graph, state))
    candidates.extend(_candidates_pad_trace_proximity(graph, state))
    return _select_candidates_activation(candidates)


def _candidates_pad_gap_bridge(graph: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Propose bridge candidates between aligned small pad-only groups."""

    groups = _groups_from_pads(state)
    if len(groups) < 4 or _largest_group_physical(groups) != 1:
        return []
    nodes = _nodes_by_id(graph)
    candidates: list[dict[str, Any]] = []
    eligible = [
        (group_id, group)
        for group_id, group in groups
        if _count_w_group(group, "physical_pads", "physical_pad_count") == 1
        and _count_w_group(group, "pads", "pad_count") == 2
        and not bool(group.get("has_plane", False))
        and _trace_nodes_for_group(group)
    ]
    for first_index, (first_id, first_group) in enumerate(eligible):
        for second_id, second_group in eligible[first_index + 1:]:
            pair = _best_aligned_pad_pair(nodes, first_group, second_group)
            if pair is None:
                continue
            first_pad, second_pad, distance, dx, dy, axis = pair
            side = str(nodes[first_pad].get("side", ""))
            if not _has_trace_by_side(first_group, side) or not _has_trace_by_side(second_group, side):
                continue
            if _edge_exists(graph, first_pad, second_pad):
                continue
            merged_physical = _merged_physical_count(first_group, second_group)
            features = {
                "missing_edge_kind": "pad_gap_bridge",
                "source_group": first_id,
                "target_group": second_id,
                "source_physical_pad_count": _count_w_group(first_group, "physical_pads", "physical_pad_count"),
                "target_physical_pad_count": _count_w_group(second_group, "physical_pads", "physical_pad_count"),
                "merged_physical_pad_count": merged_physical,
                "distance_px": round(distance, 3),
                "axis": axis,
                "dx_px": round(dx, 3),
                "dy_px": round(dy, 3),
                "side": side,
            }
            score_breakdown = _score_missing_edge(features)
            score = round(sum(float(value) for value in score_breakdown.values()), 3)
            candidates.append({
                "id": f"missing_edge:pad_gap_bridge:{first_pad}->{second_pad}",
                "edge_index": None,
                "edge_type": "manual_connection",
                "source": first_pad,
                "target": second_pad,
                "current_active": False,
                "proposed_active": False,
                "action": "observe",
                "decision": "rejected",
                "reject_reason": "pad_gap_bridge_observed_not_activated_before_component_safety_model",
                "score": score,
                "score_breakdown": score_breakdown,
                "confidence": 0.36,
                "risk_flags": ["positive_missing_edge_hypothesis"],
                "features": features,
                "reason": "V0.18 single-pad aligned gap bridge hypothesis",
                "new_edge": _new_edge(
                    first_pad,
                    second_pad,
                    "manual_connection",
                    0.36,
                    "contact_solver V0.18: aligned single-pad gap bridge",
                    {
                        "missing_edge_kind": "pad_gap_bridge",
                        "distance_px": round(distance, 3),
                        "axis": axis,
                    },
                ),
            })
    return candidates


def _candidates_pad_trace_proximity(graph: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Propose missing pad-to-trace contacts using local proximity evidence."""

    groups = _groups_from_pads(state)
    if (
        _largest_group_physical(groups) > 2
        or _count_groups_single(groups) < 8
        or _count_groups_large_non_plane(groups) > 0
    ):
        return []
    nodes = _nodes_by_id(graph)
    active_trace_contacts = {
        str(edge.get("source", ""))
        for edge in graph.get("edges", []) or []
        if bool(edge.get("active", True)) and edge.get("type") == "trace_contact"
    }
    trace_nodes = [
        node
        for node in graph.get("nodes", []) or []
        if node.get("type") == "trace_component" and node.get("bbox")
    ]
    candidates: list[dict[str, Any]] = []
    eligible = [
        (group_id, group)
        for group_id, group in groups
        if _count_w_group(group, "physical_pads", "physical_pad_count") == 1
        and _count_w_group(group, "pads", "pad_count") == 2
        and not bool(group.get("has_plane", False))
    ]
    for source_id, source_group in eligible:
        for pad_id in sorted(source_group.get("pads", set())):
            pad = nodes.get(pad_id)
            if not pad or pad.get("type") != "pad" or pad_id in active_trace_contacts:
                continue
            side = str(pad.get("side", ""))
            for trace in trace_nodes:
                trace_id = str(trace.get("id", ""))
                if str(trace.get("side", "")) != side:
                    continue
                target_id = state.get("node_to_group", {}).get(trace_id, "")
                if not target_id or target_id == source_id:
                    continue
                target_group = state.get("groups", {}).get(target_id, {})
                if (
                    _count_w_group(target_group, "physical_pads", "physical_pad_count") != 1
                    or bool(target_group.get("has_plane", False))
                    or _merged_physical_count(source_group, target_group) > 2
                ):
                    continue
                gap, dx, dy = _point_bbox_gap(float(pad.get("x", 0.0) or 0.0), float(pad.get("y", 0.0) or 0.0), trace.get("bbox", []))
                if gap > 45.0:
                    continue
                if int(trace.get("area", 0) or 0) < 24:
                    continue
                if _edge_exists(graph, pad_id, trace_id):
                    continue
                features = {
                    "missing_edge_kind": "pad_trace_proximity",
                    "source_group": source_id,
                    "target_group": target_id,
                    "source_physical_pad_count": _count_w_group(source_group, "physical_pads", "physical_pad_count"),
                    "target_physical_pad_count": _count_w_group(target_group, "physical_pads", "physical_pad_count"),
                    "merged_physical_pad_count": _merged_physical_count(source_group, target_group),
                    "distance_px": round(gap, 3),
                    "dx_px": round(dx, 3),
                    "dy_px": round(dy, 3),
                    "side": side,
                    "trace_component": trace_id,
                    "component_area": int(trace.get("area", 0) or 0),
                }
                score_breakdown = _score_missing_edge(features)
                score = round(sum(float(value) for value in score_breakdown.values()), 3)
                confidence = round(max(0.28, min(0.48, 0.48 - gap * 0.004)), 3)
                candidates.append({
                    "id": f"missing_edge:pad_trace_proximity:{pad_id}->{trace_id}",
                    "edge_index": None,
                    "edge_type": "trace_contact",
                    "source": pad_id,
                    "target": trace_id,
                    "current_active": False,
                    "proposed_active": True,
                    "action": "activate",
                    "decision": "candidate",
                    "reject_reason": "",
                    "score": score,
                    "score_breakdown": score_breakdown,
                    "confidence": confidence,
                    "risk_flags": ["positive_missing_edge_hypothesis"],
                    "features": features,
                    "reason": "V0.18 pad-to-trace proximity missing-edge hypothesis",
                    "new_edge": _new_edge(
                        pad_id,
                        trace_id,
                        "trace_contact",
                        confidence,
                        "contact_solver V0.18: pad-to-trace proximity missing-edge",
                        {
                            "missing_edge_kind": "pad_trace_proximity",
                            "distance_px": round(gap, 3),
                            "trace_component": trace_id,
                            "component_area": int(trace.get("area", 0) or 0),
                        },
                    ),
                })
    return candidates


def _select_candidates_activation(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a conservative non-overlapping subset of positive activation candidates."""

    candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), float((item.get("features", {}) or {}).get("distance_px", 9999.0)), item.get("id", "")))
    used_groups: set[str] = set()
    selected_count = 0
    for item in candidates:
        if item.get("action") != "activate":
            item["decision"] = "rejected"
            item["proposed_active"] = False
            item.setdefault("reject_reason", "observe_only_missing_edge_hypothesis")
            item.pop("decision_reason", None)
            continue
        features = item.get("features", {}) or {}
        source_group = str(features.get("source_group", ""))
        target_group = str(features.get("target_group", ""))
        if selected_count >= 3:
            item["decision"] = "rejected"
            item["reject_reason"] = "missing_edge_activation_limit"
            item["proposed_active"] = False
            item.pop("decision_reason", None)
            continue
        if source_group in used_groups or target_group in used_groups:
            item["decision"] = "rejected"
            item["reject_reason"] = "per_group_activation_limit"
            item["proposed_active"] = False
            item.pop("decision_reason", None)
            continue
        if int(features.get("merged_physical_pad_count", 0) or 0) > 2:
            item["decision"] = "rejected"
            item["reject_reason"] = "merged_physical_pad_count_above_limit"
            item["proposed_active"] = False
            item.pop("decision_reason", None)
            continue
        item["decision"] = "selected"
        item["decision_reason"] = str(features.get("missing_edge_kind", "positive_missing_edge_hypothesis"))
        item["proposed_active"] = True
        selected_count += 1
        used_groups.add(source_group)
        used_groups.add(target_group)
    return candidates


def _score_missing_edge(features: dict[str, Any]) -> dict[str, float]:
    """Score a positive missing-edge hypothesis from local evidence features."""

    distance = float(features.get("distance_px", 9999.0) or 9999.0)
    merged = int(features.get("merged_physical_pad_count", 0) or 0)
    kind = str(features.get("missing_edge_kind", ""))
    distance_bonus = max(0.0, 1.4 - distance / 90.0)
    single_pair_bonus = 1.1 if merged <= 2 else -3.0
    kind_bonus = 0.55 if kind == "pad_trace_proximity" else 0.35
    false_merge_penalty = -0.55 * max(0, merged - 1)
    return {
        "local_evidence": round(distance_bonus, 3),
        "small_group_bonus": round(single_pair_bonus, 3),
        "kind_bonus": round(kind_bonus, 3),
        "false_merge_penalty": round(false_merge_penalty, 3),
    }


def _groups_from_pads(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return state groups that contain at least one pad."""

    return [
        (str(group_id), group)
        for group_id, group in (state.get("groups", {}) or {}).items()
        if _count_w_group(group, "pads", "pad_count") > 0
    ]


def _largest_group_physical(groups: list[tuple[str, dict[str, Any]]]) -> int:
    """Return the largest physical-pad count among state groups."""

    return max(
        (_count_w_group(group, "physical_pads", "physical_pad_count") for _group_id, group in groups),
        default=0,
    )


def _count_groups_single(groups: list[tuple[str, dict[str, Any]]]) -> int:
    """Count groups that contain exactly one physical pad."""

    return sum(
        1
        for _group_id, group in groups
        if _count_w_group(group, "physical_pads", "physical_pad_count") == 1
    )


def _count_groups_large_non_plane(groups: list[tuple[str, dict[str, Any]]]) -> int:
    """Count large groups that are not explained by a plane region."""

    return sum(
        1
        for _group_id, group in groups
        if _count_w_group(group, "physical_pads", "physical_pad_count") >= 8
        and not bool(group.get("has_plane", False))
    )


def _nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index serialized graph nodes by id."""

    return {
        str(node.get("id", "")): node
        for node in graph.get("nodes", []) or []
        if str(node.get("id", ""))
    }


def _trace_nodes_for_group(group: dict[str, Any]) -> list[str]:
    """Return sorted trace-component node ids contained in a group."""

    return sorted(str(node) for node in group.get("nodes", set()) if ":TRACE:" in str(node))


def _has_trace_by_side(group: dict[str, Any], side: str) -> bool:
    """Return whether a group contains a trace component on the requested PCB side."""

    prefix = f"{side}:TRACE:"
    return any(trace.startswith(prefix) for trace in _trace_nodes_for_group(group))


def _merged_physical_count(first_group: dict[str, Any], second_group: dict[str, Any]) -> int:
    """Return physical-pad count after merging two groups."""

    first = set(first_group.get("physical_pads", set()) or set())
    second = set(second_group.get("physical_pads", set()) or set())
    return len(first | second)


def _best_aligned_pad_pair(
    nodes: dict[str, dict[str, Any]],
    first_group: dict[str, Any],
    second_group: dict[str, Any],
) -> tuple[str, str, float, float, float, str] | None:
    """Find the closest horizontally or vertically aligned pad pair between two groups."""

    best: tuple[float, str, str, float, float, str] | None = None
    for first_pad in sorted(first_group.get("pads", set())):
        first = nodes.get(str(first_pad))
        if not first or first.get("type") != "pad":
            continue
        for second_pad in sorted(second_group.get("pads", set())):
            second = nodes.get(str(second_pad))
            if not second or second.get("type") != "pad" or first.get("side") != second.get("side"):
                continue
            side = str(first.get("side", ""))
            if not _has_trace_by_side(first_group, side) or not _has_trace_by_side(second_group, side):
                continue
            dx = abs(float(first.get("x", 0.0) or 0.0) - float(second.get("x", 0.0) or 0.0))
            dy = abs(float(first.get("y", 0.0) or 0.0) - float(second.get("y", 0.0) or 0.0))
            distance = math.hypot(dx, dy)
            axis = ""
            if dy <= 3.0 and 45.0 <= dx <= 130.0:
                axis = "horizontal"
            elif dx <= 3.0 and 45.0 <= dy <= 130.0:
                axis = "vertical"
            if not axis:
                continue
            value = (distance, str(first_pad), str(second_pad), dx, dy, axis)
            if best is None or value < best:
                best = value
    if best is None:
        return None
    distance, first_pad, second_pad, dx, dy, axis = best
    return first_pad, second_pad, distance, dx, dy, axis


def _point_bbox_gap(x: float, y: float, bbox: list[Any]) -> tuple[float, float, float]:
    """Measure the gap from a point to a bounding box."""

    if len(bbox) < 4:
        return 9999.0, 9999.0, 9999.0
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return math.hypot(dx, dy), dx, dy


def _edge_exists(graph: dict[str, Any], source: str, target: str) -> bool:
    """Return whether an undirected edge already exists between two graph nodes."""

    for edge in graph.get("edges", []) or []:
        edge_source = str(edge.get("source", ""))
        edge_target = str(edge.get("target", ""))
        if {edge_source, edge_target} == {source, target}:
            return True
    return False


def _new_edge(
    source: str,
    target: str,
    edge_type: str,
    confidence: float,
    reason: str,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Create a serialized graph edge proposed by the contact solver."""

    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "source_kind": "contact_solver",
        "confidence": round(confidence, 3),
        "active": True,
        "reason": reason,
        "geometry": {},
        "attrs": attrs,
    }


def _features_edge(edge: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    """Extract scoring features for one graph edge in its current group."""

    attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs", {}), dict) else {}
    geometry = edge.get("geometry", {}) if isinstance(edge.get("geometry", {}), dict) else {}
    return {
        "group_physical_pad_count": _count_w_group(group, "physical_pads", "physical_pad_count"),
        "group_pad_count": _count_w_group(group, "pads", "pad_count"),
        "group_has_plane": bool(group.get("has_plane", False)),
        "group_edge_count": int(group.get("active_edge_count", 0) or 0),
        "contact_mode": str(attrs.get("contact_mode", geometry.get("contact_mode", "")) or ""),
        "trace_component": str(attrs.get("trace_component", "")),
        "trace_pads_touching": int(attrs.get("trace_pads_touching", 0) or 0),
        "trace_pad_activation_limit": int(attrs.get("trace_pad_activation_limit", 0) or 0),
        "trace_potential_pin_pairs": int(attrs.get("trace_potential_pin_pairs", 0) or 0),
        "risky_large_trace_component": bool(attrs.get("risky_large_trace_component", False)),
        "component_area": int(geometry.get("component_area", 0) or 0),
        "dilation_px": int(geometry.get("dilation_px", 0) or 0),
        "via_pair_distance": float(attrs.get("distance", 0.0) or 0.0),
    }


def _risk_flags(edge: dict[str, Any], group: dict[str, Any]) -> list[str]:
    """Return risk labels that explain why an edge may be unsafe."""

    flags: list[str] = []
    attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs", {}), dict) else {}
    group_physical = _count_w_group(group, "physical_pads", "physical_pad_count")
    if group_physical >= 8 and not bool(group.get("has_plane", False)):
        flags.append("large_non_plane_candidate")
    if int(attrs.get("trace_pads_touching", 0) or 0) >= 8:
        flags.append("many_pads_on_trace_component")
    if int(attrs.get("trace_potential_pin_pairs", 0) or 0) >= 28:
        flags.append("many_potential_pin_pairs")
    if bool(attrs.get("risky_large_trace_component", False)):
        flags.append("risky_large_trace_component")
    if str(attrs.get("contact_mode", "")) == "relaxed_paired_pad":
        flags.append("relaxed_paired_pad")
    if edge.get("type") == "mask_bridge" and group_physical >= 8:
        flags.append("large_net_mask_bridge")
    if not bool(edge.get("active", True)):
        flags.append("inactive_contact")
    return flags


def _proposed_action(edge: dict[str, Any], group: dict[str, Any], risk_flags: list[str]) -> str:
    """Choose the default solver action for one edge and its risk flags."""

    if not bool(edge.get("active", True)):
        return "keep_inactive"
    if edge.get("type") == "trace_contact" and "large_non_plane_candidate" in risk_flags:
        if {"many_pads_on_trace_component", "many_potential_pin_pairs", "risky_large_trace_component"} & set(risk_flags):
            return "deactivate"
    return "keep_active"


def _score_breakdown(
    edge: dict[str, Any],
    group: dict[str, Any],
    action: str,
    risk_flags: list[str],
) -> dict[str, float]:
    """Score a deactivation candidate using interpretable additive terms."""

    if action != "deactivate":
        return {
            "false_merge_risk": 0.0,
            "large_net_penalty": 0.0,
            "bridge_risk": 0.0,
            "missing_edge_penalty": 0.0,
        }
    group_physical = _count_w_group(group, "physical_pads", "physical_pad_count")
    attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs", {}), dict) else {}
    potential_pairs = int(attrs.get("trace_potential_pin_pairs", 0) or 0)
    pads_touching = int(attrs.get("trace_pads_touching", 0) or 0)
    confidence = float(edge.get("confidence", 0.0) or 0.0)
    false_merge = 1.35 + min(2.0, group_physical * 0.05) + min(1.4, potential_pairs * 0.04)
    large_net = 1.2 if "large_non_plane_candidate" in risk_flags else 0.0
    bridge_risk = 1.0 if edge.get("type") == "mask_bridge" else min(1.0, pads_touching * 0.08)
    missing_penalty = -min(1.4, confidence * 1.1)
    return {
        "false_merge_risk": round(false_merge, 3),
        "large_net_penalty": round(large_net, 3),
        "bridge_risk": round(bridge_risk, 3),
        "missing_edge_penalty": round(missing_penalty, 3),
    }


def _count_w_group(group: dict[str, Any], set_key: str, count_key: str) -> int:
    """Read a count from a group, falling back to the size of a stored set."""

    if count_key in group:
        return int(group.get(count_key, 0) or 0)
    value = group.get(set_key, set())
    return len(value) if hasattr(value, "__len__") else 0


def _reject_reason(
    edge: dict[str, Any],
    group: dict[str, Any],
    action: str,
    risk_flags: list[str],
    score: float,
) -> str:
    """Explain why a solver candidate was not selected."""

    if edge.get("type") == "via_pair":
        return "via_pair_observed_not_modified"
    if not bool(edge.get("active", True)):
        return "inactive_contact_not_promoted_in_risk_averse_mode"
    if "large_non_plane_candidate" not in risk_flags:
        return "no_large_non_plane_risk"
    if action != "deactivate":
        return "risk_below_deactivation_rule"
    return f"score_below_threshold:{score:.3f}"


def _graph_state(graph: dict[str, Any], pairs: list[HolePair]) -> dict[str, Any]:
    """Compute connected groups and physical-pad metadata for a graph."""

    node_ids = [str(node.get("id", "")) for node in graph.get("nodes", []) or [] if str(node.get("id", ""))]
    pad_nodes = {
        str(node.get("id", ""))
        for node in graph.get("nodes", []) or []
        if node.get("type") == "pad"
    }
    plane_nodes = {
        str(node.get("id", ""))
        for node in graph.get("nodes", []) or []
        if node.get("type") == "plane_region"
    }
    uf = _UnionFind()
    for node_id in node_ids:
        uf.add(node_id)
    for edge in graph.get("edges", []) or []:
        if bool(edge.get("active", True)) and edge.get("type") in ELECTRICAL_EDGE_TYPES:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source and target:
                uf.union(source, target)

    physical = _map_pads_physical(pad_nodes, pairs, graph)
    groups: dict[str, dict[str, Any]] = {}
    node_to_group: dict[str, str] = {}
    for node_id in node_ids:
        root = uf.find(node_id)
        node_to_group[node_id] = root
        group = groups.setdefault(root, {
            "nodes": set(),
            "pads": set(),
            "physical_pads": set(),
            "has_plane": False,
            "active_edge_count": 0,
        })
        group["nodes"].add(node_id)
        if node_id in pad_nodes:
            group["pads"].add(node_id)
            group["physical_pads"].add(physical.get(node_id, node_id))
        if node_id in plane_nodes:
            group["has_plane"] = True
    for edge in graph.get("edges", []) or []:
        if bool(edge.get("active", True)) and edge.get("type") in ELECTRICAL_EDGE_TYPES:
            root = node_to_group.get(str(edge.get("source", "")), "")
            if root in groups:
                groups[root]["active_edge_count"] += 1
    return {
        "groups": groups,
        "node_to_group": node_to_group,
        "pad_nodes": pad_nodes,
        "physical_pad_map": physical,
    }


def _state_to_json(state: dict[str, Any]) -> dict[str, Any]:
    """Convert graph state sets into stable JSON diagnostics."""

    groups = []
    for group_id, group in state.get("groups", {}).items():
        pads = sorted(group.get("pads", set()))
        if not pads:
            continue
        groups.append({
            "id": group_id,
            "pad_count": len(pads),
            "physical_pad_count": len(group.get("physical_pads", set())),
            "has_plane": bool(group.get("has_plane", False)),
            "active_edge_count": int(group.get("active_edge_count", 0) or 0),
            "pads": pads[:40],
        })
    large = [
        group
        for group in groups
        if group["physical_pad_count"] >= 8 and not group["has_plane"]
    ]
    single = [group for group in groups if group["physical_pad_count"] == 1]
    return {
        "net_like_group_count": len(groups),
        "single_physical_pad_group_count": len(single),
        "large_non_plane_group_count": len(large),
        "largest_physical_pad_group": max((group["physical_pad_count"] for group in groups), default=0),
        "groups": groups[:80],
    }


def _map_pads_physical(
    pad_nodes: set[str],
    pairs: list[HolePair],
    graph: dict[str, Any],
) -> dict[str, str]:
    """Map pad nodes to canonical physical pad ids using hole pairs or via edges."""

    physical = {pad: pad for pad in pad_nodes}
    for pair in pairs:
        if pair.pad_top in pad_nodes:
            physical[pair.pad_top] = pair.pad_top
        if pair.pad_bottom in pad_nodes:
            physical[pair.pad_bottom] = pair.pad_top if pair.pad_top in pad_nodes else pair.pad_bottom
    if pairs:
        return physical
    for edge in graph.get("edges", []) or []:
        if edge.get("type") != "via_pair":
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in pad_nodes and target in pad_nodes:
            top, bottom = (source, target) if source.startswith("TOP:") else (target, source)
            physical[top] = top
            physical[bottom] = top
    return physical


def _graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """Summarize serialized graph node and edge counts by type."""

    node_counter = Counter(str(node.get("type", "")) for node in graph.get("nodes", []) or [])
    edge_counter = Counter(str(edge.get("type", "")) for edge in graph.get("edges", []) or [])
    active_edge_counter = Counter(
        str(edge.get("type", ""))
        for edge in graph.get("edges", []) or []
        if bool(edge.get("active", True)) and edge.get("type") in ELECTRICAL_EDGE_TYPES
    )
    return {
        "node_count": len(graph.get("nodes", []) or []),
        "edge_count": len(graph.get("edges", []) or []),
        "active_electrical_edges": sum(active_edge_counter.values()),
        "nodes_by_type": dict(sorted(node_counter.items())),
        "edges_by_type": dict(sorted(edge_counter.items())),
        "active_electrical_edges_by_type": dict(sorted(active_edge_counter.items())),
    }


def _edge_candidate_id(index: int, edge: dict[str, Any]) -> str:
    """Build a stable candidate id from edge position and endpoints."""

    return f"{edge.get('type', '')}:{index}:{edge.get('source', '')}->{edge.get('target', '')}"


def _append_reason(reason: str, addition: str) -> str:
    """Append a semicolon-separated reason without duplicating the same text."""

    if not reason:
        return addition
    if addition in reason:
        return reason
    return f"{reason}; {addition}"


class _UnionFind:
    """Small union-find structure used to compute connected graph groups."""

    def __init__(self) -> None:
        """Create an empty disjoint-set forest."""

        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        """Add a value as its own parent if it does not exist yet."""

        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        """Return the canonical representative for a value."""

        self.add(value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        """Merge the sets containing two values."""

        self.parent[self.find(second)] = self.find(first)
