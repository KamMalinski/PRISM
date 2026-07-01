from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentCandidate:
    """Component hypothesis used by the experimental global solver."""

    identifier: str
    source: str
    pads: tuple[str, ...]
    pin_order: tuple[str, ...]
    score: float
    weight: float
    proposed_ref: str = ""
    proposed_type: str = ""
    proposed_value: str = ""
    proposed_footprint: str = ""
    feature_vector: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize the candidate into stable diagnostic JSON."""

        return {
            "id": self.identifier,
            "source": self.source,
            "pads": list(self.pads),
            "pin_order": list(self.pin_order),
            "proposed_ref": self.proposed_ref,
            "proposed_type": self.proposed_type,
            "proposed_value": self.proposed_value,
            "proposed_footprint": self.proposed_footprint,
            "score": round(float(self.score), 4),
            "weight": round(float(self.weight), 4),
            "feature_vector": self.feature_vector,
            "score_breakdown": {key: round(float(value), 4) for key, value in self.score_breakdown.items()},
            "evidence": list(self.evidence),
            "risks": list(self.risks),
        }


@dataclass(slots=True)
class ComponentSolverResult:
    """Selected candidate set and summary diagnostics."""

    selected: list[ComponentCandidate]
    rejected: list[ComponentCandidate]
    used_pads: set[str]
    total_weight: float
    beam_width: int
    state_count: int

    def to_json(self) -> dict[str, Any]:
        """Serialize solver output and selected candidate diagnostics."""

        return {
            "algorithm": "deterministic_weighted_set_packing_beam",
            "beam_width": self.beam_width,
            "candidate_count": len(self.selected) + len(self.rejected),
            "selected_count": len(self.selected),
            "rejected_count": len(self.rejected),
            "used_pad_count": len(self.used_pads),
            "total_weight": round(float(self.total_weight), 4),
            "state_count": self.state_count,
            "selected_ids": [candidate.identifier for candidate in self.selected],
            "rejected_ids": [candidate.identifier for candidate in self.rejected],
            "selected": [candidate.to_json() for candidate in self.selected],
        }


def select_candidates_globally(
    candidates: list[ComponentCandidate],
    *,
    beam_width: int = 96,
) -> ComponentSolverResult:
    """Choose a non-overlapping component hypothesis set.

    The search is deliberately small and deterministic. It is not an ILP solver;
    it is a weighted set-packing approximation that keeps the best partial
    states after every candidate. This is enough to compare a global selection
    path with the existing sequential recognizer without adding dependencies.
    """

    sorted_candidates = sorted(
        _unique_candidates(candidates),
        key=lambda candidate: (
            -candidate.weight,
            -candidate.score,
            -len(candidate.pads),
            candidate.identifier,
        ),
    )
    states: list[tuple[float, frozenset[str], tuple[int, ...]]] = [(0.0, frozenset(), ())]
    max_states = 1

    for index, candidate in enumerate(sorted_candidates):
        candidate_pads = frozenset(candidate.pads)
        next_states = list(states)
        if candidate.weight > 0.0 and candidate_pads:
            for total, used, selected_indices in states:
                if used.isdisjoint(candidate_pads):
                    next_states.append((total + candidate.weight, used | candidate_pads, (*selected_indices, index)))
        states = _trim_states(next_states, beam_width)
        max_states = max(max_states, len(states))

    best_total, best_used, best_indices = max(
        states,
        key=lambda state: (
            state[0],
            len(state[1]),
            len(state[2]),
            tuple(-idx for idx in state[2]),
        ),
    )
    selected_indices = set(best_indices)
    selected = [sorted_candidates[index] for index in best_indices]
    rejected = [candidate for index, candidate in enumerate(sorted_candidates) if index not in selected_indices]
    return ComponentSolverResult(
        selected=selected,
        rejected=rejected,
        used_pads=set(best_used),
        total_weight=best_total,
        beam_width=beam_width,
        state_count=max_states,
    )


def _trim_states(
    states: list[tuple[float, frozenset[str], tuple[int, ...]]],
    beam_width: int,
) -> list[tuple[float, frozenset[str], tuple[int, ...]]]:
    """Keep only the best beam states, deduplicated by occupied pad set."""

    best_by_used: dict[frozenset[str], tuple[float, frozenset[str], tuple[int, ...]]] = {}
    for state in states:
        previous = best_by_used.get(state[1])
        if previous is None or _ranking_state(state) > _ranking_state(previous):
            best_by_used[state[1]] = state
    ordered = sorted(best_by_used.values(), key=_ranking_state, reverse=True)
    return ordered[: max(1, int(beam_width))]


def _ranking_state(state: tuple[float, frozenset[str], tuple[int, ...]]) -> tuple[float, int, int, tuple[int, ...]]:
    """Return the deterministic ordering key for partial solver states."""

    total, used, selected_indices = state
    # Coverage is a secondary objective. It helps compare equivalent sets without
    # forcing weak candidates into the selected set.
    return (round(total, 6), len(used), len(selected_indices), tuple(-idx for idx in selected_indices))


def _unique_candidates(candidates: list[ComponentCandidate]) -> list[ComponentCandidate]:
    """Collapse candidates that cover the same pads, keeping the strongest one."""

    result: dict[tuple[str, ...], ComponentCandidate] = {}
    for candidate in candidates:
        if len(set(candidate.pads)) != len(candidate.pads):
            continue
        key = tuple(sorted(candidate.pads))
        previous = result.get(key)
        if previous is None or (candidate.weight, candidate.score, candidate.identifier) > (
            previous.weight,
            previous.score,
            previous.identifier,
        ):
            result[key] = candidate
    return list(result.values())
