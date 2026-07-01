from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from schematic_generator.models import Element, HolePair, Pad, Problem


def create_problem(
    type: str,
    category: str,
    severity: str,
    description: str,
    related: dict[str, list[str]],
    positions: dict[str, list[float]],
    suggestion: str,
) -> Problem:
    """Create a stable diagnostic problem id from problem type, related items, and rounded positions."""
    rounded_positions = round_positions(positions)
    stable_payload = json.dumps(
        {"type": type, "related": related, "positions": rounded_positions},
        sort_keys=True,
    )
    identifier = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]
    return Problem(
        f"{type}:{identifier}",
        type,
        category,
        severity,
        description,
        "open",
        related,
        rounded_positions,
        suggestion,
    )


def problem_from_data(data: dict[str, Any]) -> Problem:
    """Load one problem from JSON field names."""
    related = data.get("related", {})
    positions = data.get("positions", {})
    return Problem(
        str(data.get("identifier", "")),
        str(data.get("type", "")),
        str(data.get("category", "")),
        str(data.get("severity", "low")),
        str(data.get("description", "")),
        str(data.get("status", "open")),
        {str(key): [str(item) for item in value] for key, value in dict(related).items()},
        {str(key): [float(item) for item in value] for key, value in dict(positions).items()},
        str(data.get("suggestion", "")),
    )


def problem_identifier(problem: Problem) -> str:
    """Read the stable problem identifier."""
    return str(problem.identifier)


def problem_category(problem: Problem) -> str:
    """Read the problem category."""
    return str(problem.category)


def problem_description(problem: Problem) -> str:
    """Read the human-facing problem description."""
    return str(problem.description)


def problem_suggestion(problem: Problem) -> str:
    """Read the human-facing suggested action."""
    return str(problem.suggestion)


def pad_confidence(pad: Pad) -> float:
    """Read pad confidence."""
    return float(pad.confidence)


def pair_confidence(pair: HolePair) -> float:
    """Read pair confidence."""
    return float(pair.confidence)


def element_confidence(element: Element) -> float:
    """Read component confidence."""
    return float(element.confidence)


def median_radius(pads: list[Pad]) -> float:
    """Return a robust pad radius baseline used by pair and unpaired-pad diagnostics."""
    if not pads:
        return 1.0
    radii = sorted(pad.radius for pad in pads)
    return max(1.0, radii[len(radii) // 2])


def pad_positions(pads: list[Pad]) -> dict[str, list[float]]:
    """Return one representative coordinate per side for a list of pads."""
    result: dict[str, list[float]] = {}
    for pad in pads:
        result.setdefault(pad.side, [pad.x, pad.y])
    return round_positions(result)


def round_positions(positions: dict[str, list[float]]) -> dict[str, list[float]]:
    """Round diagnostic positions so stable problem ids do not change due to float noise."""
    return {str(key): [round(float(value[0]), 2), round(float(value[1]), 2)] for key, value in positions.items() if len(value) >= 2}


def severity_sort(severity: str) -> int:
    """Map severity names into deterministic sort order."""
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def save_json(path: Path, data: Any) -> None:
    """Write pretty UTF-8 JSON, creating the parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
