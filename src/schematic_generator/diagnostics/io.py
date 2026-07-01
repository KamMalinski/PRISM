from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schematic_generator.diagnostics.common import problem_from_data, save_json
from schematic_generator.models import Problem


def load_problems(work_folder: str | Path) -> list[Problem]:
    """Load diagnostic problems from the work folder, returning an empty list when no file exists."""
    path = Path(work_folder) / "problems.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [problem_from_data(entry) for entry in data]


def save_problems(work_folder: str | Path, problems: list[Problem]) -> None:
    """Persist edited problem statuses back to problems.json."""
    save_json(Path(work_folder) / "problems.json", [asdict(problem) for problem in problems])


def load_manual_corrections(work_folder: Path) -> list[dict[str, Any]]:
    """Load manual correction history for the evidence dataset, ignoring malformed files."""
    path = work_folder / "manual_corrections.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []
