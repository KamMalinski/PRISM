from __future__ import annotations

import json
from pathlib import Path

from schematic_generator.contact_solver.core import ContactSolverResult


def save_solver_diagnostics(folder: str | Path | None, result: ContactSolverResult) -> None:
    """Persist contact solver candidate and summary diagnostics, if a folder is available."""

    if folder is None:
        return
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / "contact_solver_candidates.json").write_text(
        json.dumps(result.candidates_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path / "contact_solver.json").write_text(
        json.dumps(result.summary_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
