from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from schematic_generator.diagnostics.common import problem_category, problem_description, problem_suggestion
from schematic_generator.models import Element, HolePair, Net, Pad, Problem


def build_summary(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets: list[Net],
    elements: list[Element],
    problems: list[Problem],
    correction_count: int,
    schematic_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build numeric summary values shared by JSON and HTML diagnostic reports."""
    counter = Counter(problem.severity for problem in problems if problem.status == "open")
    pad_type_counter = Counter((pad.type or "pad") for pad in [*top_pads, *bottom_pads])
    return {
        "top_pad_count": len(top_pads),
        "bottom_pad_count": len(bottom_pads),
        "pad_type_counts": dict(sorted(pad_type_counter.items())),
        "pair_count": len(pairs),
        "net_count": len(nets),
        "element_count": len(elements),
        "correction_count": correction_count,
        "open_high_problem_count": counter.get("high", 0),
        "open_medium_problem_count": counter.get("medium", 0),
        "open_low_problem_count": counter.get("low", 0),
        "problem_count": len(problems),
        "schematic_roundtrip_status": (schematic_validation or {}).get("status", ""),
        "schematic_roundtrip_matched_groups": (schematic_validation or {}).get("matched_group_count", 0),
        "schematic_roundtrip_expected_groups": (schematic_validation or {}).get("expected_group_count", 0),
        "schematic_roundtrip_actual_groups": (schematic_validation or {}).get("actual_group_count", 0),
        "schematic_erc_violations": ((schematic_validation or {}).get("erc", {}) or {}).get("violation_count", 0),
    }


def save_report_html(
    path: Path,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets: list[Net],
    elements: list[Element],
    problems: list[Problem],
    correction_count: int,
    schematic_validation: dict[str, Any] | None = None,
) -> None:
    """Write the human-readable reconstruction quality report as HTML."""
    summary = build_summary(top_pads, bottom_pads, pairs, nets, elements, problems, correction_count, schematic_validation)
    solver_section = _component_solver_section(path.parent)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(problem.severity)}</td>"
        f"<td>{html.escape(problem.status)}</td>"
        f"<td>{html.escape(problem_category(problem))}</td>"
        f"<td>{html.escape(problem.type)}</td>"
        f"<td>{html.escape(problem_description(problem))}</td>"
        f"<td>{html.escape(problem_suggestion(problem))}</td>"
        "</tr>"
        for problem in problems
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reconstruction Quality Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
    code {{ background: #eee; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Reconstruction Quality Report</h1>
  <p>TOP pads: <code>{summary['top_pad_count']}</code>,
     BOTTOM pads: <code>{summary['bottom_pad_count']}</code>,
     pairs: <code>{summary['pair_count']}</code>,
     nets: <code>{summary['net_count']}</code>,
     components: <code>{summary['element_count']}</code>,
     corrections: <code>{summary['correction_count']}</code>.</p>
  <p>Pad types: <code>{html.escape(json.dumps(summary['pad_type_counts'], ensure_ascii=False))}</code>.</p>
  <p>Open problems: high={summary['open_high_problem_count']},
     medium={summary['open_medium_problem_count']},
     low={summary['open_low_problem_count']}.</p>
  <p>KiCad schematic: round-trip=<code>{html.escape(str(summary['schematic_roundtrip_status']))}</code>,
     groups=<code>{summary['schematic_roundtrip_matched_groups']}/{summary['schematic_roundtrip_expected_groups']}</code>,
     ERC=<code>{summary['schematic_erc_violations']}</code>.</p>
  {solver_section}
  <h2>Problems</h2>
  <table>
    <thead><tr><th>Severity</th><th>Status</th><th>Category</th><th>Type</th><th>Description</th><th>Suggestion</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _component_solver_section(folder: Path) -> str:
    """Render the optional component solver diagnostics section for the HTML report."""
    path = folder / "component_solver.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return """
  <h2>Candidate Evidence</h2>
  <p>Could not read <code>component_solver.json</code>.</p>
"""
    selected = data.get("selected", []) if isinstance(data.get("selected", []), list) else []
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('id', '')))}</td>"
        f"<td>{html.escape(str(item.get('source', '')))}</td>"
        f"<td>{html.escape(str(item.get('proposed_ref', '')))}</td>"
        f"<td>{html.escape(str(item.get('weight', '')))}</td>"
        f"<td>{html.escape(_score_breakdown(item))}</td>"
        f"<td>{html.escape(', '.join(str(pad) for pad in item.get('pads', [])))}</td>"
        f"<td>{html.escape(', '.join(str(risk) for risk in item.get('risks', [])))}</td>"
        "</tr>"
        for item in selected[:40]
    )
    return f"""
  <h2>Candidate Evidence</h2>
  <p>Component solver: <code>{html.escape(str(data.get('algorithm', '')))}</code>,
     candidates=<code>{int(data.get('candidate_count', 0) or 0)}</code>,
     selected=<code>{int(data.get('selected_count', 0) or 0)}</code>,
     used pads=<code>{int(data.get('used_pad_count', 0) or 0)}</code>.</p>
  <p>Full data: <code>component_candidates.json</code> and <code>component_solver.json</code>.</p>
  <table>
    <thead><tr><th>ID</th><th>Source</th><th>Ref</th><th>Weight</th><th>Scoring</th><th>Pads</th><th>Risks</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
"""


def _score_breakdown(item: dict[str, Any]) -> str:
    """Format selected score fields from a component solver candidate."""
    breakdown = item.get("score_breakdown", {})
    if not isinstance(breakdown, dict):
        return ""
    keys = ("evidence_score", "geometry_score", "net_context_score", "ocr_score", "risk_penalty", "roundtrip_risk_penalty")
    return ", ".join(f"{key}={breakdown.get(key, '')}" for key in keys if key in breakdown)
