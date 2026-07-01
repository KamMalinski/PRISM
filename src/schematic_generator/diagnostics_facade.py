from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# The facade intentionally keeps the public diagnostics API in one small file.
# Detailed problem builders, report rendering, and compatibility helpers live in
# diagnostics/ so callers do not need to know about the internal split.
from schematic_generator.connection_graph_facade import build_connection_graph, refresh_net_explanations
from schematic_generator.contact_solver_facade import apply_contact_solver, is_contact_solver, save_solver_diagnostics
from schematic_generator.diagnostics.builder import build_problems
from schematic_generator.diagnostics.common import save_json
from schematic_generator.diagnostics.io import load_manual_corrections, load_problems, save_problems
from schematic_generator.diagnostics.report import build_summary, save_report_html
from schematic_generator.evidence_dataset import build_evidence_dataset
from schematic_generator.models import Element, HolePair, Net, Pad, Problem


def save_diagnostics(
    work_folder: str | Path,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets: list[Net],
    elements: list[Element],
    manual_components: list[dict[str, Any]] | None = None,
    correction_count: int = 0,
    top_mask: np.ndarray | None = None,
    bottom_mask: np.ndarray | None = None,
    plane_top: np.ndarray | None = None,
    plane_bottom: np.ndarray | None = None,
    schematic_validation: dict[str, Any] | None = None,
    expand_contacts_non_green: bool = False,
    contact_solver_mode: str | None = None,
) -> list[Problem]:
    """Save all machine-readable and HTML diagnostics for the current reconstruction state."""
    folder = Path(work_folder)

    # The connectivity graph is the shared evidence source for diagnostics,
    # problem generation, net explanations, and the evidence dataset.
    graph = build_connection_graph(
        top_pads,
        bottom_pads,
        pairs,
        top_mask=top_mask,
        bottom_mask=bottom_mask,
        plane_top=plane_top,
        plane_bottom=plane_bottom,
        nets=nets,
        elements=elements,
        expand_contacts_non_green=expand_contacts_non_green,
    )

    # When enabled, the contact solver mutates the graph with extra inferred
    # contacts. Net explanations must be refreshed after that mutation so the
    # saved graph and explanations describe the same connectivity state.
    if is_contact_solver(contact_solver_mode):
        solver_result = apply_contact_solver(graph, mode=contact_solver_mode, pairs=pairs)
        graph = refresh_net_explanations(solver_result.graph, nets, pairs, top_pads, bottom_pads)
        solver_result.graph = graph
        save_solver_diagnostics(folder, solver_result)

    # Problem generation reads the final graph, not the pre-solver graph, so risk
    # diagnostics and user-visible warnings match the artifacts written below.
    problems = build_problems(folder, top_pads, bottom_pads, pairs, nets, elements, manual_components or [], schematic_validation, graph)

    # Keep every downstream diagnostic artifact in the work folder. JSON files
    # are intended for tools/tests; the HTML report is for quick manual review.
    save_json(folder / "problems.json", [asdict(problem) for problem in problems])
    save_json(folder / "connectivity_graph.json", graph)
    save_json(folder / "net_explanations.json", graph.get("net_explanations", []))
    save_json(folder / "evidence_dataset.json", build_evidence_dataset(top_pads, bottom_pads, pairs, nets, elements, graph, load_manual_corrections(folder)))
    save_json(folder / "summary_corrected.json", build_summary(top_pads, bottom_pads, pairs, nets, elements, problems, correction_count, schematic_validation))

    report_path = folder / "quality_report.html"
    save_report_html(report_path, top_pads, bottom_pads, pairs, nets, elements, problems, correction_count, schematic_validation)
    return problems


__all__ = ["build_problems", "load_problems", "save_diagnostics", "save_problems"]
