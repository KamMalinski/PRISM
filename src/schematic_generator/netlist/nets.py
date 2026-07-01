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





def build_nets(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    pairs: list[HolePair],
    plane_top: np.ndarray | None = None,
    plane_bottom: np.ndarray | None = None,
    log: LogFn | None = None,
    expand_contacts_non_green: bool = False,
    contact_solver_mode: str | None = None,
    diagnostics_folder: str | Path | None = None,
) -> list[Net]:
    """Build electrical nets from trace masks, pads, copper planes, and paired holes."""

    _log(log, "Netlist: building an explicit connection graph from masks, planes, and hole pairs.")
    top_pads = [pad for pad in top_pads if pad.type not in {"ignore", "mounting_hole"}]
    bottom_pads = [pad for pad in bottom_pads if pad.type not in {"ignore", "mounting_hole"}]
    graph = build_connection_graph(
        top_pads,
        bottom_pads,
        pairs,
        top_mask=top_mask,
        bottom_mask=bottom_mask,
        plane_top=plane_top,
        plane_bottom=plane_bottom,
        expand_contacts_non_green=expand_contacts_non_green,
    )
    contact_solver_mode = str(contact_solver_mode or os.environ.get("SCHEMATIC_GENERATOR_CONTACT_SOLVER") or "off").strip()
    if is_contact_solver(contact_solver_mode):
        contact_solver_result = apply_contact_solver(graph, mode=contact_solver_mode, pairs=pairs)
        graph = contact_solver_result.graph
        save_solver_diagnostics(diagnostics_folder, contact_solver_result)
        _log(
            log,
            "Netlist contact_solver: "
            f"mode={contact_solver_result.mode}, candidates={len(contact_solver_result.candidates)}, "
            f"selected={len(contact_solver_result.selected)}.",
        )
    sets = UnionFind()
    all_pads = {pad.node: pad for pad in [*top_pads, *bottom_pads]}

    for node in graph.get("nodes", []):
        sets.add(str(node.get("id", "")))
    for pad in all_pads.values():
        sets.add(pad.node)

    edges = active_electrical_edges(graph)
    _log(log, f"Netlist: merging {len(edges)} active electrical graph edges.")
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            sets.connect(source, target)

    groups: dict[str, list[str]] = {}
    for node in sets.parent:
        groups.setdefault(sets.find(node), []).append(node)

    nets: list[Net] = []
    counter_n = 1
    counter_gnd = 1
    for nodes in sorted(groups.values()):
        pads_groups = sorted(node for node in nodes if node in all_pads)
        if not pads_groups:
            continue
        has_plane = any(":PLANE:" in node for node in nodes)
        if has_plane:
            name = "GND" if counter_gnd == 1 else f"GND{counter_gnd}"
            counter_gnd += 1
        else:
            name = f"N{counter_n:04d}"
            counter_n += 1
        for node in pads_groups:
            all_pads[node].net = name
        nets.append(Net(name, pads_groups))
    _log(log, f"Netlist: created {len(nets)} nets.")
    return nets
