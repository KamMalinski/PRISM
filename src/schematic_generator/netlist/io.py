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





def save_netlist(path: str | Path, nets: list[Net]) -> None:
    """Write a plain text netlist grouped by generated net name."""

    lines = ["# Generated Netlist", ""]
    for net in nets:
        lines.append(f"NET {net.name}: {' '.join(net.pads)}")
    _save_text(path, "\n".join(lines) + "\n")


def save_devices(path: str | Path, elements: list[Element]) -> None:
    """Write detected component data as a tab-separated table."""

    lines = [
        "ref\ttype\tvalue\tfootprint\tpins\tpin_descriptions\tpin_pad_nodes\tx\ty\trotation\tconfidence\t"
        "decision_source\tdecision_score\tdecision_reasons"
    ]
    for element in elements:
        pins = ",".join(f"{pin}:{net}" for pin, net in element.pins.items())
        descriptions = ",".join(f"{pin}:{description}" for pin, description in element.pin_descriptions.items())
        pin_pad_nodes = ",".join(f"{pin}:{node}" for pin, node in element.pin_pad_nodes.items())
        reasons = ";".join(element.decision_reasons)
        lines.append(
            f"{element.ref}\t{element.type}\t{element.value}\t{element.footprint}\t"
            f"{pins}\t{descriptions}\t{pin_pad_nodes}\t{element.x:.2f}\t{element.y:.2f}\t{element.rotation:.1f}\t"
            f"{element.confidence:.3f}\t{element.decision_source}\t{element.decision_score:.3f}\t{reasons}"
        )
    _save_text(path, "\n".join(lines) + "\n")
