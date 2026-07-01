from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from schematic_generator.diagnostics.common import (
    create_problem,
    median_radius,
    pad_positions,
    problem_category,
    problem_identifier,
    severity_sort,
)
from schematic_generator.diagnostics.io import load_problems
from schematic_generator.models import Element, HolePair, Net, Pad, Problem


def build_problems(
    work_folder: str | Path,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
    nets: list[Net],
    elements: list[Element],
    manual_components: list[dict[str, Any]],
    schematic_validation: dict[str, Any] | None = None,
    connection_graph: dict[str, Any] | None = None,
) -> list[Problem]:
    """Build all diagnostics and preserve user statuses from an existing problems.json file."""
    previous_statuses = {
        problem_identifier(problem): problem.status
        for problem in load_problems(work_folder)
        if problem.status in {"ignored", "checked", "resolved"}
    }
    pads = [*top_pads, *bottom_pads]
    pads_by_node = {pad.node: pad for pad in pads}
    paired_nodes = {pair.pad_top for pair in pairs} | {pair.pad_bottom for pair in pairs}
    problems: list[Problem] = []

    _add_pad_problems(problems, pads, paired_nodes)
    _add_pair_problems(problems, pairs, pads_by_node)
    _add_net_problems(problems, nets, pads, pads_by_node)
    problems.extend(_risky_trace_problems(connection_graph, pads_by_node))
    _add_element_problems(problems, elements)
    _add_manual_component_problems(problems, manual_components)
    problems.extend(_schematic_validation_problems(schematic_validation))

    for problem in problems:
        identifier = problem_identifier(problem)
        if identifier in previous_statuses:
            problem.status = previous_statuses[identifier]
    problems.sort(key=lambda item: (severity_sort(item.severity), problem_category(item), item.type, problem_identifier(item)))
    return problems


def _add_pad_problems(problems: list[Problem], pads: list[Pad], paired_nodes: set[str]) -> None:
    """Add diagnostics for pads that are isolated or missing a TOP/BOTTOM counterpart."""
    radius_baseline = median_radius(pads)
    for pad in pads:
        if pad.type in {"ignore", "mounting_hole"}:
            continue
        if not pad.net or pad.net == "NET?":
            problems.append(create_problem(
                "pad_no_net", "pads", "high",
                f"Pad {pad.node} is not assigned to any net.",
                {"pads": [pad.node]}, {pad.side: [pad.x, pad.y]},
                "Inspect the trace mask near this pad or remove a false positive pad.",
            ))
        if pad.node not in paired_nodes and pad.type in {"pad", "via"} and pad.radius >= radius_baseline * 0.75:
            problems.append(create_problem(
                "unpaired_pad", "pads", "medium",
                f"Pad {pad.node} has no TOP/BOTTOM pair.",
                {"pads": [pad.node]}, {pad.side: [pad.x, pad.y]},
                "If this is a through-hole or via, connect it to the matching pad on the opposite side.",
            ))


def _add_pair_problems(problems: list[Problem], pairs: list[HolePair], pads_by_node: dict[str, Pad]) -> None:
    """Add diagnostics for missing or geometrically suspicious TOP/BOTTOM pairs."""
    for pair in pairs:
        top = pads_by_node.get(pair.pad_top)
        bottom = pads_by_node.get(pair.pad_bottom)
        if not top or not bottom:
            problems.append(create_problem(
                "pair_missing_pad", "pads", "high",
                f"Pair {pair.pad_top} <-> {pair.pad_bottom} references a missing pad.",
                {"pads": [pair.pad_top, pair.pad_bottom]}, {},
                "Disconnect the pair or recreate the missing pad.",
            ))
            continue
        tolerance = max(12.0, 3.0 * max(top.radius, bottom.radius))
        if pair.distance > tolerance:
            problems.append(create_problem(
                "pair_large_error", "pads", "high",
                f"Pair {pair.pad_top} <-> {pair.pad_bottom} has a large alignment error of {pair.distance:.1f}px.",
                {"pads": [pair.pad_top, pair.pad_bottom]}, {"TOP": [top.x, top.y], "BOTTOM": [bottom.x, bottom.y]},
                "Check the TOP/BOTTOM pair alignment.",
            ))


def _add_net_problems(problems: list[Problem], nets: list[Net], pads: list[Pad], pads_by_node: dict[str, Pad]) -> None:
    """Add diagnostics for suspicious nets produced by mask segmentation."""
    for net in nets:
        net_pads = [pads_by_node[node] for node in net.pads if node in pads_by_node]
        if len(net_pads) == 1:
            pad = net_pads[0]
            problems.append(create_problem(
                "single_pad_net", "nets", "medium",
                f"Net {net.name} contains only one pad.",
                {"nets": [net.name], "pads": [pad.node]}, {pad.side: [pad.x, pad.y]},
                "Check whether the trace mask is broken or the pad is a false positive.",
            ))
        if len(net_pads) > max(12, int(len(pads) * 0.20)) and not net.name.upper().startswith("GND"):
            problems.append(create_problem(
                "large_non_plane_net", "nets", "high",
                f"Net {net.name} connects {len(net_pads)} pads and does not look like a named plane.",
                {"nets": [net.name], "pads": [pad.node for pad in net_pads[:20]]}, pad_positions(net_pads[:4]),
                "Check whether soldermask or background was detected as copper.",
            ))
        if net.name.upper().startswith("GND") and len(net_pads) > max(20, int(len(pads) * 0.35)):
            problems.append(create_problem(
                "huge_plane_net", "nets", "medium",
                f"Plane {net.name} connects a very large number of pads ({len(net_pads)}).",
                {"nets": [net.name], "pads": [pad.node for pad in net_pads[:20]]}, pad_positions(net_pads[:4]),
                "Check whether the ground plane shorted too many pads.",
            ))


def _add_element_problems(problems: list[Problem], elements: list[Element]) -> None:
    """Add diagnostics for duplicate references, missing pins, floating pins, and suspicious shorts."""
    refs = Counter(element.ref for element in elements if element.ref)
    for ref, count in refs.items():
        if count > 1:
            problems.append(create_problem("duplicate_ref", "components", "high", f"Reference {ref} appears {count} times.", {"components": [ref]}, {}, "Change the reference of one component."))
    for element in elements:
        if not element.pins:
            problems.append(create_problem("element_no_pins", "components", "high", f"Component {element.ref} has no pins.", {"components": [element.ref]}, {"TOP": [element.x, element.y]}, "Fix pad assignment for this component."))
            continue
        if any(not net or net == "NET?" for net in element.pins.values()):
            problems.append(create_problem("element_pin_without_net", "components", "medium", f"Component {element.ref} has a pin without a net.", {"components": [element.ref]}, {"TOP": [element.x, element.y]}, "Check the trace mask near this component's pins."))
        if len(element.pins) == 2 and len(set(element.pins.values())) == 1:
            net_name = next(iter(element.pins.values()))
            problems.append(create_problem("two_pin_short", "components", "high", f"Two-pin component {element.ref} has both pins on the same net {net_name}.", {"components": [element.ref], "nets": [net_name]}, {"TOP": [element.x, element.y]}, "Check whether the component pads were shorted by the trace mask."))
        if element.type in {"Resistor", "Capacitor", "Diode", "Inductor"} and len(element.pins) != 2:
            problems.append(create_problem("component_pin_count_mismatch", "components", "medium", f"Component {element.ref} of type {element.type} has {len(element.pins)} pins instead of 2.", {"components": [element.ref]}, {"TOP": [element.x, element.y]}, "Fix the pad set assigned to this component."))


def _add_manual_component_problems(problems: list[Problem], manual_components: list[dict[str, Any]]) -> None:
    """Add diagnostics for manually entered passive components with the wrong pad count."""
    for component in manual_components:
        component_type = str(component.get("type", ""))
        component_pads = [str(pad) for pad in component.get("pads", [])]
        if component_type in {"Resistor", "Capacitor", "Diode", "Inductor"} and len(component_pads) != 2:
            ref = str(component.get("ref", "?"))
            problems.append(create_problem("manual_component_pin_count_mismatch", "components", "medium", f"Manual component {ref} of type {component_type} has {len(component_pads)} pads.", {"components": [ref], "pads": component_pads}, {}, "Edit the component or select exactly two pads."))


def _risky_trace_problems(connection_graph: dict[str, Any] | None, pads_by_node: dict[str, Pad]) -> list[Problem]:
    """Create informational diagnostics for large trace components that touch many pads."""
    if not connection_graph:
        return []
    problems: list[Problem] = []
    for node in connection_graph.get("nodes", []):
        if node.get("type") != "trace_component":
            continue
        attrs = node.get("attrs", {}) if isinstance(node.get("attrs", {}), dict) else {}
        if not attrs.get("risky_large_trace_component"):
            continue
        pads = [str(pad) for pad in attrs.get("pads_touching_nodes", []) if str(pad)]
        pad_count = int(attrs.get("pads_touching", len(pads)) or len(pads))
        limit = int(attrs.get("pad_activation_limit", 0) or 0)
        pin_pairs = int(attrs.get("potential_pin_pairs", 0) or 0)
        problem = create_problem(
            "risky_trace_merge", "nets", "low",
            f"Active trace component {node.get('id')} touches {pad_count} pads (limit {limit}) and may create up to {pin_pairs} pin-to-pin pairs.",
            {"pads": pads[:20], "traces": [str(node.get("id", ""))]}, pad_positions([pads_by_node[pad] for pad in pads[:4] if pad in pads_by_node]),
            "Inspect net_explanations.json and connectivity_graph.json before trusting this merge.",
        )
        problem.status = "checked"
        problems.append(problem)
    return problems


def _schematic_validation_problems(validation: dict[str, Any] | None) -> list[Problem]:
    """Convert KiCad round-trip and ERC validation results into diagnostics."""
    if not validation:
        return []
    status = str(validation.get("status", ""))
    problems: list[Problem] = []
    if status == "unavailable":
        problems.append(create_problem("schematic_roundtrip_unavailable", "schematic", "low", "KiCad schematic validation was not run because kicad-cli is unavailable.", {}, {}, "Add kicad-cli to PATH or install KiCad in a standard location."))
    elif status == "error":
        problems.append(create_problem("schematic_roundtrip_error", "schematic", "high", "KiCad could not export a netlist from the generated schematic.", {}, {}, "Open the schematic in KiCad and check syntax or the round-trip report."))
    elif status == "fail":
        missing = int(len(validation.get("missing_groups", [])))
        extra = int(len(validation.get("extra_groups", [])))
        problems.append(create_problem("schematic_roundtrip_mismatch", "schematic", "high", f"The KiCad-exported netlist does not match the program model: missing groups={missing}, extra groups={extra}.", {}, {}, "Use schematic.roundtrip.json to locate missing or incorrectly connected pins."))
    erc = validation.get("erc", {}) if isinstance(validation.get("erc", {}), dict) else {}
    erc_count = int(erc.get("violation_count", 0) or 0)
    erc_errors = int((erc.get("by_severity", {}) or {}).get("error", 0) or 0)
    if erc_errors:
        by_type = erc.get("by_type", {})
        top_types = ", ".join(f"{type}:{count}" for type, count in list(by_type.items())[:6])
        problems.append(create_problem("schematic_erc_violations", "schematic", "high", f"KiCad ERC reported {erc_errors} schematic errors and {erc_count} total violations ({top_types}).", {}, {}, "Check schematic.erc.json; warnings are reported separately, but ERC errors require fixing export or connectivity."))
    return problems
