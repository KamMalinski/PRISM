from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from schematic_generator.kicad_facade import _should_skip_schematic_element
from schematic_generator.models import Element
from schematic_generator.platform_support import find_kicad_cli

LogFn = Callable[[str], None]


def validate_kicad_schematic(
    schematic_path: str | Path,
    elements: list[Element],
    log: LogFn | None = None,
) -> dict[str, Any]:
    """Exports a KiCad netlist from the generated schematic and compares it with program nets."""

    path = Path(schematic_path)
    report_path = path.with_suffix(".roundtrip.json")
    expected_groups = _groups_pins_from_elements(elements)
    expected_pins = sorted({pin for group in expected_groups for pin in group})
    cli = find_kicad_cli()
    report: dict[str, Any] = {
        "status": "unavailable",
        "reason": "",
        "kicad_cli": str(cli) if cli else "",
        "expected_group_count": len(expected_groups),
        "expected_pin_count": len(expected_pins),
        "actual_group_count": 0,
        "actual_pin_count": 0,
        "matched_group_count": 0,
        "missing_groups": [list(group) for group in expected_groups],
        "extra_groups": [],
        "missing_pins": expected_pins,
        "extra_pins": [],
        "erc": {"status": "not_run", "violation_count": 0, "by_severity": {}, "by_type": {}},
    }
    if not cli:
        report["reason"] = "kicad-cli was not found in PATH or in conventional locations for this platform."
        _save_json(report_path, report)
        _log(log, "Schematic validation: skipped because kicad-cli is unavailable.")
        return report

    export_path = path.with_suffix(".roundtrip.net")
    erc_path = path.with_suffix(".erc.json")
    export_result = _run([
        str(cli),
        "sch",
        "export",
        "netlist",
        "--format",
        "kicadxml",
        "--output",
        str(export_path),
        str(path),
    ])
    report["export"] = _process_result(export_result)
    if export_result.returncode != 0 or not export_path.exists():
        report["status"] = "error"
        report["reason"] = "KiCad did not export a netlist from the schematic."
        _save_json(report_path, report)
        _log(log, f"Schematic validation: KiCad netlist export failed ({export_result.returncode}).")
        return report

    actual_groups = _groups_pins_from_netlist_kicad(export_path)
    actual_pins = sorted({pin for group in actual_groups for pin in group})
    comparison = _compare_groups(expected_groups, actual_groups)
    report.update({
        "status": "pass" if not comparison["missing_groups"] and not comparison["extra_groups"] else "fail",
        "reason": "",
        "actual_group_count": len(actual_groups),
        "actual_pin_count": len(actual_pins),
        "matched_group_count": comparison["matched_group_count"],
        "missing_groups": [list(group) for group in comparison["missing_groups"]],
        "extra_groups": [list(group) for group in comparison["extra_groups"]],
        "missing_pins": sorted(set(expected_pins) - set(actual_pins)),
        "extra_pins": sorted(set(actual_pins) - set(expected_pins)),
    })

    erc_result = _run([
        str(cli),
        "sch",
        "erc",
        "--format",
        "json",
        "--output",
        str(erc_path),
        str(path),
    ])
    report["erc"] = _summarize_erc(erc_path, erc_result)
    _save_json(report_path, report)
    _log(
        log,
        "Schematic validation: "
        f"status={report['status']}, groups {report['matched_group_count']}/"
        f"{report['expected_group_count']}, ERC={report['erc'].get('violation_count', 0)}.",
    )
    return report

def _groups_pins_from_elements(elements: list[Element]) -> list[tuple[str, ...]]:
    """Build expected connected pin groups from generated schematic elements."""

    groups: dict[str, list[str]] = {}
    for element in elements:
        if _should_skip_schematic_element(element):
            continue
        for pin, net in element.pins.items():
            if not net or net == "NET?":
                continue
            groups.setdefault(net, []).append(f"{element.ref}:{pin}")
    return _normalize_groups(groups.values())


def _groups_pins_from_netlist_kicad(netlist_path: Path) -> list[tuple[str, ...]]:
    """Read connected pin groups from a KiCad XML netlist export."""

    root = ET.parse(netlist_path).getroot()
    groups: list[list[str]] = []
    for net in root.findall(".//nets/net"):
        group = []
        for node in net.findall("node"):
            ref = node.attrib.get("ref", "")
            pin = node.attrib.get("pin", "")
            if ref and pin:
                group.append(f"{ref}:{pin}")
        if group:
            groups.append(group)
    return _normalize_groups(groups)


def _normalize_groups(groups: Any) -> list[tuple[str, ...]]:
    """Sort and deduplicate groups, keeping only groups with at least two pins."""

    return sorted(
        {tuple(sorted(str(pin) for pin in group)) for group in groups if len(group) >= 2},
        key=lambda group: (len(group), group),
    )


def _compare_groups(expected: list[tuple[str, ...]], actual: list[tuple[str, ...]]) -> dict[str, Any]:
    """Compare expected and actual pin groups while preserving duplicate group counts."""

    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    common = expected_counter & actual_counter
    missing = expected_counter - actual_counter
    extra = actual_counter - expected_counter
    return {
        "matched_group_count": int(sum(common.values())),
        "missing_groups": _expand_counter(missing),
        "extra_groups": _expand_counter(extra),
    }


def _expand_counter(counter: Counter[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Expand a Counter of tuple groups back into a sorted list with duplicates."""

    result: list[tuple[str, ...]] = []
    for group, count in sorted(counter.items(), key=lambda item: (len(item[0]), item[0])):
        result.extend([group] * int(count))
    return result


def _summarize_erc(erc_path: Path, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Summarize KiCad ERC JSON output and include process diagnostic tails."""

    summary: dict[str, Any] = {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "violation_count": 0,
        "by_severity": {},
        "by_type": {},
    }
    if not erc_path.exists():
        return summary
    try:
        data = json.loads(erc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary["status"] = "parse_error"
        return summary
    violations = []
    for sheet in data.get("sheets", []):
        violations.extend(sheet.get("violations", []))
    summary["violation_count"] = len(violations)
    summary["by_severity"] = dict(sorted(Counter(v.get("severity", "unknown") for v in violations).items()))
    summary["by_type"] = dict(sorted(Counter(v.get("type", "unknown") for v in violations).items()))
    return summary


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture text output without raising on non-zero exit."""

    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _process_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Return compact process diagnostics suitable for JSON reports."""

    return {
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _tail(text: str, limit: int = 4000) -> str:
    """Return the last part of a process output string for diagnostics."""

    return text[-limit:] if text else ""


def _save_json(path: Path, data: Any) -> None:
    """Write JSON diagnostics with stable formatting and UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _log(log: LogFn | None, text: str) -> None:
    """Send a validation diagnostic message when a logger callback is available."""

    if log:
        log(text)
