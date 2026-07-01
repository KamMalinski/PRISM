from __future__ import annotations

from typing import Any

from schematic_generator.models import HolePair, Pad, Problem

_NAMES_MODES = {
    "move": "Select / move",
    "component": "Select element",
    "add": "Add pad",
    "delete": "Delete pad",
    "rename_pad": "Name pad",
    "pair": "Connect pair",
    "unpair": "Disconnect pair",
    "trace_add": "Add trace",
    "trace_remove": "Delete trace",
    "trace_bridge": "Bridge",
    "trace_cut": "Cut",
    "trace_ignore": "Ignore",
    "ocr_region": "OCR area",
    "ocr_manual_region": "OCR manual",
}

_SHORTCUTS_MODES = {
    "move": "W",
    "component": "E",
    "add": "A",
    "delete": "D",
    "rename_pad": "N",
    "pair": "P",
    "unpair": "R",
    "trace_add": "S",
    "trace_remove": "X",
    "trace_bridge": "B",
    "trace_cut": "C",
    "trace_ignore": "I",
    "ocr_region": "O",
    "ocr_manual_region": "T",
}

_MODE_BY_SHORTCUT = {shortcut.lower(): mode for mode, shortcut in _SHORTCUTS_MODES.items()}

_TYPES_ELEMENTS = ("Resistor", "Capacitor", "Diode", "Inductor", "Transistor", "IC", "Connector", "Device")
_PAD_Y_KEY = "pady"


def _component_type(component: dict[str, Any]) -> str:
    """Read a manual component type."""
    return str(component.get("type", "Device") or "Device")


def _component_value(component: dict[str, Any]) -> str:
    """Read a manual component value."""
    return str(component.get("value", "") or "")


def _component_pads(component: dict[str, Any]) -> list[str]:
    """Read manual component pad node names."""
    return [str(pad) for pad in component.get("pads", [])]


def _problem_category(problem: Problem) -> str:
    """Read the problem group used by filters and labels."""
    return str(problem.category)


def _problem_identifier(problem: Problem) -> str:
    """Read the stable diagnostic identifier used as the tree item id."""
    return str(problem.identifier)


def _problem_description(problem: Problem) -> str:
    """Read the human-readable diagnostic text."""
    return str(problem.description)


def _problem_suggestion(problem: Problem) -> str:
    """Read the proposed diagnostic action."""
    return str(problem.suggestion)


def _problem_related_items(problem: Problem) -> dict[str, list[str]]:
    """Read related pads, nets, or components from a problem object."""
    value = problem.related
    return value if isinstance(value, dict) else {}


def _pad_identifier(pad: Pad) -> str:
    """Read the side-local pad identifier."""
    return str(pad.identifier)


def _pad_confidence(pad: Pad) -> float:
    """Read pad detection confidence."""
    return float(pad.confidence)


def _pair_confidence(pair: HolePair) -> float:
    """Read TOP/BOTTOM pair confidence."""
    return float(pair.confidence)


def _ocr_text(entry: dict[str, Any]) -> str:
    """Read OCR text from a correction entry."""
    return str(entry.get("text", ""))


def _ocr_side(entry: dict[str, Any], default: str = "") -> str:
    """Read the OCR image side."""
    return str(entry.get("side", default))


def _ocr_confidence(entry: dict[str, Any]) -> float:
    """Read OCR confidence."""
    return float(entry.get("confidence", 0.0))


def _normalize_ocr_entry(entry: dict[str, Any], side: str | None = None) -> dict[str, Any]:
    """Copy an OCR dictionary into the editor current English-key schema."""
    normalized = dict(entry)
    normalized["text"] = _ocr_text(entry)
    normalized["side"] = side or _ocr_side(entry, "TOP")
    normalized["confidence"] = _ocr_confidence(entry)
    normalized["rotation"] = float(entry.get("rotation", 0.0))
    return normalized


def _color_net(net: str) -> tuple[int, int, int]:
    """Map a net name to a stable RGB color so redraws keep the same visual identity."""
    if not net:
        return 255, 220, 0
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (0, 128, 128),
        (170, 110, 40),
        (0, 0, 128),
    ]
    return palette[sum(ord(ch) for ch in net) % len(palette)]


def _label_pad(pad: Pad) -> str:
    """Choose the best visible label for a pad, preferring user names over generated identifiers."""
    return (pad.name or _pad_identifier(pad)).strip()


def _short_node(node: str) -> str:
    """Strip the board side prefix from a full pad node name for compact labels."""
    return node.split(":", 1)[-1]


def _cursor_for_mode(mode: str) -> str:
    """Choose the Tk cursor that best communicates the active editing mode."""
    return {
        "move": "fleur",
        "component": "hand2",
        "add": "plus",
        "delete": "X_cursor",
        "rename_pad": "xterm",
        "pair": "tcross",
        "unpair": "X_cursor",
        "trace_add": "crosshair",
        "trace_remove": "X_cursor",
        "trace_bridge": "crosshair",
        "trace_cut": "crosshair",
        "trace_ignore": "X_cursor",
        "ocr_region": "crosshair",
        "ocr_manual_region": "crosshair",
    }.get(mode, "arrow")


def _color_pairs(index: int) -> tuple[int, int, int]:
    """Pick a repeatable highlight color for a TOP/BOTTOM pair index."""
    palette = [
        (255, 99, 71),
        (64, 224, 208),
        (255, 215, 0),
        (124, 252, 0),
        (135, 206, 250),
        (255, 105, 180),
        (221, 160, 221),
        (255, 165, 0),
        (173, 255, 47),
        (176, 196, 222),
        (240, 128, 128),
        (152, 251, 152),
    ]
    return palette[index % len(palette)]


def _color_element(ref: str, index: int) -> tuple[int, int, int]:
    """Pick a repeatable outline color for a manual component reference."""
    base_index = sum(ord(ch) for ch in ref) + index * 37
    palette = [
        (0, 180, 255),
        (255, 120, 0),
        (160, 220, 70),
        (220, 100, 255),
        (255, 80, 120),
        (80, 220, 180),
        (255, 220, 80),
        (130, 170, 255),
    ]
    return palette[base_index % len(palette)]


def _type_from_prefix(text: str, _count_pads: int = 0) -> str | None:
    """Infer a component class from the first alphanumeric character of a reference designator."""
    prefix = ""
    for character in text.upper():
        if character.isalnum():
            prefix = character
            break
    if not prefix:
        return None
    return {
        "R": "Resistor",
        "C": "Capacitor",
        "D": "Diode",
        "L": "Inductor",
        "Q": "Transistor",
        "T": "Transistor",
        "U": "IC",
        "I": "IC",
        "J": "Connector",
        "P": "Connector",
        "K": "Connector",
    }.get(prefix)


def _default_value(type: str, pad_count: int) -> str:
    """Choose a useful default component value placeholder for the selected component type."""
    if type == "Resistor":
        return "R"
    if type == "Capacitor":
        return "C"
    if type == "Diode":
        return "D"
    if type == "Inductor":
        return "L"
    if type == "Connector":
        return f"{pad_count} pin"
    if type == "IC":
        return f"{pad_count} pin"
    return type


def _default_footprint(type: str, pad_count: int) -> str:
    """Choose a simple footprint hint from component type and pad count."""
    if type == "Resistor":
        return "Resistor_SMD:R_0603_1608Metric"
    if type == "Capacitor":
        return "Capacitor_SMD:C_0603_1608Metric"
    if type == "Diode":
        return "Diode_SMD:D_0603_1608Metric"
    if type == "Inductor":
        return "Inductor_SMD:L_0603_1608Metric"
    if type == "Connector":
        return f"Connector:PinHeader_1x{pad_count}_P2.54mm"
    return ""


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
