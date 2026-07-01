from __future__ import annotations

from pathlib import Path

from schematic_generator.colors import find_suggestions_colors
from schematic_generator.images import load_image
from schematic_generator.platform_support import find_tesseract


# Defaults used when the user chooses the fully automatic analysis path.
AUTOMATIC_FILTER_COMBINATION_COUNT = 15
AUTOMATIC_THRESHOLD_ALIGNMENT = 0.5
AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT = 50
AUTOMATIC_GROUNDPLANE = True

def suggestions_colors_for_image(path: str | Path) -> dict[str, tuple[int, int, int]]:
    """Load one image and return automatic BGR samples for traces and background when detected."""

    suggestions = find_suggestions_colors(load_image(path), 5)
    result: dict[str, tuple[int, int, int]] = {}
    if suggestions.trace_bgr:
        result["trace_bgr"] = suggestions.trace_bgr
    if suggestions.background_bgr:
        result["background_bgr"] = suggestions.background_bgr
    return result


def samples_colors_automatic(
    path_top: str | Path,
    path_bottom: str | Path,
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Build automatic color samples for both PCB sides."""

    return {
        "TOP": suggestions_colors_for_image(path_top),
        "BOTTOM": suggestions_colors_for_image(path_bottom),
    }
