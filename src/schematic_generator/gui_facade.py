from __future__ import annotations

import tkinter as tk
from pathlib import Path

from schematic_generator.automatic import (
    AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT,
    AUTOMATIC_FILTER_COMBINATION_COUNT,
)
from schematic_generator.colors import ColorCandidate
from schematic_generator.gui.actions import ApplicationActionsMixin
from schematic_generator.gui.analysis import ApplicationAnalysisMixin
from schematic_generator.gui.preview import ApplicationPreviewMixin
from schematic_generator.gui.samples import ApplicationSamplesMixin
from schematic_generator.gui.ui import ApplicationUiMixin
from schematic_generator.models import AnalysisReport
from schematic_generator.platform_support import default_output_directory
from schematic_generator.resources import apply_window_icon


class Application(
    ApplicationUiMixin,
    ApplicationSamplesMixin,
    ApplicationPreviewMixin,
    ApplicationAnalysisMixin,
    ApplicationActionsMixin,
    tk.Tk,
):
    """Main Tk window that wires together image selection, analysis, previews, and export actions."""

    def __init__(self) -> None:
        """Initialize widget state and build the window contents."""
        super().__init__()
        self.title("PRISM")
        self.geometry("1100x760")
        self.minsize(900, 620)
        self.window_icon = apply_window_icon(self)

        self.path_top: Path | None = None
        self.path_bottom: Path | None = None
        self.output_folder = default_output_directory()
        self.last_report: AnalysisReport | None = None
        self.traces_previews: dict[str, Path] = {}
        self.preview_view = tk.StringVar()
        self.schematic_name = tk.StringVar()
        self.path_tesseract = tk.StringVar()
        self.mode_samples = tk.StringVar(value="trace_bgr")
        self.has_groundplane = tk.BooleanVar(value=True)
        self.status_preview = tk.StringVar(value="")
        self.filter_combination_count = tk.IntVar(value=AUTOMATIC_FILTER_COMBINATION_COUNT)
        self.alignment_threshold = tk.IntVar(value=AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT)
        self.component_solver_mode = tk.StringVar(value="sequential")
        self.active_combination_count = AUTOMATIC_FILTER_COMBINATION_COUNT
        self.active_threshold_alignment = AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT
        self.active_component_solver_mode = "sequential"
        self.active_mode_automatic = False
        self.active_samples_colors: dict[str, dict[str, tuple[int, int, int]]] = {}
        self.samples_colors: dict[str, dict[str, tuple[int, int, int]]] = {"TOP": {}, "BOTTOM": {}}
        self.candidates_colors: dict[str, list[ColorCandidate]] = {"TOP": [], "BOTTOM": []}
        self.wizard_samples_active = False
        self.wizard_samples_index = 0
        self.sample_wizard_status = tk.StringVar(value="Automatic suggestions will be available after choosing images.")
        self.info_preview: dict[str, object] | None = None
        self.thumbnail_image = None
        self.last_mode_prompt_pair: tuple[Path, Path] | None = None
        self.reset_after_automatic_analysis = False
        self.color_step_completed = False

        self._build_interface()
        self._detect_tesseract_on_startup()


def run_gui() -> None:
    """Start the desktop GUI event loop."""

    application = Application()
    application.mainloop()
