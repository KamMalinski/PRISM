from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import ImageTk

from schematic_generator.correction_editor.components import CorrectionEditorComponentsMixin
from schematic_generator.correction_editor.drawing import CorrectionEditorDrawingMixin
from schematic_generator.correction_editor.geometry import CorrectionEditorGeometryMixin
from schematic_generator.correction_editor.ocr import CorrectionEditorOcrMixin
from schematic_generator.correction_editor.pads import CorrectionEditorPadsMixin
from schematic_generator.correction_editor.pairs import CorrectionEditorPairsMixin
from schematic_generator.correction_editor.panels import CorrectionEditorProblemPanelMixin
from schematic_generator.correction_editor.tables import CorrectionEditorTablePanelMixin
from schematic_generator.correction_editor.state import CorrectionEditorStateMixin
from schematic_generator.correction_editor.support import _NAMES_MODES
from schematic_generator.correction_editor.traces import CorrectionEditorTracesMixin
from schematic_generator.correction_editor.ui import CorrectionEditorUiMixin
from schematic_generator.corrections_facade import load_state_corrections
from schematic_generator.diagnostics_facade import load_problems
from schematic_generator.resources import apply_window_icon


class CorrectionEditor(
    CorrectionEditorUiMixin,
    CorrectionEditorProblemPanelMixin,
    CorrectionEditorTablePanelMixin,
    CorrectionEditorOcrMixin,
    CorrectionEditorComponentsMixin,
    CorrectionEditorStateMixin,
    CorrectionEditorDrawingMixin,
    CorrectionEditorTracesMixin,
    CorrectionEditorPadsMixin,
    CorrectionEditorPairsMixin,
    CorrectionEditorGeometryMixin,
    tk.Toplevel,
):
    """Interactive TOP/BOTTOM correction window composed from focused behavior mixins."""
    def __init__(
        self,
        parent: tk.Misc,
        output_folder: str | Path,
        on_save: Callable[[], None] | None = None,
    ) -> None:
        """Initializes persisted reconstruction state, editor variables, and the modal correction window in one place."""
        super().__init__(parent)
        self.title("Reconstruction corrections")
        self.geometry("1360x860")
        self.minsize(1080, 700)
        self.window_icon = apply_window_icon(self)

        self.state = load_state_corrections(output_folder)
        self.on_save = on_save
        self.mode = tk.StringVar(value="move")
        self.show_traces = tk.BooleanVar(value=True)
        self.colored_traces = tk.BooleanVar(value=False)
        self.show_pairs = tk.BooleanVar(value=True)
        self.show_nets = tk.BooleanVar(value=True)
        self.show_ocr = tk.BooleanVar(value=False)
        self.brush_size = tk.IntVar(value=10)
        self.type_pad = tk.StringVar(value="pad")
        self.problem_filter = tk.StringVar(value="open")
        self.problem_description = tk.StringVar(value="")
        self.problem_counter = tk.StringVar(value="")
        self.status = tk.StringVar(value="")
        self.label_mode = tk.StringVar(value=_NAMES_MODES["move"])
        self.needs_recalculation = self.state.pending_recalculation

        self.images = self._load_images()
        self.mask = self._load_mask()
        self.problems = load_problems(self.state.work_folder)
        self.canvas_info: dict[str, dict[str, object]] = {}
        self.selected_node: str | None = None
        self.selected_nodes: set[str] = set()
        self.selected_nodes_order: list[str] = []
        self.selected_trace: tuple[str, float, float] | None = None
        self.suppress_propagation_suggestions = False
        self.problem_nodes: set[str] = set()
        self.problem_positions: dict[str, list[float]] = {}
        self.ocr_positions: dict[str, list[float]] = {}
        self.pending_pair_node: str | None = None
        self.trace_start: tuple[str, float, float] | None = None
        self.ocr_region_start: tuple[str, float, float, bool] | None = None
        self.ocr_region_preview: dict[str, list[float]] = {}
        self.ocr_region_busy = False
        self.drag_node: str | None = None
        self.drag_start_pad: tuple[float, float] | None = None
        self.drag_snapshot_pushed = False
        self.trace_dirty = False
        self.undo_stack: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self.zoom = {"TOP": 1.0, "BOTTOM": 1.0}
        self.icons: dict[str, ImageTk.PhotoImage] = {}
        self.mode.trace_add("write", lambda *_args: self._mode_changed())

        self._build_ui()
        self._refresh_problems()
        self._refresh_manual_elements()
        self._refresh_ocr()
        self._redraw_all()
        self._refresh_status()
        self.protocol("WM_DELETE_WINDOW", self._close_editor)
