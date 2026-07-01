from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from schematic_generator.analysis import run_analysis
from schematic_generator.automatic import (
    AUTOMATIC_FILTER_COMBINATION_COUNT,
    AUTOMATIC_GROUNDPLANE,
    AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT,
)
from schematic_generator.colors import ColorCandidate, find_suggestions_colors
from schematic_generator.images import load_image
from schematic_generator.models import AnalysisReport
from schematic_generator.platform_support import executable_file_types, find_tesseract, open_in_file_manager

from schematic_generator.gui.common import SAMPLE_NAMES, SAMPLE_STEPS, hex_from_bgr, text_on_background


class ApplicationActionsMixin:
    """Provide application actions behavior for the main Tk window."""

    def _open_folder(self) -> None:
        """Open or create the current result directory in the system file explorer."""
        folder = self.last_report.output_folder if self.last_report else self.output_folder
        Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            open_in_file_manager(folder)
        except (OSError, RuntimeError) as error:
            messagebox.showerror("Cannot open folder", str(error))

    def _show_netlist(self) -> None:
        """Open the generated or corrected netlist in a read-only text window."""
        if not self.last_report:
            messagebox.showinfo("Missing results", "Run analysis first.")
            return
        path = Path(self.last_report.file_netlist)
        work = Path(self.last_report.output_folder) / "work"
        path_by_corrections = work / "netlist_corrected.txt"
        if path_by_corrections.exists():
            path = path_by_corrections
        content = path.read_text(encoding="utf-8")
        window = tk.Toplevel(self)
        window.title(f"Netlist - {path.name}")
        area = tk.Text(window, width=100, height=35, wrap="none")
        area.pack(fill=tk.BOTH, expand=True)
        area.insert("1.0", content)
        area.config(state="disabled")

    def _open_editor(self) -> None:
        """Open the correction editor for the last analysis output."""
        if not self.last_report:
            messagebox.showinfo("Missing results", "Run analysis first.")
            return
        from schematic_generator.correction_editor_facade import CorrectionEditor

        editor = CorrectionEditor(self, self.last_report.output_folder, self._corrections_saved)
        editor.bind("<Destroy>", lambda event, window=editor: self._editor_closed(event, window))

    def _corrections_saved(self) -> None:
        """Refresh previews and log state after the correction editor saves changes."""
        if not self.last_report:
            return
        self._refresh_previews(self.last_report)
        self._log("Corrections recalculated: previews, netlist, and optimized corrected schematic were refreshed.")
        self._set_active_step(5)

    def _editor_closed(self, event: tk.Event, editor: tk.Toplevel) -> None:
        """Move to final actions when the correction editor window closes."""
        if event.widget is editor and self.last_report:
            self._set_active_step(5)

    def _skip_manual_corrections(self) -> None:
        """Skip optional correction review and show final result actions."""
        if not self.last_report:
            messagebox.showinfo("Missing results", "Run analysis first.")
            return
        self._log("Manual corrections skipped.")
        self._set_active_step(5)

    def _new_board(self) -> None:
        """Reset the GUI to the initial import state without deleting previous results."""
        self.path_top = None
        self.path_bottom = None
        self.last_report = None
        self.traces_previews = {}
        self.samples_colors = {"TOP": {}, "BOTTOM": {}}
        self.candidates_colors = {"TOP": [], "BOTTOM": []}
        self.wizard_samples_active = False
        self.wizard_samples_index = 0
        self.color_step_completed = False
        self.reset_after_automatic_analysis = False
        self.info_preview = None
        self.thumbnail_image = None
        self.last_mode_prompt_pair = None
        self.schematic_name.set("")
        self.preview_view.set("")
        self.status_preview.set("")
        self.sample_wizard_status.set("Automatic suggestions will be available after choosing images.")
        self.lbl_top.config(text="---")
        self.lbl_bottom.config(text="---")
        self.label_preview.config(image="", text="")
        self.list_previews.config(values=[], state="disabled")
        self.progress_bar.stop()
        self.button_analysis.config(state="normal")
        self._set_parameters_automatic()
        self._refresh_label_samples()
        self._refresh_preview_colors()
        self.log.delete("1.0", "end")
        self._log("Choose TOP and BOTTOM images to start.")
        self._set_active_step(1)

    def _log(self, text: str) -> None:
        """Append a message to the GUI log area and scroll to the end."""
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _log_from_thread(self, text: str) -> None:
        """Schedule a worker-thread log message on the Tk event loop."""
        self.after(0, lambda: self._log(text))

    def _change_count_filters(self, value: str) -> None:
        """Clamp and display the selected filter-combination count."""
        count = max(1, min(100, int(round(float(value)))))
        self.filter_combination_count.set(count)
        label = getattr(self, "filter_count_label", None)
        if label is not None:
            label.config(text=str(count))

    def _change_threshold_alignment(self, value: str) -> None:
        """Clamp and display the selected alignment threshold percentage."""
        percent = max(10, min(100, int(round(float(value)))))
        self.alignment_threshold.set(percent)
        label = getattr(self, "alignment_threshold_label", None)
        if label is not None:
            label.config(text=f"{percent}%")

    def _select_tesseract(self) -> None:
        """Ask for a tesseract executable and store the selected path."""
        path = filedialog.askopenfilename(
            title="Choose Tesseract executable",
            filetypes=executable_file_types("Tesseract", "tesseract"),
        )
        if path:
            self.path_tesseract.set(path)

    def _detect_tesseract_on_startup(self) -> None:
        """Try to locate tesseract automatically and log the OCR setup status."""
        path = self._find_tesseract()
        if path:
            self.path_tesseract.set(str(path))
            self._log(f"Tesseract found: {path}")
        else:
            self._log("Tesseract not found. Select its executable manually to enable OCR.")

    def _find_tesseract(self) -> Path | None:
        """Search PATH and conventional locations for the current operating system."""
        return find_tesseract()
