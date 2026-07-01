from __future__ import annotations

import queue
import shutil
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
from schematic_generator.platform_support import open_in_file_manager

from schematic_generator.gui.common import SAMPLE_NAMES, SAMPLE_STEPS, hex_from_bgr, text_on_background, ui_font


class ApplicationAnalysisMixin:
    """Provide application analysis behavior for the main Tk window."""

    def _ask_with_mode_by_import(self) -> None:
        """Ask once per image pair whether to run automatic or manual workflow."""
        if not self.path_top or not self.path_bottom:
            return
        pair = (self.path_top, self.path_bottom)
        if self.last_mode_prompt_pair == pair:
            return
        self.last_mode_prompt_pair = pair
        automatically = self._ask_workflow_mode()
        if automatically:
            self._analyze_automatically()
        else:
            self._log("Manual workflow selected. Choose color samples next.")
            self._set_active_step(2)
            self._start_sample_wizard()

    def _ask_workflow_mode(self) -> bool:
        """Ask for the workflow mode with English buttons independent of OS locale."""
        window = tk.Toplevel(self)
        window.title("Workflow mode")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        result = {"automatic": False}

        ttk.Label(
            window,
            text="Both images are loaded. Run the full workflow automatically?",
            font=ui_font(self, 10, bold=True),
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(14, 6))
        ttk.Label(
            window,
            text=(
                "Yes: color choice, analysis, schematic generation, and optimization will run now.\n\n"
                "No: continue with guided color samples, analysis, and optional manual corrections."
            ),
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        buttons = ttk.Frame(window)
        buttons.pack(fill=tk.X, padx=14, pady=(0, 14))

        def choose(value: bool) -> None:
            result["automatic"] = value
            window.destroy()

        no_button = ttk.Button(buttons, text="No", command=lambda: choose(False))
        yes_button = ttk.Button(buttons, text="Yes", command=lambda: choose(True))
        no_button.pack(side=tk.RIGHT, padx=(8, 0))
        yes_button.pack(side=tk.RIGHT)
        window.bind("<Return>", lambda _event: choose(True))
        window.bind("<Escape>", lambda _event: choose(False))

        def focus_dialog() -> None:
            window.lift()
            window.focus_force()
            yes_button.focus_set()

        window.after_idle(focus_dialog)
        window.wait_window()
        return bool(result["automatic"])

    def _analyze_automatically(self) -> None:
        """Prepare automatic defaults and launch analysis without further color prompts."""
        if not self.path_top or not self.path_bottom:
            messagebox.showwarning("Missing images", "Choose TOP and BOTTOM images.")
            return
        self._set_parameters_automatic()
        self._suggest_samples_for_side("TOP", quiet=True)
        self._suggest_samples_for_side("BOTTOM", quiet=True)
        self.reset_after_automatic_analysis = True
        self._set_active_step(3)
        self._start_analysis(automatically=True)

    def _set_parameters_automatic(self) -> None:
        """Apply default automatic parameters to sliders, labels, and state variables."""
        self.filter_combination_count.set(AUTOMATIC_FILTER_COMBINATION_COUNT)
        self.alignment_threshold.set(AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT)
        self.has_groundplane.set(AUTOMATIC_GROUNDPLANE)
        self.component_solver_mode.set("sequential")
        if hasattr(self, "filter_slider"):
            self.filter_slider.set(AUTOMATIC_FILTER_COMBINATION_COUNT)
        if hasattr(self, "alignment_slider"):
            self.alignment_slider.set(AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT)
        if hasattr(self, "filter_count_label"):
            self.filter_count_label.config(text=str(AUTOMATIC_FILTER_COMBINATION_COUNT))
        if hasattr(self, "alignment_threshold_label"):
            self.alignment_threshold_label.config(text=f"{AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT}%")

    def _analyze(self) -> None:
        """Launch analysis in manual mode."""
        self._start_analysis(automatically=False)

    def _start_analysis(self, automatically: bool) -> None:
        """Validate inputs, freeze controls, capture active settings, and start the worker thread."""
        if not self.path_top or not self.path_bottom:
            messagebox.showwarning("Missing images", "Choose TOP and BOTTOM images.")
            return
        if not automatically and not self._samples_ready():
            messagebox.showwarning(
                "Missing color samples",
                "Click at least one trace/copper color and one soldermask background color in the preview.",
            )
            return
        if not automatically:
            self.reset_after_automatic_analysis = False

        self.log.delete("1.0", "end")
        self.status_preview.set("Analysis in progress...")
        if not self.thumbnail_image:
            self.label_preview.config(image="", text="Analysis in progress...")
        self.button_analysis.config(state="disabled")
        button_auto = getattr(self, "button_auto", None)
        if button_auto is not None:
            button_auto.config(state="disabled")
        self.progress_bar.start(12)
        self.active_combination_count = self.filter_combination_count.get()
        self.active_threshold_alignment = self.alignment_threshold.get()
        self.active_component_solver_mode = self.component_solver_mode.get()
        self.active_mode_automatic = automatically
        self.active_samples_colors = self._copy_samples()
        prefix = "Starting automatic analysis. " if automatically else "Starting manual analysis. "
        self._log(
            prefix +
            f"Filter combinations: {self.active_combination_count}. "
            f"Alignment threshold: {self.active_threshold_alignment}%. "
            f"Component solver: {self.active_component_solver_mode}. "
            f"Groundplane: {'yes' if self.has_groundplane.get() else 'no'}. "
            "This can take from a few seconds to a few minutes."
        )

        worker_thread = threading.Thread(target=self._analysis_in_background, daemon=True)
        worker_thread.start()

    def _analysis_in_background(self) -> None:
        """Run the analysis pipeline off the Tk thread and marshal completion back to the UI."""
        try:
            report = run_analysis(
                self.path_top,
                self.path_bottom,
                self.output_folder,
                filter_combination_count=self.active_combination_count,
                alignment_threshold=self.active_threshold_alignment / 100.0,
                schematic_name=self.schematic_name.get(),
                tesseract_path=self.path_tesseract.get().strip() or None,
                color_samples=self.active_samples_colors,
                has_groundplane=self.has_groundplane.get(),
                component_solver_mode=self.active_component_solver_mode,
                log=self._log_from_thread,
            )
            self.after(0, lambda: self._analysis_finished(report))
        except Exception as error:
            self.after(0, lambda error=error: self._error_analysis(error))

    def _analysis_finished(self, report: AnalysisReport) -> None:
        """Restore controls, store the report, refresh previews, and log result counts."""
        self.progress_bar.stop()
        self.status_preview.set("Analysis complete")
        self.button_analysis.config(state="normal")
        button_auto = getattr(self, "button_auto", None)
        if button_auto is not None:
            button_auto.config(state="normal")
        self.last_report = report
        self._refresh_previews(report)
        self._log(
            "Analysis complete: "
            f"pads TOP={report.count_pads_top}, "
            f"pads BOTTOM={report.count_pads_bottom}, "
            f"pairs={report.count_pairs_holes}, "
            f"nets={report.count_nets}."
        )
        self._log(f"Results: {report.output_folder}")
        if self.reset_after_automatic_analysis:
            self.reset_after_automatic_analysis = False
            self._show_automatic_completion_dialog(report.output_folder)
            self._new_board()
        else:
            self._set_active_step(4)

    def _error_analysis(self, error: Exception) -> None:
        """Restore controls and show an error dialog when analysis fails."""
        self.progress_bar.stop()
        self.status_preview.set("Analysis error")
        self.button_analysis.config(state="normal")
        button_auto = getattr(self, "button_auto", None)
        if button_auto is not None:
            button_auto.config(state="normal")
        self.reset_after_automatic_analysis = False
        self._log(f"Analysis error: {error}")
        messagebox.showerror("Analysis error", str(error))

    def _show_automatic_completion_dialog(self, folder: str) -> None:
        """Show a completion dialog for the fully automatic workflow."""
        window = tk.Toplevel(self)
        window.title("Work complete")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        ttk.Label(
            window,
            text="Work complete.",
            font=ui_font(self, 12, bold=True),
        ).pack(anchor="w", padx=14, pady=(14, 4))
        ttk.Label(
            window,
            text=f"Result folder:\n{folder}",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        buttons = ttk.Frame(window)
        buttons.pack(fill=tk.X, padx=14, pady=(0, 14))

        def open_result_folder() -> None:
            try:
                open_in_file_manager(folder)
            except (OSError, RuntimeError) as error:
                messagebox.showerror("Cannot open folder", str(error), parent=window)
            else:
                window.destroy()

        ttk.Button(buttons, text="OK", command=window.destroy).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(buttons, text="Open result folder", command=open_result_folder).pack(side=tk.RIGHT)
        window.bind("<Return>", lambda _event: window.destroy())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.wait_window()
