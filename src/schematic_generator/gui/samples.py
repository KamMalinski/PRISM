from __future__ import annotations

import os
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

from schematic_generator.gui.common import SAMPLE_NAMES, SAMPLE_STEPS, hex_from_bgr, text_on_background


class ApplicationSamplesMixin:
    """Provide application samples behavior for the main Tk window."""

    def _select_top(self) -> None:
        """Ask for a TOP image and refresh previews and color suggestions for that side."""
        path = self._select_image()
        if path:
            self.path_top = path
            self.lbl_top.config(text=str(path))
            self._log(f"TOP: {path}")
            self._refresh_input_previews("TOP")
            self._suggest_samples_for_side("TOP")
            self._ask_with_mode_by_import()

    def _select_bottom(self) -> None:
        """Ask for a BOTTOM image and refresh previews and color suggestions for that side."""
        path = self._select_image()
        if path:
            self.path_bottom = path
            self.lbl_bottom.config(text=str(path))
            self._log(f"BOTTOM: {path}")
            self._refresh_input_previews("BOTTOM")
            self._suggest_samples_for_side("BOTTOM")
            self._ask_with_mode_by_import()

    def _select_image(self) -> Path | None:
        """Open a file chooser restricted to common image formats and return the selected path."""
        path = filedialog.askopenfilename(
            title="Choose PCB image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        return Path(path) if path else None

    def _suggest_sample_colors(self) -> None:
        """Run automatic color sampling for all available sides and report whether anything was found."""
        found = False
        for side in ("TOP", "BOTTOM"):
            found = self._suggest_samples_for_side(side, quiet=True) or found
        if found:
            self._log("Color samples were suggested. Accept them or click another color in the image or palette.")
        else:
            messagebox.showinfo("Missing images", "Choose a TOP or BOTTOM image first.")

    def _choose_colors_automatically(self) -> None:
        """Accept automatic color suggestions and continue to the analysis step."""
        if not self.path_top or not self.path_bottom:
            messagebox.showinfo("Missing images", "Choose TOP and BOTTOM images first.")
            return
        found = False
        for side in ("TOP", "BOTTOM"):
            found = self._suggest_samples_for_side(side, quiet=True) or found
        if not found:
            messagebox.showinfo("Missing images", "Choose TOP and BOTTOM images first.")
            return
        self._log("Automatic color choice accepted.")
        self._finish_color_step()

    def _suggest_samples_for_side(self, side: str, quiet: bool = False) -> bool:
        """Detect candidate soldermask and copper colors for one side and store them in GUI state."""
        path = self._path_side(side)
        if not path:
            self.candidates_colors[side] = []
            self._refresh_preview_colors()
            return False
        try:
            suggestions = find_suggestions_colors(load_image(path), 5)
        except Exception as error:
            self._log(f"{side}: could not suggest colors: {error}")
            return False
        self.candidates_colors[side] = suggestions.dominant
        data = self.samples_colors.setdefault(side, {})
        if suggestions.trace_bgr:
            data["trace_bgr"] = suggestions.trace_bgr
        if suggestions.background_bgr:
            data["background_bgr"] = suggestions.background_bgr
        if not quiet:
            self._log(
                f"{side}: color suggestions: "
                f"trace/copper={suggestions.trace_bgr}, soldermask background={suggestions.background_bgr}."
            )
        self._refresh_label_samples()
        self._refresh_preview_colors()
        return True

    def _start_sample_wizard(self) -> None:
        """Start the guided color sampling workflow from the first available side."""
        if not self.path_top and not self.path_bottom:
            messagebox.showinfo("Missing images", "Choose a TOP or BOTTOM image first.")
            return
        self.color_step_completed = False
        self.wizard_samples_active = True
        self.wizard_samples_index = 0
        self._advance_to_available_step()

    def _accept_sample_wizard_step(self) -> None:
        """Validate the current wizard sample and advance to the next required sample."""
        if not self.wizard_samples_active:
            self._start_sample_wizard()
            return
        side, key = SAMPLE_STEPS[self.wizard_samples_index]
        if key not in self.samples_colors.get(side, {}):
            messagebox.showinfo("Missing sample", "Click a color in the image or choose a palette swatch.")
            return
        self._next_sample_wizard_step()

    def _set_sample_wizard_step(self) -> None:
        """Switch the preview and instructions to the current wizard sample target."""
        side, key = SAMPLE_STEPS[self.wizard_samples_index]
        self.mode_samples.set(key)
        self._show_input_preview(side)
        current = self.samples_colors.get(side, {}).get(key)
        suffix = f" Current suggestion BGR={current}." if current else ""
        self.sample_wizard_status.set(
            f"Wizard: {side}, choose {SAMPLE_NAMES[key]}.{suffix}"
        )

    def _next_sample_wizard_step(self) -> None:
        """Move the sample wizard index forward and skip unavailable sides."""
        self.wizard_samples_index += 1
        self._advance_to_available_step()

    def _advance_to_available_step(self) -> None:
        """Skip wizard steps for missing images and finish the workflow when all samples are present."""
        while self.wizard_samples_index < len(SAMPLE_STEPS):
            side, _key = SAMPLE_STEPS[self.wizard_samples_index]
            if self._path_side(side):
                self._set_sample_wizard_step()
                return
            self.wizard_samples_index += 1
        if self.wizard_samples_index >= len(SAMPLE_STEPS):
            self.wizard_samples_active = False
            self.sample_wizard_status.set("Sample wizard is complete. You can start the analysis.")
            self._log("Sample wizard is complete.")
            self._finish_color_step()
            return

    def _finish_color_step(self) -> None:
        """Mark color selection complete and move the guided workflow to analysis."""
        if not self._samples_ready():
            messagebox.showinfo(
                "Missing color samples",
                "Choose at least one trace/copper color and one soldermask background color.",
            )
            return
        if self.color_step_completed:
            return
        self.color_step_completed = True
        self.wizard_samples_active = False
        self.sample_wizard_status.set("Color samples are ready. Continue with analysis.")
        self._log("Color samples are ready.")
        self._set_active_step(3)

    def _path_side(self, side: str) -> Path | None:
        """Return the selected input image path for TOP or BOTTOM."""
        return self.path_top if side == "TOP" else self.path_bottom

    def _refresh_label_samples(self) -> None:
        """Update the compact sample status label for both board sides."""
        self.lbl_samples.config(text="\n".join(self._sample_description(side) for side in ("TOP", "BOTTOM")))

    def _sample_description(self, side: str) -> str:
        """Format whether a side has trace and soldermask samples selected."""
        data = self.samples_colors.get(side, {})
        path = "trace" if "trace_bgr" in data else "missing trace"
        background = "background" if "background_bgr" in data else "missing background"
        return f"{side}: {path}, {background}"

    def _save_sample_color(self, side: str, bgr: tuple[int, int, int], source: str) -> None:
        """Store a clicked or palette-selected color sample and advance the wizard if active."""
        key = self.mode_samples.get()
        if self.wizard_samples_active:
            expected_side, expected_key = SAMPLE_STEPS[self.wizard_samples_index]
            if side != expected_side:
                self._show_input_preview(expected_side)
                self._log(f"The wizard expects a sample from {expected_side}. Preview switched.")
                return
            key = expected_key
            self.mode_samples.set(key)
        self.samples_colors.setdefault(side, {})[key] = tuple(int(x) for x in bgr)
        self._log(f"{side}: sample {SAMPLE_NAMES.get(key, key)} from {source} BGR={bgr}.")
        self._refresh_label_samples()
        self._refresh_preview_colors()
        if self.wizard_samples_active:
            self._next_sample_wizard_step()

    def _select_dominant_color(self, side: str, bgr: tuple[int, int, int]) -> None:
        """Select a dominant-color swatch as the active sample for a board side."""
        self._show_input_preview(side)
        self._save_sample_color(side, bgr, "dominant color palette")

    def _refresh_preview_colors(self) -> None:
        """Rebuild the dominant-color swatch buttons from current color candidates."""
        if not hasattr(self, "frame_colors"):
            return
        for child in self.frame_colors.winfo_children():
            child.destroy()
        for side in ("TOP", "BOTTOM"):
            row = ttk.Frame(self.frame_colors)
            row.pack(fill="x", padx=0, pady=3)
            ttk.Label(row, text=side, width=7).pack(side=tk.LEFT)
            colors = self.candidates_colors.get(side, [])
            if not colors:
                ttk.Label(row, text="none").pack(side=tk.LEFT)
                continue
            for color in colors[:5]:
                hex_color = hex_from_bgr(color.bgr)
                text = f"{color.share:.0%}"
                button = tk.Button(
                    row,
                    text=text,
                    width=5,
                    relief=tk.RAISED,
                    background=hex_color,
                    activebackground=hex_color,
                    foreground=text_on_background(color.bgr),
                    activeforeground=text_on_background(color.bgr),
                    command=lambda s=side, b=color.bgr: self._select_dominant_color(s, b),
                )
                button.pack(side=tk.LEFT, padx=(0, 3))

    def _samples_ready(self) -> bool:
        """Check whether manual analysis has enough trace and background color samples."""
        all = [data for data in self.samples_colors.values()]
        return any("trace_bgr" in data for data in all) and any("background_bgr" in data for data in all)

    def _copy_samples(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """Return an immutable copy of selected color samples for the analysis worker."""
        return {
            side: {key: tuple(value) for key, value in data.items()}
            for side, data in self.samples_colors.items()
            if data
        }
