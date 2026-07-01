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

from schematic_generator.gui.common import (
    CORRECTED_BOTTOM_CONNECTIONS_PNG,
    CORRECTED_SCHEMATIC_PNG,
    CORRECTED_TOP_CONNECTIONS_PNG,
    OPTIMIZED_CORRECTED_SCHEMATIC_PNG,
    OPTIMIZED_SCHEMATIC_PNG,
    SAMPLE_NAMES,
    SAMPLE_STEPS,
    hex_from_bgr,
    text_on_background,
)


class ApplicationPreviewMixin:
    """Provide application preview behavior for the main Tk window."""

    def _show_input_preview(self, side: str) -> None:
        """Display the original TOP or BOTTOM image in the preview selector."""
        name = f"{side} input"
        if name not in self.traces_previews:
            self._refresh_input_previews(side)
        if name in self.traces_previews:
            self.preview_view.set(name)
            self._show_selected_preview()

    def _show_preview(self, path: str) -> None:
        """Load an image preview, create a thumbnail, and remember coordinate mapping metadata."""
        image = Image.open(path)
        original_size = image.size
        image.thumbnail((760, 520), Image.Resampling.LANCZOS)
        thumbnail_size = image.size
        self.thumbnail_image = ImageTk.PhotoImage(image)
        self.info_preview = {
            "trace": Path(path),
            "original": original_size,
            "thumbnail_image": thumbnail_size,
        }
        self.label_preview.config(image=self.thumbnail_image, text="")

    def _refresh_previews(self, report: AnalysisReport) -> None:
        """Collect generated output previews after analysis and update the preview selector."""
        folder = Path(report.output_folder)
        work = folder / "work"
        candidates = self._candidates_input()
        candidates.update({
            "Schematic": folder / "schematic.png",
            "Optimized schematic": folder / OPTIMIZED_SCHEMATIC_PNG,
            "Corrected schematic": folder / CORRECTED_SCHEMATIC_PNG,
            "Optimized corrected schematic": folder / OPTIMIZED_CORRECTED_SCHEMATIC_PNG,
            "Corrected TOP preview": work / "preview_top_corrected.png",
            "Corrected BOTTOM preview": work / "preview_bottom_corrected.png",
            "Corrected TOP connections": work / CORRECTED_TOP_CONNECTIONS_PNG,
            "Corrected BOTTOM connections": work / CORRECTED_BOTTOM_CONNECTIONS_PNG,
            "Corrected TOP + BOTTOM alignment": work / "alignment_top_bottom_50_corrected.png",
            "Corrected hole alignment": work / "alignment_holes_corrected.png",
            "TOP preview": work / "preview_top.png",
            "BOTTOM preview": work / "preview_bottom.png",
            "TOP colored connections": work / "top_connections_color.png",
            "BOTTOM colored connections": work / "bottom_connections_color.png",
            "Normalized TOP": work / "top_normalized.png",
            "Normalized BOTTOM": work / "bottom_normalized.png",
            "TOP thick traces": work / "top_traces_thick.png",
            "BOTTOM thick traces": work / "bottom_traces_thick.png",
            "TOP + BOTTOM alignment 50%": work / "alignment_top_bottom_50.png",
            "Hole alignment": work / "alignment_holes.png",
            "TOP plane": work / "plane_top.png",
            "BOTTOM plane": work / "plane_bottom.png",
            "TOP trace mask": work / "top_mask.png",
            "BOTTOM trace mask": work / "bottom_mask.png",
        })
        self.traces_previews = {name: path for name, path in candidates.items() if path.exists()}
        names = list(self.traces_previews)
        self.list_previews.config(values=names, state="readonly" if names else "disabled")
        if names:
            self.preview_view.set(names[0])
            self._show_selected_preview()

    def _show_selected_preview(self) -> None:
        """Display the preview currently selected in the combobox."""
        name = self.preview_view.get()
        path = self.traces_previews.get(name)
        if path:
            self._show_preview(str(path))

    def _refresh_input_previews(self, preferred_side: str | None = None) -> None:
        """Add currently selected input images to the preview selector before analysis runs."""
        candidates = self._candidates_input()
        self.traces_previews.update(candidates)
        names = list(self.traces_previews)
        self.list_previews.config(values=names, state="readonly" if names else "disabled")
        preferred = f"{preferred_side} input" if preferred_side else ""
        if preferred in self.traces_previews:
            self.preview_view.set(preferred)
            self._show_selected_preview()
        elif names and not self.preview_view.get():
            self.preview_view.set(names[0])
            self._show_selected_preview()

    def _candidates_input(self) -> dict[str, Path]:
        """Build preview entries for the selected original input images."""
        candidates: dict[str, Path] = {}
        if self.path_top:
            candidates["TOP input"] = self.path_top
        if self.path_bottom:
            candidates["BOTTOM input"] = self.path_bottom
        return candidates

    def _handle_preview_click(self, event: tk.Event) -> None:
        """Convert a thumbnail click to image coordinates and save the sampled BGR color."""
        if not self.info_preview:
            return
        path = Path(self.info_preview["trace"])
        width_original, height_original = self.info_preview["original"]  # type: ignore[misc]
        min_width, min_height = self.info_preview["thumbnail_image"]  # type: ignore[misc]
        x0 = max(0, (self.label_preview.winfo_width() - int(min_width)) // 2)
        y0 = max(0, (self.label_preview.winfo_height() - int(min_height)) // 2)
        x_mini = event.x - x0
        y_mini = event.y - y0
        if x_mini < 0 or y_mini < 0 or x_mini >= int(min_width) or y_mini >= int(min_height):
            return
        x = min(int(width_original) - 1, max(0, round(x_mini * int(width_original) / int(min_width))))
        y = min(int(height_original) - 1, max(0, round(y_mini * int(height_original) / int(min_height))))
        side = self._side_active_preview()
        if not side:
            self._log("Color samples can be picked only from TOP/BOTTOM views.")
            return
        image = load_image(path)
        bgr = tuple(int(v) for v in image[y, x].tolist())
        self._save_sample_color(side, bgr, f"point ({x}, {y})")

    def _side_active_preview(self) -> str | None:
        """Infer whether the currently displayed preview belongs to TOP or BOTTOM."""
        name = self.preview_view.get().upper()
        if "TOP" in name:
            return "TOP"
        if "BOTTOM" in name:
            return "BOTTOM"
        return None
