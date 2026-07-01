from __future__ import annotations

import copy
import math
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from schematic_generator.correction_editor.support import *
from schematic_generator.corrections_facade import (
    add_correction,
    add_manual_element,
    recalculate_and_save_by_corrections,
    save_mask_by_corrections,
    save_state_corrections_without_recalculation,
    find_similar_pads,
)
from schematic_generator.models import Pad
from schematic_generator.netlist_facade import build_nets
from schematic_generator.ocr import read_texts, save_texts_ocr as save_ocr_texts
from schematic_generator.preview import draw_connections_colored


class CorrectionEditorOcrMixin:
    def _add_ocr_manually(
        self,
        side: str = "TOP",
        x: float = 0.0,
        y: float = 0.0,
        w: float = 40.0,
        h: float = 16.0,
        text_start: str = "",
    ) -> None:
        """Open a small dialog for inserting text for a manually selected OCR rectangle."""
        if (x, y, w, h) == (0.0, 0.0, 40.0, 16.0) and not text_start:
            self._start_manual_ocr_region()
            return
        window = tk.Toplevel(self)
        window.title("Add OCR text manually")
        window.resizable(False, False)
        side = side if side in {"TOP", "BOTTOM"} else "TOP"
        text_var = tk.StringVar(value=text_start)
        x0 = int(round(x))
        y0 = int(round(y))
        width = max(1, int(round(w)))
        height = max(1, int(round(h)))

        ttk.Label(window, text="Area").grid(row=0, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: (8, 2)})
        ttk.Label(window, text=f"{side}: x={x0}, y={y0}, w={width}, h={height}").grid(row=0, column=1, sticky="w", padx=8, **{_PAD_Y_KEY: (8, 2)})
        ttk.Label(window, text="Text").grid(row=1, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: 2})
        ttk.Entry(window, textvariable=text_var, width=28).grid(row=1, column=1, padx=8, **{_PAD_Y_KEY: 2})

        def save() -> None:
            """Validate the modal form, apply the requested edit to correction state, and close the dialog after a successful save."""
            text = text_var.get().strip().upper()
            if not text:
                messagebox.showerror("No text", "Enter text.", parent=window)
                return
            entry = {
                "side": side,
                "text": text,
                "confidence": 100.0,
                "x": x0,
                "y": y0,
                "w": width,
                "h": height,
                "rotation": 0,
                "variant": "manual",
            }
            self._add_entry_ocr(entry, "manual_ocr", f"Added manual OCR text '{text}'.")
            window.destroy()

        buttons = ttk.Frame(window)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=8, **{_PAD_Y_KEY: 8})
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT)

    def _add_entry_ocr(self, entry: dict[str, Any], type_corrections: str, description: str) -> None:
        """Append a normalized OCR entry, persist the OCR table, record the correction, and highlight the added box."""
        self._push_undo()
        self.state.ocr_texts.append(entry)
        self._save_ocr_tsv()
        add_correction(self.state, type_corrections, description, dict(entry))
        self.ocr_positions = {
            str(entry.get("side", "TOP")).upper(): [
                float(entry.get("x", 0.0)),
                float(entry.get("y", 0.0)),
                float(entry.get("w", 0.0)),
                float(entry.get("h", 0.0)),
            ]
        }
        self.show_ocr.set(True)
        self._save_corrections_and_redraw()

    def _save_ocr_tsv(self) -> None:
        """Write the current OCR entries back to the TSV file consumed by later reconstruction steps."""
        save_ocr_texts(self.state.work_folder / "ocr_text.tsv", self.state.ocr_texts)

    def _start_manual_ocr_region(self) -> None:
        """Enter rectangle-selection mode for a manually typed OCR label."""
        self.mode.set("ocr_manual_region")
        self.ocr_region_start = None
        self.ocr_region_preview.clear()
        self.status.set("Manual OCR: select the text rectangle on TOP or BOTTOM.")
        self._redraw_all()

    def _click_ocr_region(self, side: str, x: float, y: float, manual: bool = False) -> None:
        """Collect the two rectangle corners used for automatic or manual OCR region selection."""
        if self.ocr_region_busy:
            self.status.set("OCR: previous area is still being processed.")
            return
        if not self.ocr_region_start or self.ocr_region_start[0] != side or self.ocr_region_start[3] != manual:
            self.ocr_region_start = (side, x, y, manual)
            self._set_ocr_region_preview(side, x, y, x, y)
            if manual:
                self.status.set(f"Manual OCR {side}: click the opposite rectangle corner.")
            else:
                self.status.set(f"OCR {side}: click the opposite text corner.")
            return
        self._finish_ocr_region_selection(side, x, y)

    def _finish_ocr_region_selection(self, side: str, x: float, y: float) -> None:
        """Validate the selected OCR rectangle and dispatch it to either manual entry or automatic recognition."""
        if not self.ocr_region_start:
            return
        _side, x0, y0, manual = self.ocr_region_start
        self.ocr_region_start = None
        self.ocr_region_preview.clear()
        self._redraw_side(side)
        x1, x2 = sorted((x0, x))
        y1, y2 = sorted((y0, y))
        if x2 - x1 < 4 or y2 - y1 < 4:
            self.status.set("OCR: selected area is too small.")
            return
        if manual:
            self._add_ocr_manually(side, x1, y1, x2 - x1, y2 - y1)
        else:
            self._read_ocr_region(side, x1, y1, x2 - x1, y2 - y1)

    def _read_ocr_region(self, side: str, x: float, y: float, w: float, h: float) -> None:
        """Crop the selected image region and run OCR in a worker thread so the Tk window remains responsive."""
        if self.ocr_region_busy:
            self.status.set("OCR: previous area is still being processed.")
            return
        image = self.images[side].convert("RGB")
        width, height = image.size
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(width, int(round(x + w)))
        y1 = min(height, int(round(y + h)))
        if x1 <= x0 or y1 <= y0:
            return
        crop = image.crop((x0, y0, x1, y1))
        crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        tesseract_path = self.state.parameters.get("tesseract_path") or None
        worker_logs: list[str] = []
        self.ocr_region_busy = True
        self.status.set(f"OCR {side}: processing selected area...")
        self.configure(cursor="watch")

        def log(text: str) -> None:
            """Collect OCR worker messages for status reporting when no text is found."""
            worker_logs.append(text)

        def worker() -> None:
            """Run OCR on the cropped image and marshal the result back to the Tk event loop."""
            try:
                results = read_texts(
                    crop_bgr,
                    self.state.work_folder,
                    f"{side}_manual",
                    tesseract_path,
                    pass_count=6,
                    log=log,
                )
            except Exception as error:
                self.after(0, lambda: self._finish_ocr_region_with_error(str(error)))
                return
            self.after(0, lambda: self._finish_ocr_region(side, x0, y0, x1 - x0, y1 - y0, results, worker_logs))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_ocr_region_with_error(self, error: str) -> None:
        """Reset busy UI state and show the OCR error produced by the worker thread."""
        self.ocr_region_busy = False
        self.configure(cursor="")
        self.status.set(f"OCR: area read error: {error}")
        messagebox.showerror("OCR", f"Could not read the OCR area:\n{error}", parent=self)

    def _finish_ocr_region(
        self,
        side: str,
        x0: int,
        y0: int,
        w: int,
        h: int,
        results: list[dict[str, Any]],
        worker_logs: list[str],
    ) -> None:
        """Normalize OCR worker results, shift boxes into board coordinates, and ask which recognized text to keep."""
        self.ocr_region_busy = False
        self.configure(cursor="")
        if not results:
            last_log = worker_logs[-1] if worker_logs else "no result"
            self.status.set(f"OCR {side}: no text found in area ({last_log}).")
            if messagebox.askyesno("OCR", "Could not read text automatically. Enter it manually?", parent=self):
                self._add_ocr_manually(side, x0, y0, w, h)
            return
        normalized_results = []
        for entry in results:
            normalized = _normalize_ocr_entry(entry, side)
            normalized["x"] = int(normalized.get("x", 0)) + x0
            normalized["y"] = int(normalized.get("y", 0)) + y0
            normalized["variant"] = f"manual_region:{entry.get('variant', '')}"
            normalized_results.append(normalized)
        best = max(normalized_results, key=_ocr_confidence)
        texts = ", ".join(_ocr_text(entry) for entry in normalized_results)
        self.status.set(f"OCR {side}: read {len(results)} text items.")
        if not messagebox.askyesno("OCR", f"Read: {texts}\nAdd the best text '{_ocr_text(best)}'?", parent=self):
            return
        self._add_entry_ocr(best, "region_ocr", f"Read OCR from manually selected area: {_ocr_text(best)}.")
