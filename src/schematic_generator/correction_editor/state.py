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
from schematic_generator.diagnostics_facade import load_problems
from schematic_generator.models import Pad
from schematic_generator.netlist_facade import build_nets
from schematic_generator.ocr import read_texts, save_texts_ocr as save_ocr_texts
from schematic_generator.preview import draw_connections_colored


class CorrectionEditorStateMixin:
    def _snapshot(self) -> dict[str, Any]:
        """Capture all mutable editor state needed to undo pad edits, OCR edits, mask edits, and component changes."""
        return {
            "top_pads": copy.deepcopy(self.state.top_pads),
            "bottom_pads": copy.deepcopy(self.state.bottom_pads),
            "pairs": copy.deepcopy(self.state.pairs),
            "corrections": copy.deepcopy(self.state.corrections),
            "manual_components": copy.deepcopy(self.state.manual_components),
            "ocr_texts": copy.deepcopy(self.state.ocr_texts),
            "masks": {side: self.mask[side].copy() for side in ("TOP", "BOTTOM")},
            "selected_node": self.selected_node,
            "selected_nodes": set(self.selected_nodes),
            "selected_nodes_order": list(self.selected_nodes_order),
            "selected_trace": self.selected_trace,
            "ocr_positions": copy.deepcopy(self.ocr_positions),
        }

    def _push_undo(self) -> None:
        """Store the current snapshot on the undo stack and clear redo history before a new user operation mutates state."""
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore a saved editor snapshot, rewrite dependent files, and redraw the board from that state."""
        self.state.top_pads = copy.deepcopy(snapshot["top_pads"])
        self.state.bottom_pads = copy.deepcopy(snapshot["bottom_pads"])
        self.state.pairs = copy.deepcopy(snapshot["pairs"])
        self.state.corrections = copy.deepcopy(snapshot["corrections"])
        self.state.manual_components = copy.deepcopy(snapshot["manual_components"])
        self.state.ocr_texts = copy.deepcopy(snapshot.get("ocr_texts", self.state.ocr_texts))
        self.mask = {side: snapshot["masks"][side].copy() for side in ("TOP", "BOTTOM")}
        self.selected_node = snapshot.get("selected_node")
        self.selected_nodes = set(snapshot.get("selected_nodes", set()))
        self.selected_nodes_order = list(snapshot.get("selected_nodes_order", []))
        self.selected_trace = snapshot.get("selected_trace")
        self.ocr_positions = copy.deepcopy(snapshot.get("ocr_positions", {}))
        self.problem_nodes.clear()
        self.problem_positions.clear()
        self.ocr_region_start = None
        self.ocr_region_preview.clear()
        for side in ("TOP", "BOTTOM"):
            self._save_mask_side(side)
        self._save_ocr_tsv()
        self._save_corrections_and_redraw()

    def _undo(self) -> None:
        """Restore the previous snapshot while preserving the current state on the redo stack."""
        if not self.undo_stack:
            self.status.set("There is no operation to undo.")
            return
        self.redo_stack.append(self._snapshot())
        self._restore_snapshot(self.undo_stack.pop())

    def _redo(self) -> None:
        """Restore the next redo snapshot while preserving the current state on the undo stack."""
        if not self.redo_stack:
            self.status.set("There is no operation to redo.")
            return
        self.undo_stack.append(self._snapshot())
        self._restore_snapshot(self.redo_stack.pop())

    def _save_corrections_and_redraw(self) -> None:
        """Persist correction state without recalculation, refresh side panels, and redraw overlays."""
        try:
            save_state_corrections_without_recalculation(self.state)
        except Exception as error:
            messagebox.showerror("Correction save error", str(error), parent=self)
            self.status.set(f"Correction save error: {error}")
            return
        self.needs_recalculation = True
        self._refresh_manual_elements()
        self._refresh_ocr()
        self._refresh_status(extra="corrections saved")
        self._redraw_all()

    def _recalculate_save_redraw(self) -> None:
        """Run reconstruction from the accumulated corrections, reload derived state, and refresh the editor."""
        try:
            result = recalculate_and_save_by_corrections(self.state, self._set_status)
        except Exception as error:
            messagebox.showerror("Correction error", str(error), parent=self)
            self.status.set(f"Correction error: {error}")
            return
        self.state.pending_recalculation = False
        self.needs_recalculation = False
        self.problems = load_problems(self.state.work_folder)
        self._refresh_problems()
        self._refresh_manual_elements()
        self._refresh_ocr()
        self._refresh_status(extra=f"nets={len(result.nets)}, corrections={len(self.state.corrections)}")
        if self.on_save:
            self.on_save()
        self._redraw_all()

    def _clear_selection(self) -> None:
        """Clear pad, problem, trace, and OCR transient selections without deleting saved corrections."""
        self.selected_nodes.clear()
        self.selected_nodes_order.clear()
        self.selected_node = None
        self.selected_trace = None
        self.problem_nodes.clear()
        self.problem_positions.clear()
        self.pending_pair_node = None
        self.trace_start = None
        self.ocr_region_start = None
        self.ocr_region_preview.clear()
        self._refresh_status()
        self._redraw_all()

    def _cancel_current_operation(self, _event: tk.Event | None = None) -> str:
        """Cancel the active transient operation or clear selection when Escape is pressed."""
        mode = self.mode.get()
        if mode in {"ocr_region", "ocr_manual_region"} or self.ocr_region_start or self.ocr_region_preview:
            if self.ocr_region_start or self.ocr_region_preview:
                self.ocr_region_start = None
                self.ocr_region_preview.clear()
                self.status.set("Reverted the last OCR selection point.")
            elif not self.ocr_region_busy:
                self.mode.set("move")
                self.status.set("Returned to select/move mode.")
            self._redraw_all()
            return "break"
        if self.trace_start:
            self.trace_start = None
            self.status.set("Reverted the first trace tool point.")
            self._redraw_all()
            return "break"
        if self.pending_pair_node:
            self.pending_pair_node = None
            self.status.set("Reverted the first pair pad.")
            self._redraw_all()
            return "break"
        if mode == "component" and self.selected_nodes_order:
            last_node = self.selected_nodes_order.pop()
            self.selected_nodes.discard(last_node)
            self.selected_node = self.selected_nodes_order[-1] if self.selected_nodes_order else None
            self.status.set(f"Reverted pad selection {_short_node(last_node)}.")
            self._redraw_all()
            return "break"
        if mode != "move":
            self.mode.set("move")
            self.status.set("Returned to select/move mode.")
            return "break"
        return "break"

    def _delete_selected(self, event: tk.Event | None = None) -> str | None:
        """Delete the currently selected pad, trace, or manual component."""
        widget = getattr(event, "widget", None) if event is not None else None
        if isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox)):
            return None
        if hasattr(self, "tree_elements") and widget is self.tree_elements:
            self._remove_element_manual()
            return "break"
        if hasattr(self, "tree_ocr") and widget is self.tree_ocr:
            self._remove_ocr()
            return "break"
        if self.selected_trace and self._delete_selected_trace():
            return "break"
        if self._remove_selected_pads():
            return "break"
        if self._selected_element_index() is not None:
            self._remove_element_manual()
            return "break"
        self.status.set("Select a pad, trace, or component before pressing Delete.")
        return "break"

    def _close_editor(self) -> None:
        """Ask how to handle pending corrections before closing the editor."""
        if self.needs_recalculation:
            recalculate = messagebox.askyesno(
                "Recalculate schematics",
                "Corrections are saved but the generated files have not been recalculated.\n\n"
                "Recalculate corrected schematics now?",
                parent=self,
            )
            if recalculate:
                self._recalculate_save_redraw()
                if self.needs_recalculation:
                    return
        self.destroy()

    def _refresh_status(self, extra: str = "") -> None:
        """Update the status bar with counts, selection context, recalculation state, and optional operation details."""
        paired = self._paired_nodes()
        pad_count = len(self.state.top_pads) + len(self.state.bottom_pads)
        unpaired_count = pad_count - len(paired)
        selected = ""
        if self.selected_node:
            pad = self._pads_by_node().get(self.selected_node)
            selected = f", selected={self._pad_description(pad) if pad else self.selected_node}"
        if self.selected_trace:
            selected = f", selected=trace {self.selected_trace[0]}"
        component_selection = f", component pads={len(self.selected_nodes)}" if self.selected_nodes else ""
        recalculation_note = ", requires recalculation" if self.needs_recalculation else ""
        suffix = f", {extra}" if extra else ""
        self.status.set(
            f"TOP={len(self.state.top_pads)}, BOTTOM={len(self.state.bottom_pads)}, "
            f"pairs={len(self.state.pairs)}, unpaired={unpaired_count}{selected}{component_selection}{recalculation_note}{suffix}"
        )

    def _set_status(self, text: str) -> None:
        """Write a message into the editor status variable from worker callbacks."""
        self.status.set(text)
