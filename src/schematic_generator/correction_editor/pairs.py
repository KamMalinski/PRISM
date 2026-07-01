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
from schematic_generator.models import Pad, HolePair
from schematic_generator.netlist_facade import build_nets
from schematic_generator.ocr import read_texts, save_texts_ocr as save_ocr_texts
from schematic_generator.preview import draw_connections_colored


class CorrectionEditorPairsMixin:
    def _pairing_click(self, side: str, x: float, y: float) -> None:
        """Collect the two pads needed to create a TOP/BOTTOM hole pair."""
        pad = self._nearest_pad(self._pads_side(side), x, y, side)
        if not pad:
            self.status.set("Click the pad to pair.")
            return
        if not self.pending_pair_node:
            self.pending_pair_node = pad.node
            self.selected_node = pad.node
            self.status.set(f"{pad.node} selected. Click the matching pad on the other side.")
            self._redraw_all()
            return

        first = self._pads_by_node().get(self.pending_pair_node)
        self.pending_pair_node = None
        if not first:
            return
        if first.side == pad.side:
            self.pending_pair_node = pad.node
            self.selected_node = pad.node
            self.status.set("A pad on the same side was selected. Now click a pad on the other side.")
            self._redraw_all()
            return
        self._connect_two_pads(first, pad)

    def _connect_two_pads(self, first: Pad, second: Pad) -> None:
        """Connects or replaces a TOP/BOTTOM pair, warns about distant candidates, and shares a common pad name."""
        top = first if first.side == "TOP" else second
        bottom = second if first.side == "TOP" else first
        distance = math.hypot(top.x - bottom.x, top.y - bottom.y)
        tolerance = max(12.0, 3.0 * max(top.radius, bottom.radius))
        if distance > tolerance:
            ok = messagebox.askyesno(
                "Distant pair",
                f"Pads are {distance:.1f}px apart. Connect anyway?",
                parent=self,
            )
            if not ok:
                return
        self._push_undo()
        self.state.pairs = [
            pair for pair in self.state.pairs
            if pair.pad_top != top.node and pair.pad_bottom != bottom.node
        ]
        common_name = top.name or bottom.name
        if common_name:
            top.name = common_name
            bottom.name = common_name
        confidence = max(0.05, 1.0 - distance / max(tolerance, 1.0))
        self.state.pairs.append(HolePair(top.node, bottom.node, distance, confidence))
        self.selected_node = top.node
        add_correction(
            self.state,
            "connect_pair",
            f"Connected pair {top.node} <-> {bottom.node}.",
            {"pad_top": top.node, "pad_bottom": bottom.node, "distance": round(distance, 2), "confidence": round(confidence, 3)},
        )
        self._save_corrections_and_redraw()

    def _disconnect_pair(self, side: str, x: float, y: float) -> None:
        """Deletes any TOP/BOTTOM pair that contains the clicked pad."""
        pad = self._nearest_pad(self._pads_side(side), x, y, side)
        if not pad:
            self.status.set("No pad found to disconnect.")
            return
        removed = [
            pair for pair in self.state.pairs
            if pair.pad_top == pad.node or pair.pad_bottom == pad.node
        ]
        if not removed:
            self.status.set(f"Pad {pad.node} has no TOP/BOTTOM pair.")
            return
        self._push_undo()
        self.state.pairs = [
            pair for pair in self.state.pairs
            if pair.pad_top != pad.node and pair.pad_bottom != pad.node
        ]
        add_correction(
            self.state,
            "disconnect_pair",
            f"Disconnected {len(removed)} pairs for {pad.node}.",
            {"pad": pad.node, "pairs": [{"pad_top": p.pad_top, "pad_bottom": p.pad_bottom} for p in removed]},
        )
        self._save_corrections_and_redraw()

    def _select_propagation_candidates(self, title: str, description: str, candidates: list[Pad]) -> list[Pad]:
        """Show a checklist of similar pads so a change can be applied to repeated geometry deliberately."""
        if not candidates or self.suppress_propagation_suggestions:
            return []
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("560x420")
        window.transient(self)
        window.grab_set()
        result: list[Pad] = []
        do_not_show = tk.BooleanVar(value=False)

        ttk.Label(window, text=description, wraplength=520, justify="left").pack(anchor="w", padx=10, **{_PAD_Y_KEY: (10, 6)})
        container = ttk.Frame(window)
        container.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(container, highlightthickness=0)
        scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        list = ttk.Frame(canvas)
        list.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        variables: list[tuple[tk.BooleanVar, Pad]] = []
        for pad in candidates:
            variable = tk.BooleanVar(value=False)
            variables.append((variable, pad))
            text = f"{self._pad_description(pad)}  x={pad.x:.1f} y={pad.y:.1f} r={pad.radius:.1f} conf={_pad_confidence(pad):.2f} type={pad.type}"
            ttk.Checkbutton(list, text=text, variable=variable).pack(anchor="w", **{_PAD_Y_KEY: 1})

        buttons = ttk.Frame(window)
        buttons.pack(fill=tk.X, padx=10, **{_PAD_Y_KEY: 10})

        def select(value: bool) -> None:
            """Applies the requested checked state to every propagation candidate checkbox."""
            for variable, _pad in variables:
                variable.set(value)

        def apply_selection() -> None:
            """Collect checked propagation candidates and close the selection dialog."""
            if do_not_show.get():
                self.suppress_propagation_suggestions = True
            result.extend(pad for variable, pad in variables if variable.get())
            window.destroy()

        def close_without_apply() -> None:
            """Close the dialog without applying propagation to any similar pads."""
            if do_not_show.get():
                self.suppress_propagation_suggestions = True
            window.destroy()

        ttk.Checkbutton(
            buttons,
            text="Do not show suggestions in this session",
            variable=do_not_show,
        ).pack(side=tk.LEFT, padx=(0, 8))
        all_button = ttk.Button(buttons, text="All", command=lambda: select(True))
        none_button = ttk.Button(buttons, text="None", command=lambda: select(False))
        cancel_button = ttk.Button(buttons, text="Cancel", command=close_without_apply)
        apply_button = ttk.Button(buttons, text="Apply", command=apply_selection)
        all_button.pack(side=tk.LEFT)
        none_button.pack(side=tk.LEFT, padx=(4, 0))
        cancel_button.pack(side=tk.RIGHT, padx=(6, 0))
        apply_button.pack(side=tk.RIGHT)

        def close_from_enter(_event: tk.Event) -> str:
            """Close without applying even when a dialog button has focus."""
            close_without_apply()
            return "break"

        window.bind("<Return>", close_from_enter)
        window.bind("<Escape>", lambda _event: close_without_apply())
        for button in (all_button, none_button, cancel_button, apply_button):
            button.bind("<Return>", close_from_enter)

        def focus_dialog() -> None:
            window.lift()
            window.focus_force()
            cancel_button.focus_set()

        window.after_idle(focus_dialog)
        self.wait_window(window)
        return result
