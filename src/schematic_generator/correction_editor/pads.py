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


class CorrectionEditorPadsMixin:
    def _add_pad(self, side: str, x: float, y: float) -> None:
        """Adds a manual pad at the clicked board coordinate and selects it for immediate inspection."""
        self._push_undo()
        radius = self._typical_radius(side)
        pad = Pad(self._next_id(side), side, x, y, radius, 1.0, type=self.type_pad.get(), status="manual")
        self._pads_side(side).append(pad)
        self.selected_node = pad.node
        add_correction(
            self.state,
            "add_pad",
            f"Added pad {pad.node}.",
            {"pad": pad.node, "x": round(x, 2), "y": round(y, 2), "radius": round(radius, 2)},
        )
        self._save_corrections_and_redraw()

    def _name_pad(self, side: str | None = None, x: float | None = None, y: float | None = None) -> None:
        """Open the pad naming dialog for the clicked pad or currently selected pad."""
        pad: Pad | None = None
        if side is not None and x is not None and y is not None:
            pad = self._nearest_pad(self._pads_side(side), x, y, side)
        if not pad and self.selected_node:
            pad = self._pads_by_node().get(self.selected_node)
        if not pad:
            self.status.set("Click or select the pad you want to name.")
            return

        window = tk.Toplevel(self)
        window.title("Pad name")
        window.resizable(False, False)
        name = tk.StringVar(value=pad.name or _pad_identifier(pad))
        ttk.Label(window, text="Pad").grid(row=0, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: (8, 2)})
        ttk.Label(window, text=self._pad_description(pad)).grid(row=0, column=1, sticky="w", padx=8, **{_PAD_Y_KEY: (8, 2)})
        ttk.Label(window, text="Name").grid(row=1, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: 2})
        area = ttk.Entry(window, textvariable=name, width=30)
        area.grid(row=1, column=1, padx=8, **{_PAD_Y_KEY: 2})
        area.focus_set()

        def save() -> None:
            """Validate the modal form, apply the requested edit to correction state, and close the dialog after a successful save."""
            new_name = name.get().strip()
            self._set_name_pad(pad, new_name)
            window.destroy()

        buttons = ttk.Frame(window)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=8, **{_PAD_Y_KEY: 8})
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT)
        window.bind("<Return>", lambda _event: save())
        window.bind("<Escape>", lambda _event: window.destroy())

    def _set_name_pad(self, pad: Pad, name: str) -> None:
        """Apply a pad name to the selected pad and its paired counterpart when present."""
        pads = self._expand_with_pads_paired([pad])
        self._push_undo()
        for candidate in pads:
            candidate.name = name
            candidate.status = "manual"
        add_correction(
            self.state,
            "rename_pad",
            f"Set name '{name or '(empty)'}' for {len(pads)} pads.",
            {"name": name, "pads": [p.node for p in pads]},
        )
        self.selected_node = pad.node
        self._save_corrections_and_redraw()

    def _remove_pad(self, side: str, x: float, y: float) -> None:
        """Delete the clicked pad, optionally propagate the deletion to similar pads, and remove invalidated pairs."""
        pad = self._nearest_pad(self._pads_side(side), x, y, side)
        if not pad:
            self.status.set("No pad found at the click location.")
            return
        self._remove_pads([pad], allow_similar=True)

    def _remove_selected_pads(self) -> bool:
        """Delete currently selected pads and return whether anything was removed."""
        selected = set(self.selected_nodes)
        if self.selected_node:
            selected.add(self.selected_node)
        pads_by_node = self._pads_by_node()
        pads = [pads_by_node[node] for node in sorted(selected) if node in pads_by_node]
        if not pads:
            return False
        self._remove_pads(pads, allow_similar=len(pads) == 1)
        return True

    def _remove_pads(self, pads: list[Pad], allow_similar: bool) -> None:
        """Remove pads, optional similar-pad propagation, and pairs invalidated by the deletion."""
        if not pads:
            return
        primary_pad = pads[0]
        removed = list(pads)
        applied_similar_pads = False
        if allow_similar and len(pads) == 1:
            similar_pads = find_similar_pads(primary_pad, self._pads_side(primary_pad.side))
        else:
            similar_pads = []
        if similar_pads:
            selected = self._select_propagation_candidates(
                "Deletion propagation",
                f"Selected pad: {self._pad_description(primary_pad)}. Select similar pads to delete with it.",
                similar_pads,
            )
            applied_similar_pads = bool(selected)
            removed.extend(selected)
        self._push_undo()
        removed_nodes = {p.node for p in removed}
        for side_remove in ("TOP", "BOTTOM"):
            self._set_pads_side(side_remove, [p for p in self._pads_side(side_remove) if p.node not in removed_nodes])
        previous_pair_count = len(self.state.pairs)
        self.state.pairs = [
            pair for pair in self.state.pairs
            if pair.pad_top not in removed_nodes and pair.pad_bottom not in removed_nodes
        ]
        self.selected_node = None
        self.selected_nodes.difference_update(removed_nodes)
        self.selected_nodes_order = [node for node in self.selected_nodes_order if node not in removed_nodes]
        add_correction(
            self.state,
            "delete_pad",
            f"Removed {len(removed)} pads.",
            {
                "removed": sorted(removed_nodes),
                "removed_pairs": previous_pair_count - len(self.state.pairs),
                "applied_similar": applied_similar_pads,
            },
            suggestions=[{"pad": p.node, "name": _label_pad(p), "radius": round(p.radius, 2), "confidence": round(p.confidence, 3)} for p in similar_pads],
        )
        self._save_corrections_and_redraw()

    def _set_type_pad(self) -> None:
        """Apply the selected pad type to chosen pads and their paired counterparts, with optional propagation."""
        selected = set(self.selected_nodes)
        if self.selected_node:
            selected.add(self.selected_node)
        pads_by_node = self._pads_by_node()
        pads = [pads_by_node[w] for w in sorted(selected) if w in pads_by_node]
        if not pads:
            messagebox.showinfo("No pads", "Select one or more pads whose type you want to set.", parent=self)
            return
        new_type = self.type_pad.get()
        similar_pads: list[Pad] = []
        if len(pads) == 1:
            similar_pads = find_similar_pads(pads[0], self._pads_side(pads[0].side), limit=50)
            if similar_pads:
                pads.extend(self._select_propagation_candidates(
                    "Pad type propagation",
                    f"Selected pad: {self._pad_description(pads[0])}. Select similar pads that should also be set to type {new_type}.",
                    similar_pads,
                ))
        pads = self._expand_with_pads_paired(pads)
        self._push_undo()
        for pad in pads:
            pad.type = new_type
            pad.status = "manual"
        add_correction(
            self.state,
            "set_pad_type",
            f"Set type {new_type} for {len(pads)} pads.",
            {"type": new_type, "pads": [pad.node for pad in pads]},
            suggestions=[{"pad": pad.node, "name": _label_pad(pad), "radius": round(pad.radius, 2), "confidence": round(_pad_confidence(pad), 3)} for pad in similar_pads],
        )
        self._save_corrections_and_redraw()

    def _toggle_pad_element(self, side: str, x: float, y: float) -> None:
        """Toggle a pad in the manual component selection set while preserving click order."""
        pad = self._nearest_pad(self._pads_side(side), x, y, side)
        if not pad:
            return
        if pad.node in self.selected_nodes:
            self.selected_nodes.remove(pad.node)
            self.selected_nodes_order = [w for w in self.selected_nodes_order if w != pad.node]
        else:
            self.selected_nodes.add(pad.node)
            self.selected_nodes_order = [w for w in self.selected_nodes_order if w != pad.node]
            self.selected_nodes_order.append(pad.node)
        self.selected_node = pad.node
        self.type_pad.set(pad.type or "pad")
        self._refresh_status()
        self._redraw_all()
