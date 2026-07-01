from __future__ import annotations

import copy
import math
import threading
import tkinter as tk
from collections.abc import Callable
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


class CorrectionEditorComponentsMixin:
    def _edit_manual_element(self) -> None:
        """Open the editor for the currently selected manual component."""
        index = self._selected_element_index()
        if index is None:
            self.status.set("Select a manual component to edit.")
            return
        self._open_element_window(index)

    def _remove_element_manual(self) -> None:
        """Delete the selected manual component and persist the correction record."""
        index = self._selected_element_index()
        if index is None:
            self.status.set("Select a manual component to delete.")
            return
        component = self.state.manual_components[index]
        if not messagebox.askyesno("Delete component", f"Delete component {component.get('ref', '?')}?", parent=self):
            return
        self._push_undo()
        removed = self.state.manual_components.pop(index)
        add_correction(self.state, "delete_manual_component", f"Removed component {removed.get('ref', '?')}.", removed)
        self._save_corrections_and_redraw()

    def _open_element_window(self, index: int) -> None:
        """Show the component form used for editing reference, type, value, footprint, and attached pad nodes."""
        component = self.state.manual_components[index]
        window = tk.Toplevel(self)
        window.title("Edit component")
        window.resizable(False, False)
        ref = tk.StringVar(value=str(component.get("ref", "")))
        type = tk.StringVar(value=str(_component_type(component)))
        value = tk.StringVar(value=str(_component_value(component)))
        footprint = tk.StringVar(value=str(component.get("footprint", "")))
        pads = tk.StringVar(value=", ".join(str(p) for p in _component_pads(component)))

        fields = (
            ("Ref", ref),
            ("Type", type),
            ("Value", value),
            ("Footprint", footprint),
            ("Pads", pads),
        )
        first_entry: tk.Widget | None = None
        type_box: ttk.Combobox | None = None
        for row, (label, variable) in enumerate(fields):
            ttk.Label(window, text=label).grid(row=row, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: (8 if row == 0 else 2, 2)})
            if label == "Type":
                type_box = ttk.Combobox(
                    window,
                    textvariable=type,
                    values=_TYPES_ELEMENTS,
                    state="readonly",
                    width=28,
                )
                type_box.grid(row=row, column=1, padx=8, **{_PAD_Y_KEY: (8 if row == 0 else 2, 2)})
            else:
                area = ttk.Entry(window, textvariable=variable, width=31)
                area.grid(row=row, column=1, padx=8, **{_PAD_Y_KEY: (8 if row == 0 else 2, 2)})
                if first_entry is None:
                    first_entry = area

        def pad_count() -> int:
            """Compute the current pad count used to infer default component values and footprints."""
            return len([w for w in pads.get().replace(";", ",").split(",") if w.strip()])

        def change_type(_event=None) -> None:
            """Update dependent default fields when the component type changes."""
            if not value.get().strip():
                value.set(_default_value(type.get(), pad_count()))
            if not footprint.get().strip():
                footprint.set(_default_footprint(type.get(), pad_count()))

        if type_box is not None:
            type_box.bind("<<ComboboxSelected>>", change_type)
            type_box.bind("<KeyPress>", lambda event: self._select_type_from_key(event, type, pad_count, change_type))

        def save() -> None:
            """Validate the modal form, apply the requested edit to correction state, and close the dialog after a successful save."""
            pads_nodes = [w.strip() for w in pads.get().replace(";", ",").split(",") if w.strip()]
            known = self._pads_by_node()
            unknown = [w for w in pads_nodes if w not in known]
            if unknown:
                messagebox.showerror("Unknown pads", "Unknown pads: " + ", ".join(unknown[:8]), parent=window)
                return
            self._push_undo()
            component.update({
                "ref": ref.get().strip() or self._default_ref(),
                "type": type.get(),
                "value": value.get().strip() or _default_value(type.get(), len(pads_nodes)),
                "footprint": footprint.get().strip(),
                "pads": pads_nodes,
            })
            add_correction(self.state, "edit_manual_component", f"Changed component {component['ref']}.", dict(component))
            window.destroy()
            self._save_corrections_and_redraw()

        buttons = ttk.Frame(window)
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", padx=8, **{_PAD_Y_KEY: 8})
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT)
        window.bind("<Return>", lambda _event: save())
        self._set_modal_window(window, first_entry)

    def _nearest_unused_ocr(self, pads: list[Pad]) -> tuple[int, dict[str, Any], float] | None:
        """Find the closest OCR label that is not already attached to a manual component."""
        if not pads or not self.state.ocr_texts:
            return None
        used_ocr_indices = {
            int(component.get("ocr", {}).get("index", -1))
            for component in self.state.manual_components
            if isinstance(component.get("ocr"), dict)
        }
        cx = sum(p.x for p in pads) / len(pads)
        cy = sum(p.y for p in pads) / len(pads)
        side = {p.side for p in pads}
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for index, entry in enumerate(self.state.ocr_texts):
            if index in used_ocr_indices:
                continue
            if str(entry.get("side", "")).upper().replace("_MANUAL", "") not in side:
                continue
            tx = float(entry.get("x", 0.0)) + float(entry.get("w", 0.0)) / 2.0
            ty = float(entry.get("y", 0.0)) + float(entry.get("h", 0.0)) / 2.0
            candidates.append((math.hypot(tx - cx, ty - cy), index, dict(entry)))
        if not candidates:
            return None
        distance, index, entry = min(candidates, key=lambda item: item[0])
        limit = max(90.0, max(p.radius for p in pads) * 14.0)
        return (index, entry, distance) if distance <= limit else None

    def _type_from_ocr_text(self, text: str, pad_count: int) -> str:
        """Infer a component type from nearby OCR text, falling back to a generic device."""
        return _type_from_prefix(text, pad_count) or "Device"

    def _select_type_from_key(
        self,
        event: tk.Event,
        type: tk.StringVar,
        count_pads_fn: Callable[[], int],
        after_change: Callable[[], None],
    ) -> str | None:
        """Use a one-key prefix shortcut to choose a component type in component dialogs."""
        if event.state & 0x0004:
            return None
        new_type = _type_from_prefix(str(getattr(event, "char", "")), count_pads_fn())
        if not new_type:
            return None
        type.set(new_type)
        after_change()
        return "break"

    def _create_element_from_selection(self) -> None:
        """Open the component creation dialog for currently selected pads and prefill it from nearby OCR text."""
        if not self.selected_nodes:
            messagebox.showinfo("No pads", "First choose the 'Select component pads' mode and click the component pads.", parent=self)
            return
        pads_start = self._sorted_selected_pads()
        suggested_ocr = self._nearest_unused_ocr(pads_start)
        ocr_text = str(suggested_ocr[1].get("text", "")) if suggested_ocr else ""
        window = tk.Toplevel(self)
        window.title("Create component")
        window.resizable(False, False)
        ref = tk.StringVar(value=ocr_text or self._default_ref())
        type = tk.StringVar(value=self._type_from_ocr_text(ocr_text, len(pads_start)) if ocr_text else ("Resistor" if len(self.selected_nodes) == 2 else "Device"))
        value = tk.StringVar(value=_default_value(type.get(), len(pads_start)) if ocr_text else "")
        footprint = tk.StringVar(value=_default_footprint(type.get(), len(pads_start)))

        ttk.Label(window, text="Ref").grid(row=0, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: (8, 2)})
        ref_entry = ttk.Entry(window, textvariable=ref, width=28)
        ref_entry.grid(row=0, column=1, padx=8, **{_PAD_Y_KEY: (8, 2)})
        ttk.Label(window, text="Type").grid(row=1, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: 2})
        type_box = ttk.Combobox(
            window,
            textvariable=type,
            values=_TYPES_ELEMENTS,
            state="readonly",
            width=25,
        )
        type_box.grid(row=1, column=1, padx=8, **{_PAD_Y_KEY: 2})
        ttk.Label(window, text="Value").grid(row=2, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: 2})
        ttk.Entry(window, textvariable=value, width=28).grid(row=2, column=1, padx=8, **{_PAD_Y_KEY: 2})
        ttk.Label(window, text="Footprint").grid(row=3, column=0, sticky="w", padx=8, **{_PAD_Y_KEY: 2})
        ttk.Entry(window, textvariable=footprint, width=28).grid(row=3, column=1, padx=8, **{_PAD_Y_KEY: 2})
        ocr_info = tk.StringVar(
            value=(
                f"OCR: {ocr_text} ({suggested_ocr[2]:.1f}px)"
                if suggested_ocr
                else "OCR: no unused nearby text"
            )
        )
        ttk.Label(window, textvariable=ocr_info, foreground="#555").grid(row=4, column=0, columnspan=2, sticky="w", padx=8, **{_PAD_Y_KEY: (4, 2)})
        type_manually_changed = {"value": False}

        def pad_count() -> int:
            """Compute the current pad count used to infer default component values and footprints."""
            return len(self.selected_nodes) or len(pads_start)

        def change_type(_event=None, manually: bool = False) -> None:
            """Update dependent default fields when the component type changes."""
            if manually:
                type_manually_changed["value"] = True
            if not value.get():
                value.set(_default_value(type.get(), pad_count()))
            footprint.set(_default_footprint(type.get(), pad_count()))

        def align_type_from_ref(_event=None) -> None:
            """Infer component type from the edited reference designator unless the user already chose a type manually."""
            if type_manually_changed["value"]:
                return
            new_type = _type_from_prefix(ref.get(), pad_count())
            if new_type:
                type.set(new_type)
                change_type()

        def select_type_from_key(event: tk.Event) -> str | None:
            """Apply keyboard type shortcuts inside the component creation dialog."""
            result = self._select_type_from_key(event, type, pad_count, change_type)
            if result == "break":
                type_manually_changed["value"] = True
            return result

        type_box.bind("<<ComboboxSelected>>", lambda event: change_type(event, True))
        type_box.bind("<KeyPress>", select_type_from_key)
        ref_entry.bind("<KeyRelease>", align_type_from_ref)
        change_type()

        def save() -> None:
            """Validate the modal form, apply the requested edit to correction state, and close the dialog after a successful save."""
            pads = self._sorted_selected_pads()
            self._push_undo()
            component = add_manual_element(
                self.state,
                ref.get().strip() or self._default_ref(),
                type.get(),
                value.get().strip() or _default_value(type.get(), len(pads)),
                footprint.get().strip(),
                [pad.node for pad in pads],
            )
            if suggested_ocr:
                index_ocr, entry_ocr, distance = suggested_ocr
                component["ocr"] = {
                    "index": index_ocr,
                    "text": str(entry_ocr.get("text", "")),
                    "side": str(entry_ocr.get("side", "")),
                    "x": float(entry_ocr.get("x", 0.0)),
                    "y": float(entry_ocr.get("y", 0.0)),
                    "w": float(entry_ocr.get("w", 0.0)),
                    "h": float(entry_ocr.get("h", 0.0)),
                    "distance": round(distance, 2),
                    "target_field": "ref",
                }
            add_correction(
                self.state,
                "manual_component",
                f"Added component {component['ref']} with {len(pads)} pads.",
                component,
            )
            self.selected_nodes.clear()
            window.destroy()
            self._save_corrections_and_redraw()

        buttons = ttk.Frame(window)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", padx=8, **{_PAD_Y_KEY: 8})
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT)
        window.bind("<Return>", lambda _event: save())
        self._set_modal_window(window, ref_entry)
