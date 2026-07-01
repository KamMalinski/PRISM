from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image

from schematic_generator.correction_editor.support import *
from schematic_generator.corrections_facade import add_correction
from schematic_generator.diagnostics_facade import save_problems
from schematic_generator.ocr import save_texts_ocr as save_ocr_texts


class CorrectionEditorTablePanelMixin:
    def _build_panel_elements(self, panel: ttk.Notebook) -> None:
        """Assembles the manual components tab used to create, edit, and remove components from selected pads."""
        tab = ttk.Frame(panel, padding=6)
        panel.add(tab, text="Components")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.tree_elements = ttk.Treeview(
            tab,
            columns=("ref", "type", "value", "pads"),
            show="headings",
            height=16,
        )
        for column, title, width in (
            ("ref", "Ref", 55),
            ("type", "Type", 92),
            ("value", "Value", 78),
            ("pads", "Pads", 80),
        ):
            self.tree_elements.heading(column, text=title)
        self.tree_elements.column(column, width=width, stretch=column == "pads")
        self.tree_elements.grid(row=0, column=0, sticky="nsew")
        self.tree_elements.bind("<Double-1>", lambda _event: self._edit_manual_element())
        self.tree_elements.bind("<Delete>", lambda _event: (self._remove_element_manual(), "break")[-1])

        actions = ttk.Frame(tab)
        actions.grid(row=1, column=0, sticky="ew", **{_PAD_Y_KEY: (6, 0)})
        ttk.Button(actions, text="From selection", command=self._create_element_from_selection).pack(side=tk.LEFT)
        ttk.Button(actions, text="Edit", command=self._edit_manual_element).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Delete", command=self._remove_element_manual).pack(side=tk.LEFT, padx=(4, 0))

    def _build_panel_ocr(self, panel: ttk.Notebook) -> None:
        """Assembles the OCR tab for inspecting recognized labels, assigning them to components, adding manual text, and deleting bad reads."""
        tab = ttk.Frame(panel, padding=6)
        panel.add(tab, text="OCR")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.tree_ocr = ttk.Treeview(
            tab,
            columns=("side", "text", "x", "y"),
            show="headings",
            height=16,
        )
        for column, title, width in (
            ("side", "Side", 46),
            ("text", "Text", 145),
            ("x", "X", 54),
            ("y", "Y", 54),
        ):
            self.tree_ocr.heading(column, text=title)
            self.tree_ocr.column(column, width=width, stretch=column == "text")
        self.tree_ocr.grid(row=0, column=0, sticky="nsew")
        self.tree_ocr.bind("<<TreeviewSelect>>", lambda _event: self._update_selected_ocr())
        self.tree_ocr.bind("<Delete>", lambda _event: (self._remove_ocr(), "break")[-1])

        actions = ttk.Frame(tab)
        actions.grid(row=1, column=0, sticky="ew", **{_PAD_Y_KEY: (6, 0)})
        ttk.Button(actions, text="As ref", command=lambda: self._assign_ocr_to_element("ref")).pack(side=tk.LEFT)
        ttk.Button(actions, text="As value", command=lambda: self._assign_ocr_to_element("value")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Show", command=self._show_ocr).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Add from area", command=self._start_manual_ocr_region).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Delete", command=self._remove_ocr).pack(side=tk.LEFT, padx=(4, 0))

    def _load_images(self) -> dict[str, Image.Image]:
        """Opens normalized TOP and BOTTOM board images that serve as immutable backgrounds for every redraw."""
        return {
            "TOP": Image.open(self.state.work_folder / "top_normalized.png").convert("RGB"),
            "BOTTOM": Image.open(self.state.work_folder / "bottom_normalized.png").convert("RGB"),
        }

    def _load_mask(self) -> dict[str, Image.Image]:
        """Opens corrected trace masks when available, otherwise falls back to automatic masks or an empty mask for each side."""
        result: dict[str, Image.Image] = {}
        for side in ("TOP", "BOTTOM"):
            corrected = self.state.work_folder / f"{side.lower()}_mask_corrected.png"
            auto = self.state.work_folder / f"{side.lower()}_mask.png"
            path = corrected if corrected.exists() else auto
            result[side] = Image.open(path).convert("L") if path.exists() else Image.new("L", self.images[side].size, 0)
        return result

    def _refresh_manual_elements(self) -> None:
        """Rebuild the manual component table from the editable correction state."""
        if not hasattr(self, "tree_elements"):
            return
        for item in self.tree_elements.get_children():
            self.tree_elements.delete(item)
        for index, component in enumerate(self.state.manual_components):
            pads = _component_pads(component)
            self.tree_elements.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    str(component.get("ref", "")),
                    str(_component_type(component)),
                    str(_component_value(component)),
                    str(len(pads)),
                ),
            )

    def _selected_element_index(self) -> int | None:
        """Looks up the selected manual component index after validating that it still exists."""
        if not hasattr(self, "tree_elements"):
            return None
        choice = self.tree_elements.selection()
        if not choice:
            return None
        index = int(choice[0])
        return index if 0 <= index < len(self.state.manual_components) else None

    def _refresh_ocr(self) -> None:
        """Rebuild the OCR table from saved OCR entries."""
        if not hasattr(self, "tree_ocr"):
            return
        for item in self.tree_ocr.get_children():
            self.tree_ocr.delete(item)
        for index, entry in enumerate(self.state.ocr_texts):
            text = str(entry.get("text", ""))
            self.tree_ocr.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    str(entry.get("side", "")),
                    text if len(text) <= 32 else text[:29] + "...",
                    int(float(entry.get("x", 0))),
                    int(float(entry.get("y", 0))),
                ),
            )

    def _selected_ocr(self) -> tuple[int, dict[str, Any]] | None:
        """Looks up a copy of the selected OCR entry and its index after validating the current tree selection."""
        if not hasattr(self, "tree_ocr"):
            return None
        choice = self.tree_ocr.selection()
        if not choice:
            return None
        index = int(choice[0])
        if index < 0 or index >= len(self.state.ocr_texts):
            return None
        return index, dict(self.state.ocr_texts[index])

    def _update_selected_ocr(self) -> None:
        """Display the selected OCR box on the board canvases."""
        self._show_ocr()

    def _show_ocr(self) -> None:
        """Center editor attention on the selected OCR entry by enabling OCR overlay and recording its bounding box."""
        selected = self._selected_ocr()
        if not selected:
            return
        _index, entry = selected
        side = _ocr_side(entry, "TOP").upper()
        x = float(entry.get("x", 0.0))
        y = float(entry.get("y", 0.0))
        w = float(entry.get("w", 0.0))
        h = float(entry.get("h", 0.0))
        self.ocr_positions = {side: [x, y, w, h]}
        self.show_ocr.set(True)
        self.status.set(f"OCR {side}: {_ocr_text(entry)}")
        self._redraw_all()

    def _remove_ocr(self) -> None:
        """Delete the selected OCR entry, detach component references to it, and record the correction for persistence."""
        selected = self._selected_ocr()
        if not selected:
            self.status.set("Select OCR text to delete.")
            return
        index, entry = selected
        text = _ocr_text(entry)
        if not messagebox.askyesno("Delete OCR", f"Delete OCR label '{text}'?", parent=self):
            return
        self._push_undo()
        removed = self.state.ocr_texts.pop(index)
        for component in self.state.manual_components:
            ocr = component.get("ocr")
            if not isinstance(ocr, dict):
                continue
            ocr_index = int(ocr.get("index", -1))
            if ocr_index == index:
                component.pop("ocr", None)
            elif ocr_index > index:
                ocr["index"] = ocr_index - 1
        self._save_ocr_tsv()
        self.ocr_positions.clear()
        add_correction(
            self.state,
            "delete_ocr",
            f"Removed OCR '{text}'.",
            {"index": index, "entry": removed},
        )
        self._save_corrections_and_redraw()

    def _assign_ocr_to_element(self, area: str) -> None:
        """Copy the selected OCR text into a chosen manual component field and store the source OCR box metadata."""
        selected_ocr = self._selected_ocr()
        index_element = self._selected_element_index()
        if not selected_ocr or index_element is None:
            messagebox.showinfo("No selection", "Select OCR text and a manual component.", parent=self)
            return
        index_ocr, entry = selected_ocr
        text = _ocr_text(entry).strip()
        if not text:
            return
        self._push_undo()
        component = self.state.manual_components[index_element]
        component[area] = text
        component["ocr"] = {
            "index": index_ocr,
            "text": text,
            "side": _ocr_side(entry),
            "x": float(entry.get("x", 0.0)),
            "y": float(entry.get("y", 0.0)),
            "w": float(entry.get("w", 0.0)),
            "h": float(entry.get("h", 0.0)),
            "target_field": area,
        }
        add_correction(
            self.state,
            "assign_ocr",
            f"Assigned OCR '{text}' to field {area} on component {component.get('ref', '?')}.",
            {"component": component.get("ref", ""), "ocr_index": index_ocr, "field": area, "text": text},
        )
        self._save_corrections_and_redraw()
