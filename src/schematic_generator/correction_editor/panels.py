from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image

from schematic_generator.correction_editor.support import *
from schematic_generator.corrections_facade import add_correction
from schematic_generator.diagnostics_facade import save_problems
from schematic_generator.models import Problem
from schematic_generator.ocr import save_texts_ocr as save_ocr_texts


class CorrectionEditorProblemPanelMixin:
    def _build_panel_problems(self, panel: ttk.Notebook) -> None:
        """Assembles the diagnostics tab for filtering reconstruction problems and marking them open, ignored, or resolved."""
        tab = ttk.Frame(panel, padding=6)
        panel.add(tab, text="Problems")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        ttk.Label(tab, text="Filter").grid(row=0, column=0, sticky="w")
        filter_value = ttk.Combobox(
            tab,
            textvariable=self.problem_filter,
            values=("open", "high", "pads", "nets", "components", "ignored", "all"),
            state="readonly",
        )
        filter_value.grid(row=1, column=0, sticky="ew", **{_PAD_Y_KEY: (2, 6)})
        filter_value.bind("<<ComboboxSelected>>", lambda _event: self._refresh_problems())

        tree_frame = ttk.Frame(tab)
        tree_frame.grid(row=2, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree_problems = ttk.Treeview(
            tree_frame,
            columns=("severity", "category", "status"),
            show="tree headings",
            height=15,
        )
        self.tree_problems.heading("#0", text="Description")
        self.tree_problems.heading("severity", text="Risk")
        self.tree_problems.heading("category", text="Cat.")
        self.tree_problems.heading("status", text="Status")
        self.tree_problems.column("#0", width=180, stretch=True)
        self.tree_problems.column("severity", width=54, stretch=False)
        self.tree_problems.column("category", width=62, stretch=False)
        self.tree_problems.column("status", width=70, stretch=False)
        self.tree_problems.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tree_frame, command=self.tree_problems.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree_problems.configure(yscrollcommand=scrollbar.set)
        self.tree_problems.bind("<<TreeviewSelect>>", lambda _event: self._update_selected_problem())
        self.tree_problems.bind("<Double-1>", lambda _event: self._show_problem())

        ttk.Label(tab, textvariable=self.problem_counter).grid(row=3, column=0, sticky="w", **{_PAD_Y_KEY: (6, 0)})
        ttk.Label(tab, textvariable=self.problem_description, wraplength=315, justify="left").grid(
            row=4,
            column=0,
            sticky="ew",
            **{_PAD_Y_KEY: (4, 6)},
        )
        actions = ttk.Frame(tab)
        actions.grid(row=5, column=0, sticky="ew")
        ttk.Button(actions, text="Show", command=self._show_problem).pack(side=tk.LEFT)
        ttk.Button(actions, text="Ignore", command=lambda: self._set_problem_status("ignored")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Checked", command=lambda: self._set_problem_status("resolved")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(actions, text="Open", command=lambda: self._set_problem_status("open")).pack(side=tk.LEFT, padx=(4, 0))

    def _refresh_problems(self) -> None:
        """Repopulate the problem tree from current diagnostics and update the open/high/medium counters."""
        if not hasattr(self, "tree_problems"):
            return
        selected = self.tree_problems.selection()
        selected = selected[0] if selected else ""
        for item in self.tree_problems.get_children():
            self.tree_problems.delete(item)
        category_names = {"pads": "pads", "nets": "nets", "components": "components"}
        for problem in self._filtered_problems():
            description = _problem_description(problem) if len(_problem_description(problem)) <= 90 else _problem_description(problem)[:87] + "..."
            self.tree_problems.insert(
                "",
                "end",
                iid=_problem_identifier(problem),
                text=description,
                values=(problem.severity, category_names.get(_problem_category(problem), _problem_category(problem)), problem.status),
            )
        if selected and self.tree_problems.exists(selected):
            self.tree_problems.selection_set(selected)
        open_problems = [p for p in self.problems if p.status == "open"]
        high = sum(1 for p in open_problems if p.severity == "high")
        medium = sum(1 for p in open_problems if p.severity == "medium")
        self.problem_counter.set(f"Open: {len(open_problems)}  high={high}  medium={medium}  total={len(self.problems)}")

    def _filtered_problems(self) -> list[Problem]:
        """Apply the selected diagnostics filter without mutating the underlying problem list."""
        filter_value = self.problem_filter.get()
        if filter_value == "all":
            return list(self.problems)
        if filter_value == "open":
            return [p for p in self.problems if p.status == "open"]
        if filter_value == "high":
            return [p for p in self.problems if p.status == "open" and p.severity == "high"]
        if filter_value == "ignored":
            return [p for p in self.problems if p.status == "ignored"]
        category_filter_map = {"pads": "pads", "nets": "nets", "components": "components"}
        if filter_value in category_filter_map:
            return [p for p in self.problems if p.status == "open" and _problem_category(p) == category_filter_map[filter_value]]
        return list(self.problems)

    def _problem_by_id(self, identifier: str) -> Problem | None:
        """Find a diagnostic problem by its stable tree identifier."""
        for problem in self.problems:
            if _problem_identifier(problem) == identifier:
                return problem
        return None

    def _selected_problem(self) -> Problem | None:
        """Looks up the currently selected diagnostic problem when the tree selection still points to a valid item."""
        if not hasattr(self, "tree_problems"):
            return None
        choice = self.tree_problems.selection()
        return self._problem_by_id(choice[0]) if choice else None

    def _update_selected_problem(self) -> None:
        """Render the selected diagnostic details in the side panel without changing editor state."""
        problem = self._selected_problem()
        if not problem:
            self.problem_description.set("")
            return
        related_items = []
        category_names = {"pads": "pads", "nets": "nets", "components": "components"}
        for category, elements in _problem_related_items(problem).items():
            if elements:
                related_items.append(f"{category_names.get(category, category)}: {', '.join(elements[:6])}")
        text = f"{_problem_description(problem)}\nSuggestion: {_problem_suggestion(problem)}"
        if related_items:
            text += "\n" + "\n".join(related_items)
        self.problem_description.set(text)

    def _show_problem(self) -> None:
        """Highlight pads and positions referenced by the selected diagnostic and bring that context onto the board canvases."""
        problem = self._selected_problem()
        if not problem:
            return
        self.problem_nodes = set(_problem_related_items(problem).get("pads", []))
        self.problem_positions = dict(problem.positions)
        if self.problem_nodes:
            self.selected_node = sorted(self.problem_nodes)[0]
            self.selected_nodes.update(self.problem_nodes)
        self._update_selected_problem()
        self._refresh_status(extra=f"problem={problem.type}")
        self._redraw_all()

    def _set_problem_status(self, status: str) -> None:
        """Persist a status change for the selected diagnostic and refresh the problem list around the current filter."""
        problem = self._selected_problem()
        if not problem:
            return
        problem.status = status
        save_problems(self.state.work_folder, self.problems)
        self._refresh_problems()
        self._update_selected_problem()
