from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from schematic_generator.analysis import run_analysis
from schematic_generator.automatic import (
    AUTOMATIC_FILTER_COMBINATION_COUNT,
    AUTOMATIC_GROUNDPLANE,
    AUTOMATIC_ALIGNMENT_THRESHOLD_PERCENT,
)
from schematic_generator.colors import ColorCandidate, find_suggestions_colors
from schematic_generator.images import load_image
from schematic_generator.models import AnalysisReport
from schematic_generator.resources import resource_path

from schematic_generator.gui.common import SAMPLE_NAMES, SAMPLE_STEPS, hex_from_bgr, text_on_background, ui_font


class ApplicationUiMixin:
    """Provide application ui behavior for the main Tk window."""

    def _build_interface(self) -> None:
        """Create the main two-column layout with controls on the left and preview/log on the right."""
        container = ttk.Frame(self, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        panel_container = ttk.Frame(container, width=330)
        panel_container.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        panel_container.grid_propagate(False)
        panel_container.columnconfigure(0, weight=1)
        panel_container.rowconfigure(0, weight=1)
        panel_canvas = tk.Canvas(panel_container, width=310, highlightthickness=0)
        panel_scroll = ttk.Scrollbar(panel_container, orient=tk.VERTICAL, command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scroll.set)
        panel_canvas.grid(row=0, column=0, sticky="nsew")
        panel_scroll.grid(row=0, column=1, sticky="ns")
        panel = ttk.Frame(panel_canvas)
        panel_window = panel_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _e: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
        panel_canvas.bind("<Configure>", lambda e: panel_canvas.itemconfigure(panel_window, width=e.width))
        panel_canvas.bind("<Enter>", lambda _event: self._bind_panel_mouse_wheel(panel_canvas))
        panel_canvas.bind("<Leave>", lambda _event: self._unbind_panel_mouse_wheel(panel_canvas))

        preview = ttk.Frame(container)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        self._build_panel_steps(panel)
        view_bar = ttk.Frame(preview)
        view_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=8)
        ttk.Label(view_bar, text="View").pack(side=tk.LEFT)
        self.list_previews = ttk.Combobox(
            view_bar,
            textvariable=self.preview_view,
            state="disabled",
            values=[],
            width=34,
        )
        self.list_previews.pack(side=tk.LEFT, padx=(8, 0))
        self.list_previews.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_preview())
        ttk.Label(view_bar, textvariable=self.status_preview, foreground="#555").pack(side=tk.LEFT, padx=(12, 0))

        self.label_preview = ttk.Label(preview, anchor="center")
        self.label_preview.grid(row=1, column=0, sticky="nsew")
        self.label_preview.bind("<Button-1>", self._handle_preview_click)
        self.label_preview.configure(cursor="crosshair")

        self.log = tk.Text(preview, height=11, wrap="word")
        self.log.grid(row=2, column=0, sticky="ew", padx=10, pady=0)
        self._log("Choose TOP and BOTTOM images to start.")
        self._refresh_preview_colors()

    def _bind_panel_mouse_wheel(self, canvas: tk.Canvas) -> None:
        """Bind Windows/macOS and Linux wheel events while the pointer is over the panel."""

        canvas.bind_all(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"),
        )
        canvas.bind_all("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))

    def _unbind_panel_mouse_wheel(self, canvas: tk.Canvas) -> None:
        """Remove global wheel bindings when the pointer leaves the control panel."""

        for event_name in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.unbind_all(event_name)

    def _build_panel_steps(self, panel: ttk.Frame) -> None:
        """Populate the control panel with a guided vertical workflow accordion."""
        self.main_icons = self._create_main_icons()
        self.accordion_headers: dict[int, tk.Button] = {}
        self.accordion_bodies: dict[int, ttk.Frame] = {}
        self.active_step = 1

        self.brand_logo = self._load_brand_logo((112, 112))
        ttk.Label(panel, image=self.brand_logo).pack(anchor="center", pady=(2, 10))

        import_body = self._add_accordion_section(panel, 1, "Import images")
        self._build_import_step(import_body)

        color_body = self._add_accordion_section(panel, 2, "Color Samples")
        self._build_color_step(color_body)

        analysis_body = self._add_accordion_section(panel, 3, "Analysis")
        self._build_analysis_step(analysis_body)

        corrections_body = self._add_accordion_section(panel, 4, "Manual Corrections")
        self._build_corrections_step(corrections_body)

        results_body = self._add_accordion_section(panel, 5, "Results")
        self._build_results_step(results_body)

        self._build_footer(panel)
        self._set_active_step(1)

    def _load_brand_logo(self, size: tuple[int, int]) -> ImageTk.PhotoImage:
        """Load the packaged brand artwork and fit it inside the requested display area."""

        image = Image.open(resource_path("assets/icon.png")).convert("RGBA")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _build_footer(self, panel: ttk.Frame) -> None:
        """Build the formatted project footer."""
        footer = ttk.Frame(panel)
        footer.pack(fill="x", padx=10, pady=(14, 0))

        ttk.Label(
            footer,
            text="PRISM",
            font=ui_font(self, 15, bold=True),
            anchor="center",
            justify="center",
        ).pack(fill="x")
        ttk.Label(
            footer,
            text="PCB Recognition and Intelligent Schematic Mapping",
            font=ui_font(self, 9, bold=True),
            anchor="center",
            justify="center",
            wraplength=280,
        ).pack(fill="x", pady=(2, 0))
        ttk.Label(
            footer,
            text="Kamil Maliński 2026",
            anchor="center",
            justify="center",
            foreground="#444",
        ).pack(fill="x", pady=(4, 10))
        ttk.Label(
            footer,
            text="Open-source tools",
            font=ui_font(self, 9, bold=True),
            foreground="#444",
        ).pack(anchor="w")
        for tool in (
            "Python",
            "Tcl/Tk",
            "OpenCV",
            "NumPy",
            "Pillow",
            "pytesseract",
            "Tesseract OCR",
            "KiCad CLI",
            "PyInstaller",
        ):
            ttk.Label(footer, text=f"• {tool}", foreground="#444").pack(anchor="w", padx=(8, 0))

    def _build_import_step(self, panel: ttk.Frame) -> None:
        """Build image import controls and board-level settings."""
        self._icon_button(panel, "Choose TOP", "image", self._select_top).pack(fill="x")
        self.lbl_top = ttk.Label(panel, text="---", wraplength=280)
        self.lbl_top.pack(anchor="w", padx=3, pady=(4, 8))

        self._icon_button(panel, "Choose BOTTOM", "image", self._select_bottom).pack(fill="x")
        self.lbl_bottom = ttk.Label(panel, text="---", wraplength=280)
        self.lbl_bottom.pack(anchor="w", padx=3, pady=(4, 8))

        ttk.Label(panel, text="Schematic name").pack(anchor="w")
        ttk.Entry(panel, textvariable=self.schematic_name).pack(fill="x", padx=0, pady=(2, 8))

        ttk.Label(panel, text="Tesseract exe").pack(anchor="w")
        tesseract_row = ttk.Frame(panel)
        tesseract_row.pack(fill="x", padx=0, pady=(2, 4))
        ttk.Entry(tesseract_row, textvariable=self.path_tesseract).pack(side=tk.LEFT, fill="x", expand=True)
        ttk.Button(
            tesseract_row,
            text="...",
            width=3,
            image=self.main_icons.get("browse"),
            compound=tk.LEFT,
            command=self._select_tesseract,
        ).pack(side=tk.LEFT, padx=(6, 0))

    def _build_color_step(self, panel: ttk.Frame) -> None:
        """Build guided and automatic color sampling controls."""
        self._icon_button(panel, "Automatic color choice", "auto", self._choose_colors_automatically).pack(fill="x")
        wizard_row = ttk.Frame(panel)
        wizard_row.pack(fill="x", padx=4, pady=(6, 0))
        ttk.Button(
            wizard_row,
            text="Sample wizard",
            image=self.main_icons.get("pick"),
            compound=tk.LEFT,
            command=self._start_sample_wizard,
        ).pack(side=tk.LEFT, fill="x", expand=True)
        ttk.Label(panel, textvariable=self.sample_wizard_status, wraplength=280, foreground="#555").pack(anchor="w", padx=4, pady=6)
        ttk.Radiobutton(panel, text="Manual: trace / copper", variable=self.mode_samples, value="trace_bgr").pack(anchor="w")
        ttk.Radiobutton(panel, text="Manual: soldermask", variable=self.mode_samples, value="background_bgr").pack(anchor="w")
        self.lbl_samples = ttk.Label(panel, text="TOP: no samples\nBOTTOM: no samples", wraplength=280)
        self.lbl_samples.pack(anchor="w", padx=6, pady=8)
        ttk.Label(panel, text="5 dominant colors").pack(anchor="w", padx=2, pady=2)
        self.frame_colors = ttk.Frame(panel)
        self.frame_colors.pack(fill="x", padx=0, pady=8)
        ttk.Checkbutton(panel, text="PCB has a groundplane", variable=self.has_groundplane).pack(anchor="w")

    def _build_analysis_step(self, panel: ttk.Frame) -> None:
        """Build analysis parameters and the single manual analysis start action."""
        ttk.Label(panel, text="Number of filter combinations").pack(anchor="w")
        filter_row = ttk.Frame(panel)
        filter_row.pack(fill="x", padx=0, pady=8)
        self.filter_slider = ttk.Scale(
            filter_row,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._change_count_filters,
        )
        self.filter_count_label = ttk.Label(filter_row, text=str(self.filter_combination_count.get()), width=3)
        self.filter_slider.set(self.filter_combination_count.get())
        self.filter_slider.pack(side=tk.LEFT, fill="x", expand=True)
        self.filter_count_label.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(panel, text="Alignment threshold [%]").pack(anchor="w")
        alignment_row = ttk.Frame(panel)
        alignment_row.pack(fill="x", padx=0, pady=8)
        self.alignment_slider = ttk.Scale(
            alignment_row,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._change_threshold_alignment,
        )
        self.alignment_threshold_label = ttk.Label(alignment_row, text=f"{self.alignment_threshold.get()}%", width=5)
        self.alignment_slider.set(self.alignment_threshold.get())
        self.alignment_slider.pack(side=tk.LEFT, fill="x", expand=True)
        self.alignment_threshold_label.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(panel, text="Component solver").pack(anchor="w")
        ttk.Radiobutton(
            panel,
            text="Sequential solver",
            variable=self.component_solver_mode,
            value="sequential",
        ).pack(anchor="w")
        ttk.Radiobutton(
            panel,
            text="Global solver",
            variable=self.component_solver_mode,
            value="global",
        ).pack(anchor="w", pady=(0, 8))

        self.button_analysis = self._icon_button(panel, "Start Analysis", "start", self._analyze)
        self.button_analysis.pack(fill="x")
        self.progress_bar = ttk.Progressbar(panel, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=8, pady=(6, 0))

    def _build_corrections_step(self, panel: ttk.Frame) -> None:
        """Build manual correction actions shown after a successful analysis."""
        self._icon_button(panel, "Correction editor", "edit", self._open_editor).pack(fill="x")
        self._icon_button(panel, "Skip manual corrections", "skip", self._skip_manual_corrections).pack(fill="x", pady=(6, 0))

    def _build_results_step(self, panel: ttk.Frame) -> None:
        """Build final result actions."""
        self._icon_button(panel, "Show netlist", "netlist", self._show_netlist).pack(fill="x")
        self._icon_button(panel, "Open result directory", "folder", self._open_folder).pack(fill="x", pady=(6, 0))
        self._icon_button(panel, "New board", "new", self._new_board).pack(fill="x", pady=(6, 0))

    def _add_accordion_section(self, panel: ttk.Frame, step: int, title: str) -> ttk.Frame:
        """Create one accordion section and return its body frame."""
        section = ttk.Frame(panel)
        section.pack(fill="x", pady=(8 if step == 1 else 4, 0))
        header = tk.Button(
            section,
            anchor="w",
            background="#e4edf7",
            activebackground="#c9ddf2",
            foreground="#1f3552",
            activeforeground="#1f3552",
            relief=tk.FLAT,
            borderwidth=1,
            padx=10,
            pady=6,
            font=ui_font(self, 10, bold=True),
            justify=tk.LEFT,
            command=lambda s=step: self._toggle_step(s),
        )
        header.pack(fill="x")
        body = ttk.Frame(section, padding=(8, 8, 4, 4))
        self.accordion_headers[step] = header
        self.accordion_bodies[step] = body
        header.configure(text=f"{step}. {title}", image=self.main_icons.get("section_closed"), compound=tk.LEFT)
        return body

    def _toggle_step(self, step: int) -> None:
        """Expand a selected accordion section and collapse the others."""
        body = self.accordion_bodies.get(step)
        if body is not None and body.winfo_ismapped() and self.active_step == step:
            body.pack_forget()
            self.active_step = 0
            self._refresh_accordion_headers()
            return
        self._set_active_step(step)

    def _set_active_step(self, step: int) -> None:
        """Programmatically move the guided workflow to a step."""
        self.active_step = step
        for number, body in self.accordion_bodies.items():
            if number == step:
                body.pack(fill="x")
            else:
                body.pack_forget()
        self._refresh_accordion_headers()

    def _refresh_accordion_headers(self) -> None:
        """Update accordion header markers after a state change."""
        labels = {
            1: "Import images",
            2: "Color Samples",
            3: "Analysis",
            4: "Manual Corrections",
            5: "Results",
        }
        for step, header in self.accordion_headers.items():
            background = "#cfe1f5" if self.active_step == step else "#e4edf7"
            relief = tk.SOLID if self.active_step == step else tk.FLAT
            icon = self.main_icons.get("section_open" if self.active_step == step else "section_closed")
            header.configure(text=f"{step}. {labels[step]}", image=icon, compound=tk.LEFT)
            header.configure(background=background, activebackground="#c9ddf2", relief=relief)

    def _icon_button(self, parent: tk.Misc, text: str, icon: str, command: Callable[[], None]) -> ttk.Button:
        """Create a text button with a small generated icon."""
        return ttk.Button(parent, text=text, image=self.main_icons.get(icon), compound=tk.LEFT, command=command)

    def _create_main_icons(self) -> dict[str, ImageTk.PhotoImage]:
        """Generate small in-memory icons for the main workflow buttons."""
        color = (35, 73, 120, 255)
        accent = (32, 156, 119, 255)
        warning = (210, 65, 65, 255)

        def image(draw_fn: Callable[[ImageDraw.ImageDraw], None]) -> ImageTk.PhotoImage:
            img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            draw_fn(ImageDraw.Draw(img, "RGBA"))
            return ImageTk.PhotoImage(img)

        return {
            "image": image(lambda d: (d.rectangle((3, 5, 17, 15), outline=color, width=2), d.polygon([(5, 14), (9, 9), (12, 12), (15, 8), (17, 15)], fill=accent))),
            "browse": image(lambda d: (d.rectangle((3, 6, 17, 15), outline=color, width=2), d.rectangle((5, 4, 11, 7), fill=accent))),
            "auto": image(lambda d: (d.arc((4, 4, 16, 16), 35, 320, fill=accent, width=2), d.polygon([(15, 4), (17, 10), (11, 8)], fill=accent))),
            "pick": image(lambda d: (d.line((5, 15, 14, 6), fill=color, width=3), d.rectangle((12, 3, 17, 8), outline=accent, width=2))),
            "check": image(lambda d: d.line((4, 10, 8, 15, 16, 5), fill=accent, width=3)),
            "start": image(lambda d: d.polygon([(6, 4), (16, 10), (6, 16)], fill=accent)),
            "edit": image(lambda d: (d.rectangle((4, 4, 16, 16), outline=color, width=2), d.line((7, 13, 14, 6), fill=accent, width=2))),
            "skip": image(lambda d: (d.line((5, 5, 10, 10, 5, 15), fill=color, width=2), d.line((11, 5, 16, 10, 11, 15), fill=color, width=2))),
            "netlist": image(lambda d: (d.rectangle((5, 3, 15, 17), outline=color, width=2), d.line((7, 8, 13, 8), fill=accent, width=2), d.line((7, 12, 13, 12), fill=accent, width=2))),
            "folder": image(lambda d: (d.rectangle((3, 7, 17, 16), outline=color, width=2), d.rectangle((5, 4, 11, 8), fill=accent))),
            "new": image(lambda d: (d.rectangle((5, 4, 15, 16), outline=color, width=2), d.line((10, 7, 10, 13), fill=accent, width=2), d.line((7, 10, 13, 10), fill=accent, width=2))),
            "delete": image(lambda d: (d.ellipse((4, 4, 16, 16), outline=warning, width=2), d.line((7, 7, 13, 13), fill=warning, width=2), d.line((13, 7, 7, 13), fill=warning, width=2))),
            "section_closed": image(lambda d: d.polygon([(7, 5), (14, 10), (7, 15)], fill=color)),
            "section_open": image(lambda d: d.polygon([(5, 7), (15, 7), (10, 14)], fill=color)),
        }
