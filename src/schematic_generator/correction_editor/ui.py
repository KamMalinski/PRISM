from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from schematic_generator.correction_editor.support import *
from schematic_generator.gui.common import ui_font
from schematic_generator.resources import resource_path

class CorrectionEditorUiMixin:
    def _build_ui(self) -> None:
        """Assemble the toolbar, synchronized board canvases, side notebook, keyboard bindings, and status area used by the correction workflow."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")

        self.icons = self._create_icons()
        self.editor_brand_logo = self._load_editor_brand_logo()
        ttk.Label(toolbar, image=self.editor_brand_logo).pack(side=tk.LEFT, padx=(0, 10))
        self._build_tools_menu(toolbar)
        ttk.Label(toolbar, textvariable=self.label_mode, width=18, foreground="#333").pack(side=tk.LEFT, padx=(8, 14))
        ttk.Button(toolbar, text="Recalculate", image=self.icons.get("save"), compound=tk.LEFT, command=self._recalculate_save_redraw).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(toolbar, text="Brush").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(toolbar, from_=2, to=80, textvariable=self.brush_size, width=4).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Pad type").pack(side=tk.LEFT, padx=(10, 4))
        self.combo_type_pad = ttk.Combobox(
            toolbar,
            textvariable=self.type_pad,
            values=("pad", "via", "mounting_hole", "testpoint", "ignore"),
            state="readonly",
            width=13,
        )
        self.combo_type_pad.pack(side=tk.LEFT)
        self.combo_type_pad.bind("<<ComboboxSelected>>", lambda _event: self._set_type_pad())
        ttk.Button(toolbar, text="Set type", command=self._set_type_pad).pack(side=tk.LEFT, padx=(4, 0))

        self._build_view_and_action_menu(toolbar)
        self.bind("<Escape>", self._cancel_current_operation)
        self.bind("<Control-z>", self._undo_shortcut)
        self.bind("<Control-Z>", self._undo_shortcut)
        self.bind("<Control-y>", self._redo_shortcut)
        self.bind("<Control-Y>", self._redo_shortcut)
        self.bind("<Delete>", self._delete_selected)
        self.bind("<BackSpace>", self._delete_selected)
        self.bind("<KeyPress>", self._handle_tool_shortcut)

        area = ttk.Frame(self, padding=(8, 0, 8, 8))
        area.grid(row=1, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        area.columnconfigure(1, weight=1)
        area.columnconfigure(2, weight=0, minsize=340)
        area.rowconfigure(1, weight=1)

        for column, side in enumerate(("TOP", "BOTTOM")):
            ttk.Label(area, text=side, font=ui_font(self, 13, bold=True)).grid(row=0, column=column, sticky="w")
            frame_canvas = ttk.Frame(area)
            frame_canvas.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0))
            frame_canvas.columnconfigure(0, weight=1)
            frame_canvas.rowconfigure(0, weight=1)
            canvas = tk.Canvas(frame_canvas, background="#181818", highlightthickness=1, highlightbackground="#444")
            canvas.grid(row=0, column=0, sticky="nsew")
            hbar = ttk.Scrollbar(frame_canvas, orient=tk.HORIZONTAL, command=canvas.xview)
            vbar = ttk.Scrollbar(frame_canvas, orient=tk.VERTICAL, command=canvas.yview)
            hbar.grid(row=1, column=0, sticky="ew")
            vbar.grid(row=0, column=1, sticky="ns")
            canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
            canvas.bind("<Enter>", lambda _event, c=canvas: self._focus_canvas_without_dialog(c))
            canvas.bind("<Escape>", self._cancel_current_operation)
            canvas.bind("<Control-z>", self._undo_shortcut)
            canvas.bind("<Control-Z>", self._undo_shortcut)
            canvas.bind("<Control-y>", self._redo_shortcut)
            canvas.bind("<Control-Y>", self._redo_shortcut)
            canvas.bind("<Delete>", self._delete_selected)
            canvas.bind("<BackSpace>", self._delete_selected)
            canvas.bind("<KeyPress>", self._handle_tool_shortcut)
            canvas.bind("<Button-1>", lambda event, s=side: self._click(event, s))
            canvas.bind("<B1-Motion>", lambda event, s=side: self._drag(event, s))
            canvas.bind("<Motion>", lambda event, s=side: self._mouse_move(event, s))
            canvas.bind("<ButtonRelease-1>", lambda event, s=side: self._end_drag(event, s))
            canvas.bind("<MouseWheel>", lambda event, s=side: self._mouse_wheel(event, s))
            canvas.bind("<Button-4>", lambda event, s=side: self._mouse_wheel(event, s, 1))
            canvas.bind("<Button-5>", lambda event, s=side: self._mouse_wheel(event, s, -1))
            canvas.bind("<ButtonPress-2>", lambda event, c=canvas: c.scan_mark(event.x, event.y))
            canvas.bind("<B2-Motion>", lambda event, c=canvas: c.scan_dragto(event.x, event.y, gain=1))
            canvas.bind("<ButtonPress-3>", lambda event, c=canvas: c.scan_mark(event.x, event.y))
            canvas.bind("<B3-Motion>", lambda event, c=canvas: c.scan_dragto(event.x, event.y, gain=1))
            canvas.bind("<Configure>", lambda _event: self._redraw_all())
            self.canvas_info[side] = {"canvas": canvas, "scale": 1.0, "tk_image": None}

        panel = ttk.Notebook(area)
        panel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(8, 0))
        self._build_panel_problems(panel)
        self._build_panel_elements(panel)
        self._build_panel_ocr(panel)

        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.grid(row=2, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Label(
            bottom,
            text="Clicking a pad highlights the pair on the other side; Connect pair mode requires clicking both pads.",
            foreground="#555",
        ).pack(side=tk.RIGHT)
        self._set_tool_cursor()

    def _load_editor_brand_logo(self) -> ImageTk.PhotoImage:
        """Load a compact copy of the packaged brand artwork for the editor toolbar."""

        image = Image.open(resource_path("assets/icon.png")).convert("RGBA")
        image.thumbnail((44, 44), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _build_tools_menu(self, toolbar: ttk.Frame) -> None:
        """Assembles grouped tool menus for pad editing, pair editing, trace painting, OCR selection, and component selection modes."""
        groups = [
            ("Select", "move", [
                ("move", "Select / move", "move"),
            ]),
            ("Pads", "pad", [
                ("add", "Add pad", "add"),
                ("delete", "Delete pad", "delete"),
                ("rename_pad", "Name pad", "text"),
            ]),
            ("Pairs", "pair", [
                ("pair", "Connect pair", "pair"),
                ("unpair", "Disconnect pair", "unpair"),
            ]),
            ("Elements", "component", [
                ("component", "Select component pads", "component"),
                ("__create_component__", "Create component from selection", "component"),
            ]),
            ("Traces", "trace", [
                ("trace_add", "Add trace", "trace_add"),
                ("trace_remove", "Delete trace", "trace_remove"),
                ("trace_bridge", "Bridge", "bridge"),
                ("trace_cut", "Cut", "cut"),
                ("trace_ignore", "Ignore", "ignore"),
            ]),
            ("OCR", "ocr", [
                ("ocr_region", "OCR area", "ocr"),
                ("ocr_manual_region", "Manual OCR", "text"),
            ]),
        ]
        for name, icon, positions in groups:
            button = ttk.Menubutton(toolbar, text=f" {name}", image=self.icons.get(icon), compound=tk.LEFT)
            menu = tk.Menu(button, tearoff=False)
            for mode, label, icon_position in positions:
                if mode == "__create_component__":
                    menu.add_command(label=label, image=self.icons.get(icon_position), compound=tk.LEFT, command=self._create_element_from_selection)
                else:
                    menu.add_command(
                        label=label,
                        accelerator=_SHORTCUTS_MODES.get(mode, ""),
                        image=self.icons.get(icon_position),
                        compound=tk.LEFT,
                        command=lambda t=mode: self._set_mode(t),
                    )
            button["menu"] = menu
            button.pack(side=tk.LEFT, padx=(0, 4))

    def _build_view_and_action_menu(self, toolbar: ttk.Frame) -> None:
        """Assembles view toggles and global actions for trace overlays, pair labels, OCR overlays, undo, redo, and clearing selections."""
        actions = ttk.Menubutton(toolbar, text=" View / actions", image=self.icons.get("view"), compound=tk.LEFT)
        actions_menu = tk.Menu(actions, tearoff=False)
        actions_menu.add_checkbutton(label="Traces", variable=self.show_traces, command=self._redraw_all)
        actions_menu.add_checkbutton(label="Colored traces", variable=self.colored_traces, command=self._redraw_all)
        actions_menu.add_checkbutton(label="Pairs", variable=self.show_pairs, command=self._redraw_all)
        actions_menu.add_checkbutton(label="Nets", variable=self.show_nets, command=self._redraw_all)
        actions_menu.add_checkbutton(label="OCR", variable=self.show_ocr, command=self._redraw_all)
        actions_menu.add_separator()
        actions_menu.add_command(label="Zoom +", image=self.icons.get("zoom_in"), compound=tk.LEFT, command=lambda: self._change_zoom(None, 1.25))
        actions_menu.add_command(label="Zoom -", image=self.icons.get("zoom_out"), compound=tk.LEFT, command=lambda: self._change_zoom(None, 0.8))
        actions_menu.add_command(label="Fit", image=self.icons.get("fit"), compound=tk.LEFT, command=self._align_zoom)
        actions_menu.add_separator()
        for label, icon, command in (
            ("Clear selection", "clear", self._clear_selection),
            ("Undo", "undo", self._undo),
            ("Redo", "redo", self._redo),
        ):
            actions_menu.add_command(label=label, image=self.icons.get(icon), compound=tk.LEFT, command=command)
        actions["menu"] = actions_menu
        actions.pack(side=tk.LEFT, padx=(0, 10))

    def _set_mode(self, mode: str) -> None:
        """Switch the active editing mode and refresh cursor and label feedback so subsequent canvas clicks use the selected tool."""
        self.mode.set(mode)
        self._update_mode_label()
        self._set_tool_cursor()

    def _update_mode_label(self) -> None:
        """Synchronize the toolbar mode label with the current Tk mode variable."""
        self.label_mode.set(_NAMES_MODES.get(self.mode.get(), self.mode.get()))

    def _set_tool_cursor(self) -> None:
        """Apply the cursor shape that matches the active tool on both board canvases."""
        cursor = _cursor_for_mode(self.mode.get())
        for info in self.canvas_info.values():
            canvas = info.get("canvas")
            if isinstance(canvas, tk.Canvas):
                canvas.configure(cursor=cursor)

    def _focus_canvas_without_dialog(self, canvas: tk.Canvas) -> None:
        """Move keyboard focus to a canvas only when no modal editor dialog is currently active."""
        grab = self.grab_current()
        if grab is not None and grab.winfo_toplevel() is not self:
            return
        focus = self.focus_get()
        if focus is not None and focus.winfo_toplevel() is not self:
            return
        canvas.focus_set()

    def _set_modal_window(self, window: tk.Toplevel, focus_widget: tk.Widget | None = None) -> None:
        """Configure a child dialog as modal, focus its first input, and keep keyboard shortcuts scoped to that dialog."""
        window.transient(self)
        window.lift()
        window.grab_set()
        window.bind("<Escape>", lambda _event: window.destroy())
        if focus_widget is not None:
            window.after_idle(focus_widget.focus_set)

    def _undo_shortcut(self, _event: tk.Event | None = None) -> str:
        """Route Ctrl+Z from the editor or canvases into the undo stack and stop Tk from handling it twice."""
        self._undo()
        return "break"

    def _redo_shortcut(self, _event: tk.Event | None = None) -> str:
        """Route Ctrl+Y from the editor or canvases into the redo stack and stop Tk from handling it twice."""
        self._redo()
        return "break"

    def _handle_tool_shortcut(self, event: tk.Event) -> str | None:
        """Translate single-key shortcuts into editor modes while ignoring keystrokes that belong to focused text inputs."""
        if event.state & 0x0004:
            return None
        widget = getattr(event, "widget", None)
        if isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox)):
            return None
        shortcut_key = str(getattr(event, "char", "")).lower()
        if not shortcut_key:
            return None
        mode = _MODE_BY_SHORTCUT.get(shortcut_key)
        if not mode:
            return None
        self._set_mode(mode)
        return "break"

    def _create_icons(self) -> dict[str, ImageTk.PhotoImage]:
        """Generates small in-memory toolbar icons so the editor has no external image dependency for its controls."""
        color = (35, 73, 120, 255)
        accent_color = (32, 156, 119, 255)
        warning_color = (210, 65, 65, 255)

        def image(draw_fn: Callable[[ImageDraw.ImageDraw], None]) -> ImageTk.PhotoImage:
            """Render one toolbar icon into a Tk image and keep the drawing callback isolated from Tk widget setup."""
            img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            draw_fn(ImageDraw.Draw(img, "RGBA"))
            return ImageTk.PhotoImage(img)

        return {
            "move": image(lambda d: (d.polygon([(4, 3), (15, 10), (10, 11), (13, 17), (10, 18), (7, 12), (4, 15)], fill=color),)),
            "component": image(lambda d: (d.rectangle((4, 5, 16, 15), outline=accent_color, width=2), d.line((7, 10, 13, 10), fill=accent_color, width=2))),
            "pad": image(lambda d: (d.ellipse((4, 4, 16, 16), outline=color, width=2), d.ellipse((8, 8, 12, 12), fill=color))),
            "add": image(lambda d: (d.ellipse((4, 4, 16, 16), outline=accent_color, width=2), d.line((10, 6, 10, 14), fill=accent_color, width=2), d.line((6, 10, 14, 10), fill=accent_color, width=2))),
            "delete": image(lambda d: (d.ellipse((4, 4, 16, 16), outline=warning_color, width=2), d.line((7, 7, 13, 13), fill=warning_color, width=2), d.line((13, 7, 7, 13), fill=warning_color, width=2))),
            "pair": image(lambda d: (d.ellipse((3, 7, 9, 13), outline=color, width=2), d.ellipse((11, 7, 17, 13), outline=color, width=2), d.line((8, 10, 12, 10), fill=accent_color, width=2))),
            "unpair": image(lambda d: (d.ellipse((3, 7, 9, 13), outline=color, width=2), d.ellipse((11, 7, 17, 13), outline=color, width=2), d.line((8, 12, 12, 8), fill=warning_color, width=2))),
            "trace": image(lambda d: d.line((3, 14, 8, 6, 13, 6, 17, 3), fill=color, width=3)),
            "trace_add": image(lambda d: (d.line((3, 14, 8, 6, 13, 6), fill=accent_color, width=3), d.line((15, 5, 15, 13), fill=accent_color, width=2), d.line((11, 9, 19, 9), fill=accent_color, width=2))),
            "trace_remove": image(lambda d: (d.line((3, 14, 8, 6, 17, 6), fill=warning_color, width=3), d.line((5, 5, 15, 15), fill=warning_color, width=2))),
            "bridge": image(lambda d: (d.arc((3, 5, 17, 17), 200, 340, fill=accent_color, width=3), d.line((3, 14, 17, 14), fill=color, width=2))),
            "cut": image(lambda d: (d.line((5, 5, 15, 15), fill=warning_color, width=2), d.line((15, 5, 5, 15), fill=warning_color, width=2), d.ellipse((4, 4, 7, 7), outline=color), d.ellipse((13, 13, 16, 16), outline=color))),
            "ignore": image(lambda d: (d.rectangle((4, 4, 16, 16), outline=warning_color, width=2), d.line((5, 15, 15, 5), fill=warning_color, width=2))),
            "ocr": image(lambda d: (d.rectangle((3, 5, 17, 15), outline=color, width=2), d.text((6, 5), "T", fill=color))),
            "text": image(lambda d: (d.rectangle((4, 4, 16, 16), outline=accent_color, width=2), d.line((7, 8, 13, 8), fill=accent_color, width=2), d.line((7, 12, 12, 12), fill=accent_color, width=2))),
            "save": image(lambda d: (d.rectangle((4, 3, 16, 17), outline=color, width=2), d.rectangle((7, 4, 13, 8), fill=accent_color), d.rectangle((7, 12, 13, 16), outline=color))),
            "clear": image(lambda d: (d.rectangle((5, 7, 15, 15), outline=warning_color, width=2), d.line((7, 5, 13, 5), fill=warning_color, width=2))),
            "undo": image(lambda d: (d.arc((4, 5, 16, 17), 80, 300, fill=color, width=2), d.polygon([(5, 6), (10, 4), (9, 9)], fill=color))),
            "redo": image(lambda d: (d.arc((4, 5, 16, 17), 240, 100, fill=color, width=2), d.polygon([(15, 6), (10, 4), (11, 9)], fill=color))),
            "view": image(lambda d: (d.ellipse((3, 6, 17, 14), outline=color, width=2), d.ellipse((8, 8, 12, 12), fill=color))),
            "zoom_in": image(lambda d: (d.ellipse((4, 4, 12, 12), outline=color, width=2), d.line((11, 11, 17, 17), fill=color, width=2), d.line((8, 6, 8, 10), fill=accent_color, width=2), d.line((6, 8, 10, 8), fill=accent_color, width=2))),
            "zoom_out": image(lambda d: (d.ellipse((4, 4, 12, 12), outline=color, width=2), d.line((11, 11, 17, 17), fill=color, width=2), d.line((6, 8, 10, 8), fill=accent_color, width=2))),
            "fit": image(lambda d: (d.rectangle((4, 4, 16, 16), outline=color, width=2), d.line((4, 8, 8, 4), fill=accent_color, width=2), d.line((16, 12, 12, 16), fill=accent_color, width=2))),
        }

    def _mode_changed(self) -> None:
        """React to changes in the current tool by updating toolbar text, cursor feedback, and status guidance."""
        if hasattr(self, "mode_label_widget"):
            self._update_mode_label()
        if hasattr(self, "canvas_info"):
            self._set_tool_cursor()
        if self.mode.get() in {"ocr_region", "ocr_manual_region"}:
            return
        if self.ocr_region_start or self.ocr_region_preview:
            self.ocr_region_start = None
            self.ocr_region_preview.clear()
            self._redraw_all()
