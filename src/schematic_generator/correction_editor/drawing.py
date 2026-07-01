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


class CorrectionEditorDrawingMixin:
    def _redraw_all(self) -> None:
        """Redraw both board canvases after a state, zoom, or overlay change."""
        if not self.canvas_info:
            return
        for side in ("TOP", "BOTTOM"):
            self._redraw_side(side)

    def _redraw_side(self, side: str) -> None:
        """Compose one side image from the base photo, trace mask, pairs, pads, OCR boxes, diagnostics, and manual components."""
        info = self.canvas_info[side]
        canvas = info["canvas"]
        if not isinstance(canvas, tk.Canvas):
            return

        image = self._image_base_side(side).convert("RGBA")
        if self.show_traces.get():
            if self.colored_traces.get():
                image = self._image_colored_traces(side).convert("RGBA")
            else:
                alpha = self.mask[side].point(lambda p: 95 if p > 127 else 0)
                color = Image.new("RGBA", image.size, (20, 220, 90, 0))
                color.putalpha(alpha)
                image = Image.alpha_composite(image, color)

        draw = ImageDraw.Draw(image, "RGBA")
        pads_by_node = self._pads_by_node()
        self._draw_elements_manual(draw, side)
        self._draw_selected_trace(draw, side)
        if self.show_pairs.get():
            self._draw_pairs_on_side(draw, pads_by_node, side)
        self._draw_pads(draw, side)
        self._draw_positions_problem(draw, side)
        if self.show_ocr.get():
            self._draw_ocr(draw, side)
        self._draw_region_ocr_preview(draw, side)

        max_w = max(120, canvas.winfo_width() - 8)
        max_h = max(120, canvas.winfo_height() - 8)
        scale = min(max_w / image.width, max_h / image.height, 1.0) * self.zoom[side]
        info["scale"] = scale
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image)
        info["tk_image"] = image_tk
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
        canvas.config(scrollregion=(0, 0, size[0], size[1]))

    def _image_base_side(self, side: str) -> Image.Image:
        """Provides a fresh copy of one side background image so drawing overlays never mutate the cached original."""
        return self.images[side].copy()

    def _image_colored_traces(self, side: str) -> Image.Image:
        """Render a colored trace preview for one side using the current mask and pads."""
        image_rgb = np.array(self.images[side].convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask = np.array(self.mask[side], dtype=np.uint8)
        preview = draw_connections_colored(image_bgr, mask, self._pads_side(side))
        return Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

    def _draw_pairs_on_side(self, draw: ImageDraw.ImageDraw, pads_by_node: dict[str, Pad], side: str) -> None:
        """Renders TOP/BOTTOM pair indicators around pads and emphasizes the pair related to the current selection."""
        for index, pair in enumerate(self.state.pairs):
            pad = pads_by_node.get(pair.pad_top if side == "TOP" else pair.pad_bottom)
            second = pads_by_node.get(pair.pad_bottom if side == "TOP" else pair.pad_top)
            if not pad or not second:
                continue
            selected = pad.node == self.selected_node or second.node == self.selected_node
            base = _color_pairs(index)
            color = (255, 255, 255, 255) if selected else (*base, 215)
            r = max(6.0, pad.radius + (6.0 if selected else 3.0))
            draw.ellipse((pad.x - r, pad.y - r, pad.x + r, pad.y + r), outline=color, width=3 if selected else 2)

    def _draw_pads(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Renders every pad marker, selection highlight, diagnostic highlight, pair color, and optional net label for one board side."""
        paired = self._paired_nodes()
        colors_pairs = self._colors_pairs_by_nodes()
        counterpart_node = self._counterpart(self.selected_node) if self.selected_node else None
        for pad in self._pads_side(side):
            r = max(4.0, pad.radius)
            color = _color_net(pad.net)
            if pad.node == self.selected_node:
                outline_color = (255, 255, 255, 255)
                fill_color = (255, 130, 0, 150)
            elif pad.type == "ignore":
                outline_color = (120, 120, 120, 210)
                fill_color = (80, 80, 80, 50)
            elif pad.node == counterpart_node:
                outline_color = (255, 255, 255, 255)
                fill_color = (0, 190, 255, 150)
            elif pad.node in self.selected_nodes:
                outline_color = (255, 255, 255, 240)
                fill_color = (170, 90, 255, 130)
            elif pad.node in self.problem_nodes:
                outline_color = (255, 0, 0, 255)
                fill_color = (255, 0, 0, 125)
            elif _pad_identifier(pad).startswith("M"):
                outline_color = (80, 190, 255, 255)
                fill_color = (80, 190, 255, 75)
            elif pad.node not in paired:
                outline_color = (255, 70, 70, 240)
                fill_color = (255, 70, 70, 60)
            elif pad.node in colors_pairs:
                outline_color = (*colors_pairs[pad.node], 245)
                fill_color = (*colors_pairs[pad.node], 70)
            else:
                outline_color = (*color, 245)
                fill_color = (*color, 70)
            draw.ellipse((pad.x - r, pad.y - r, pad.x + r, pad.y + r), fill=fill_color, outline=outline_color, width=2)
            if self.show_nets.get():
                text = _label_pad(pad)
                if pad.net:
                    text = f"{text}:{pad.net}" if pad.name else pad.net
                if pad.type not in {"pad", ""}:
                    text = f"{pad.type}:{text}"
                draw.text((pad.x + r + 2, pad.y - r - 2), text, fill=(255, 255, 255, 230))

    def _draw_positions_problem(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Renders a crosshair at a diagnostic position that is not tied to a specific pad node."""
        position = self.problem_positions.get(side)
        if not position or len(position) < 2:
            return
        x, y = float(position[0]), float(position[1])
        r = 18.0
        draw.line((x - r, y, x + r, y), fill=(255, 0, 0, 255), width=4)
        draw.line((x, y - r, x, y + r), fill=(255, 0, 0, 255), width=4)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, 255), width=2)

    def _draw_selected_trace(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Render a marker around the selected trace component."""
        if not self.selected_trace or self.selected_trace[0] != side:
            return
        _side, x, y = self.selected_trace
        r = 16.0
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, 255), width=3)
        draw.line((x - r, y, x + r, y), fill=(255, 130, 0, 255), width=3)
        draw.line((x, y - r, x, y + r), fill=(255, 130, 0, 255), width=3)

    def _draw_ocr(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Renders OCR bounding boxes and labels for the requested board side."""
        for entry in self.state.ocr_texts:
            if str(entry.get("side", "")).upper() != side:
                continue
            x = float(entry.get("x", 0.0))
            y = float(entry.get("y", 0.0))
            w = float(entry.get("w", 0.0))
            h = float(entry.get("h", 0.0))
            text = str(entry.get("text", ""))
            selected = self.ocr_positions.get(side) == [x, y, w, h]
            color = (0, 255, 255, 255) if selected else (255, 255, 0, 190)
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3 if selected else 2)
            if text:
                draw.text((x, max(0.0, y - 12.0)), text[:24], fill=color)

    def _draw_region_ocr_preview(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Renders the temporary rectangle while the user is selecting an OCR area."""
        box = self.ocr_region_preview.get(side)
        if not box or len(box) < 4:
            return
        x, y, w, h = box[:4]
        manual = bool(self.ocr_region_start and self.ocr_region_start[3])
        color = (0, 210, 255, 255) if manual else (255, 180, 0, 255)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.rectangle((x, y, x + w, y + h), fill=(*color[:3], 35))
        draw.text((x + 3, max(0.0, y - 14.0)), "Manual OCR" if manual else "OCR", fill=color)

    def _draw_elements_manual(self, draw: ImageDraw.ImageDraw, side: str) -> None:
        """Renders outlines and labels for manually defined components based on their attached pad geometry."""
        pads_by_node = self._pads_by_node()
        for index, component in enumerate(self.state.manual_components):
            nodes = _component_pads(component)
            pads: list[Pad] = []
            seen_nodes: set[str] = set()
            for node in nodes:
                for candidate in (node, self._counterpart(node)):
                    if not candidate or candidate in seen_nodes or candidate not in pads_by_node:
                        continue
                    pad = pads_by_node[candidate]
                    if pad.side == side:
                        pads.append(pad)
                        seen_nodes.add(candidate)
            if not pads:
                continue
            color = _color_element(str(component.get("ref", "")), index)
            margin = max(8.0, max(p.radius for p in pads) + 6.0)
            min_x = min(p.x for p in pads) - margin
            max_x = max(p.x for p in pads) + margin
            min_y = min(p.y for p in pads) - margin
            max_y = max(p.y for p in pads) + margin
            if len(pads) == 1:
                pad = pads[0]
                min_x, max_x = pad.x - margin, pad.x + margin
                min_y, max_y = pad.y - margin, pad.y + margin
            if len(pads) >= 2:
                sorted_pads = sorted(pads, key=lambda p: (p.y, p.x))
                for a, b in zip(sorted_pads, sorted_pads[1:], strict=False):
                    draw.line((a.x, a.y, b.x, b.y), fill=(*color, 210), width=3)
            draw.rounded_rectangle((min_x, min_y, max_x, max_y), radius=6, outline=(*color, 255), width=3)
            ref = str(component.get("ref", "?"))
            type = _component_type(component)
            draw.text((min_x + 3, max(0.0, min_y - 14.0)), f"{ref} {type}".strip(), fill=(*color, 255))
