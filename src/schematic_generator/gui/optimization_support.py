from __future__ import annotations

import tkinter as tk


class SchematicOptimizationSupportMixin:
    """Provide chart drawing and value validation helpers for the optimization dialog."""

    def _draw_chart(self) -> None:
        """Render the rolling optimization score chart on the canvas."""
        if self.chart is None:
            return
        self.chart.delete("all")
        width = max(1, self.chart.winfo_width() or 560)
        height = max(1, self.chart.winfo_height() or 190)
        margin = 28
        self.chart.create_rectangle(margin, 12, width - 10, height - margin, outline="#d0d0d0")
        if not self.chart_samples:
            self.chart.create_text(width // 2, height // 2, text="No data", fill="#666")
            return
        values = self.chart_samples[-500:]
        min_y = min(values)
        max_y = max(values)
        if abs(max_y - min_y) < 0.001:
            max_y += 1.0
            min_y -= 1.0
        points: list[float] = []
        for index, value in enumerate(values):
            x = margin + index * (width - margin - 10) / max(1, len(values) - 1)
            y = (height - margin) - (value - min_y) * (height - margin - 12) / (max_y - min_y)
            points.extend([x, y])
        if len(points) >= 4:
            self.chart.create_line(*points, fill="#2563eb", width=2)
        last_x, last_y = points[-2], points[-1]
        self.chart.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill="#2563eb", outline="")
        self.chart.create_text(margin, 6, anchor="nw", text=f"{max_y:.1f}", fill="#555")
        self.chart.create_text(margin, height - margin + 4, anchor="nw", text=f"{min_y:.1f}", fill="#555")

    def _format_metrics(metrics: object) -> str:
        """Format schematic metrics into the multiline text shown in the dialog."""
        return (
            f"Elements: {getattr(metrics, 'element_count')}, wires: {getattr(metrics, 'wire_count')}\n"
            f"Wire length: {float(getattr(metrics, 'wire_length')):.3f}\n"
            f"Wire crossings: {getattr(metrics, 'crossing_count')}\n"
            f"Minimum element distance: {float(getattr(metrics, 'min_element_distance')):.3f}\n"
            f"Average nearest distance: {float(getattr(metrics, 'average_nearest_distance')):.3f}\n"
            f"Wire-element conflicts: {getattr(metrics, 'wire_element_conflicts')}\n"
            f"Wire bends: {getattr(metrics, 'bend_count', 0)}\n"
            f"Wire backtracks: {getattr(metrics, 'backtrack_count', 0)}"
        )

    def _int_var(variable: tk.IntVar, name: str, minimum: int, maximum: int) -> int:
        """Read and validate an integer Tk variable within a bounded range."""
        try:
            value = int(variable.get())
        except Exception as exc:
            raise ValueError(f"{name}: provide an integer.") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name}: value must be in range {minimum}-{maximum}.")
        return value

    def _float_var(variable: tk.DoubleVar, name: str) -> float:
        """Read and validate a floating-point Tk variable."""
        try:
            return float(variable.get())
        except Exception as exc:
            raise ValueError(f"{name}: provide a number.") from exc
