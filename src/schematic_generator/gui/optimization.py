from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from schematic_generator.gui.optimization_support import SchematicOptimizationSupportMixin
from schematic_generator.gui.common import (
    OPT_GENERATION_FIELD,
    OPT_IMPROVED_FIELD,
    OPT_POPULATION_FIELD,
    OPT_THREADS_FIELD,
    OPT_TIME_LIMIT_FIELD,
    ui_font,
)


class SchematicOptimizationWindow(SchematicOptimizationSupportMixin, tk.Toplevel):
    """Dialog for configuring and running KiCad schematic optimization."""
    def __init__(self, parent: tk.Misc, path: Path) -> None:
        """Initialize widget state and build the window contents."""
        super().__init__(parent)
        self.path = path
        self.title(f"Schematic optimization - {path.name}")
        self.geometry("680x730")
        self.minsize(600, 620)
        self.transient(parent)

        self.max_cpu = max(1, os.cpu_count() or 1)
        self.time_limit = tk.IntVar(value=30)
        self.thread_count = tk.IntVar(value=min(4, self.max_cpu))
        self.population_size = tk.IntVar(value=128)
        self.mutation_rate = tk.DoubleVar(value=0.25)
        self.penalty_length = tk.DoubleVar(value=1.0)
        self.penalty_crossings = tk.DoubleVar(value=40.0)
        self.penalty_wire_element = tk.DoubleVar(value=20.0)
        self.penalty_nearby_elements = tk.DoubleVar(value=5.0)
        self.reward_distances = tk.DoubleVar(value=0.2)
        self.metrics_var = tk.StringVar(value="Loading metrics...")
        self.status_var = tk.StringVar(value="Ready.")
        self.time_status_var = tk.StringVar(value="Remaining: --")
        self.start_time: float | None = None
        self.active_limit = 0

        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.chart_samples: list[float] = []
        self.button_start: ttk.Button | None = None
        self.button_cancel: ttk.Button | None = None
        self.chart: tk.Canvas | None = None

        self._build()
        self._load_metrics()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.after(100, self._handle_queue)

    def _build(self) -> None:
        """Build the schematic optimization dialog controls and score chart."""
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)

        ttk.Label(container, text=str(self.path), wraplength=460).grid(row=0, column=0, sticky="ew", padx=0, pady=10)
        ttk.Label(container, textvariable=self.time_status_var, foreground="#555").grid(row=0, column=1, sticky="ne", padx=0, pady=10)

        ttk.Label(container, text="Time limit [s]").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(container, from_=0, to=300, textvariable=self.time_limit, width=8).grid(row=1, column=1, sticky="w", padx=2, pady=2)

        ttk.Label(container, text=f"Thread count [1-{self.max_cpu}]").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(container, from_=1, to=self.max_cpu, textvariable=self.thread_count, width=8).grid(row=2, column=1, sticky="w", padx=2, pady=2)

        ttk.Label(container, text="Population size [8-512]").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(container, from_=8, to=512, increment=8, textvariable=self.population_size, width=8).grid(row=3, column=1, sticky="w", padx=2, pady=2)

        ttk.Label(container, text="Mutation rate [0-1]").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(container, textvariable=self.mutation_rate, width=10).grid(row=4, column=1, sticky="w", padx=2, pady=2)

        ttk.Separator(container).grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        ttk.Label(container, text="Evaluation function", font=ui_font(self, 10, bold=True)).grid(row=6, column=0, columnspan=2, sticky="w")

        fields = [
            ("Wire length penalty", self.penalty_length),
            ("Wire crossing penalty", self.penalty_crossings),
            ("Wire through element penalty", self.penalty_wire_element),
            ("Nearby element penalty", self.penalty_nearby_elements),
            ("Element distance reward", self.reward_distances),
        ]
        for index, (label, variable) in enumerate(fields, start=7):
            ttk.Label(container, text=label).grid(row=index, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(container, textvariable=variable, width=10).grid(row=index, column=1, sticky="w", padx=2, pady=2)

        ttk.Separator(container).grid(row=12, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        ttk.Label(container, textvariable=self.metrics_var, justify=tk.LEFT).grid(row=13, column=0, columnspan=2, sticky="ew")

        self.chart = tk.Canvas(container, height=190, background="white", highlightthickness=1, highlightbackground="#bbb")
        self.chart.grid(row=14, column=0, columnspan=2, sticky="nsew", padx=12, pady=6)
        container.rowconfigure(14, weight=1)
        self._draw_chart()

        ttk.Label(container, textvariable=self.status_var, wraplength=620).grid(row=15, column=0, columnspan=2, sticky="ew", padx=4, pady=10)
        buttons = ttk.Frame(container)
        buttons.grid(row=16, column=0, columnspan=2, sticky="e")
        self.button_start = ttk.Button(buttons, text="Start", command=self._start)
        self.button_start.pack(side=tk.LEFT)
        self.button_cancel = ttk.Button(buttons, text="Cancel", command=self._cancel)
        self.button_cancel.pack(side=tk.LEFT, padx=(8, 0))

    def _load_metrics(self) -> None:
        """Load baseline schematic metrics and disable optimization if parsing fails."""
        try:
            from schematic_generator.schematic_facade import analyze_kicad_schematic

            metrics = analyze_kicad_schematic(self.path)
            self.metrics_var.set("Before optimization:\n" + self._format_metrics(metrics))
        except Exception as exc:
            self.metrics_var.set("Could not read schematic metrics.")
            self.status_var.set(str(exc))
            if self.button_start is not None:
                self.button_start.config(state="disabled")

    def _start(self) -> None:
        """Validate optimization settings and start the optimizer worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            config = self._build_config()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=self)
            return

        from schematic_generator.schematic_facade import optimize_kicad_schematic

        self.stop_event.clear()
        self.chart_samples.clear()
        self._draw_chart()
        self.status_var.set("Optimization is running...")
        self.active_limit = int(getattr(config, OPT_TIME_LIMIT_FIELD))
        self.start_time = time.monotonic()
        self._refresh_countdown()
        if self.button_start is not None:
            self.button_start.config(state="disabled")
        if self.button_cancel is not None:
            self.button_cancel.config(text="Cancel", state="normal")

        def progress(data: object) -> None:
            """Forward optimizer progress events into the dialog queue."""
            self.queue.put(("progress", data))

        def work() -> None:
            """Run the optimizer and report completion or failure through the dialog queue."""
            try:
                result = optimize_kicad_schematic(self.path, config, progress, self.stop_event)
                self.queue.put(("finish", result))
            except Exception as exc:
                self.queue.put(("error", exc))

        self.worker_thread = threading.Thread(target=work, name="schematic-optimization", daemon=True)
        self.worker_thread.start()

    def _build_config(self) -> object:
        """Create an OptimizationConfig from the dialog values."""
        from schematic_generator.schematic_facade import OptimizationConfig

        limit = self._int_var(self.time_limit, "Time limit", 0, 300)
        thread_count = self._int_var(self.thread_count, "Thread count", 1, self.max_cpu)
        population = self._int_var(self.population_size, "Population size", 8, 512)
        mutation_rate = self._float_var(self.mutation_rate, "Mutation rate")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate: value must be in range 0-1.")
        return OptimizationConfig(
            **{
                OPT_TIME_LIMIT_FIELD: limit,
                OPT_THREADS_FIELD: thread_count,
                OPT_POPULATION_FIELD: population,
            },
            mutation_rate=mutation_rate,
            penalty_length=self._float_var(self.penalty_length, "Wire length penalty"),
            penalty_crossings=self._float_var(self.penalty_crossings, "Wire crossing penalty"),
            penalty_wire_element=self._float_var(self.penalty_wire_element, "Wire through element penalty"),
            penalty_nearby_elements=self._float_var(self.penalty_nearby_elements, "Nearby element penalty"),
            reward_distances=self._float_var(self.reward_distances, "Element distance reward"),
        )

    def _handle_queue(self) -> None:
        """Process optimizer progress, finish, and error messages on the Tk thread."""
        while True:
            try:
                kind, data = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self._handle_progress(data)
            elif kind == "finish":
                self._handle_finish(data)
            elif kind == "error":
                self._error(data)
        if self.winfo_exists():
            self.after(100, self._handle_queue)

    def _refresh_countdown(self) -> None:
        """Update the remaining-time label while optimization is running."""
        if not self.winfo_exists():
            return
        if self.worker_thread and self.worker_thread.is_alive() and self.start_time is not None:
            remaining = max(0, self.active_limit - int(time.monotonic() - self.start_time))
            self.time_status_var.set(f"Remaining: {remaining}s")
            self.after(250, self._refresh_countdown)
            return
        self.time_status_var.set("Remaining: --")

    def _handle_progress(self, data: object) -> None:
        """Add a progress score sample to the chart and status label."""
        best = float(getattr(data, "best_result"))
        self.chart_samples.append(best)
        self._draw_chart()
        generation = int(getattr(data, OPT_GENERATION_FIELD))
        self.status_var.set(f"Generation: {generation}, best score: {best:.2f}")

    def _handle_finish(self, data: object) -> None:
        """Show final optimization metrics, artifacts, and completion dialog."""
        self.start_time = None
        self.time_status_var.set("Remaining: 0s")
        if self.button_start is not None:
            self.button_start.config(state="normal")
        if self.button_cancel is not None:
            self.button_cancel.config(text="Close", state="normal")
        before = getattr(data, "metrics_before")
        by = getattr(data, "metrics_after")
        self.metrics_var.set("Before optimization:\n" + self._format_metrics(before) + "\n\nAfter optimization:\n" + self._format_metrics(by))
        improved = bool(getattr(data, OPT_IMPROVED_FIELD, False))
        changed_wire_count = int(getattr(data, "changed_wire_count", 0))
        count_elements = int(getattr(data, "changed_element_count", 0))
        svg_path = getattr(data, "svg_path", None)
        artifacts = f"{getattr(data, 'kicad_path').name} and {getattr(data, 'png_path').name}"
        if svg_path:
            artifacts += f", SVG: {svg_path.name}"
        result_description = (
            f"Schematic improved, changed elements: {count_elements}, changed wires: {changed_wire_count}."
            if improved
            else "No better variant found; unchanged copies were saved."
        )
        self.status_var.set(
            f"{result_description} Saved: {artifacts}. "
            f"Generations: {getattr(data, 'generations')}, score: {float(getattr(data, 'result')):.2f}"
        )
        if self.stop_event.is_set():
            messagebox.showinfo("Optimization interrupted", result_description, parent=self)
        elif improved:
            messagebox.showinfo("Optimization complete", result_description, parent=self)
        else:
            messagebox.showinfo("No improvement", result_description, parent=self)

    def _error(self, data: object) -> None:
        """Restore optimizer controls and show the worker exception."""
        self.start_time = None
        self.time_status_var.set("Remaining: --")
        if self.button_start is not None:
            self.button_start.config(state="normal")
        if self.button_cancel is not None:
            self.button_cancel.config(text="Close", state="normal")
        self.status_var.set(str(data))
        messagebox.showerror("Optimization error", str(data), parent=self)

    def _cancel(self) -> None:
        """Request optimizer cancellation or close the dialog when idle."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set("Canceling after the current generation...")
            if self.button_cancel is not None:
                self.button_cancel.config(state="disabled")
            return
        self.destroy()

