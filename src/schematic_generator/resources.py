from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path


def resource_path(relative_path: str | Path) -> Path:
    """Resolve a packaged resource from PyInstaller or the source tree."""

    if getattr(sys, "frozen", False):
        base_directory = Path(sys._MEIPASS)
    else:
        base_directory = Path(__file__).resolve().parent
    return base_directory / relative_path


def apply_window_icon(window: tk.Misc) -> tk.PhotoImage | None:
    """Set the application icon and return the Tk image that must remain referenced."""

    icon_path = resource_path("assets/icon.png")
    if not icon_path.exists():
        return None

    icon_image = tk.PhotoImage(file=str(icon_path))
    window.wm_iconphoto(True, icon_image)
    return icon_image
