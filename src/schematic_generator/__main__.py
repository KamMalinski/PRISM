import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"


if __package__ in {None, ""}:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import tkinter  # noqa: F401

        print("OK")
    elif "--ui-smoke-test" in sys.argv:
        from schematic_generator.gui_facade import Application

        application = Application()
        application.update_idletasks()
        application.destroy()
        print("UI_OK")
    else:
        from schematic_generator.gui_facade import run_gui

        try:
            run_gui()
        except KeyboardInterrupt:
            raise SystemExit(130) from None
