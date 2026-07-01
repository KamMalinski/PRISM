from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
ASSET_DIRECTORY = SOURCE_DIRECTORY / "schematic_generator" / "assets"
PLATFORM_NAMES = {
    "windows": "windows",
    "darwin": "macos",
    "linux": "linux",
}


def main() -> None:
    """Package PRISM with one shared PyInstaller configuration."""

    parser = argparse.ArgumentParser(description="Run the shared PRISM PyInstaller build.")
    parser.add_argument("--platform", choices=("windows", "macos", "linux"), required=True)
    arguments = parser.parse_args()
    current_platform = PLATFORM_NAMES.get(platform.system().lower())
    if current_platform != arguments.platform:
        raise SystemExit(
            f"Cannot create a {arguments.platform} package on "
            f"{current_platform or platform.system()}; PyInstaller builds are native."
        )

    work_directory = PROJECT_ROOT / "build" / arguments.platform
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "PRISM",
        "--paths",
        str(SOURCE_DIRECTORY),
        "--workpath",
        str(work_directory),
        "--specpath",
        str(work_directory),
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--hidden-import",
        "tkinter",
        "--hidden-import",
        "_tkinter",
        "--collect-submodules",
        "tkinter",
        "--add-data",
        f"{ASSET_DIRECTORY / 'icon.png'}{os.pathsep}assets",
    ]
    if arguments.platform == "windows":
        command.extend(["--icon", str(ASSET_DIRECTORY / "icon.ico")])
    elif arguments.platform == "macos":
        command.extend(["--icon", str(ASSET_DIRECTORY / "icon.png")])
    command.append(str(SOURCE_DIRECTORY / "schematic_generator" / "__main__.py"))

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"PRISM build completed: {_output_path(arguments.platform)}")


def _output_path(target_platform: str) -> Path:
    """Return the primary distributable path generated for a target platform."""

    if target_platform == "windows":
        return PROJECT_ROOT / "dist" / "PRISM" / "PRISM.exe"
    if target_platform == "macos":
        return PROJECT_ROOT / "dist" / "PRISM.app"
    return PROJECT_ROOT / "dist" / "PRISM" / "PRISM"


if __name__ == "__main__":
    main()
