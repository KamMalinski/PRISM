from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


SYSTEM = platform.system().lower()


def default_output_directory() -> Path:
    """Return a user-writable result directory with an optional environment override."""

    configured_directory = os.environ.get("PRISM_OUTPUT_DIR")
    if configured_directory:
        return Path(configured_directory).expanduser().resolve()
    return Path.home() / "PRISM" / "results"


def open_in_file_manager(path: str | Path) -> None:
    """Open a directory with the native file manager for the current platform."""

    directory = Path(path).resolve()
    if SYSTEM == "windows":
        os.startfile(directory)
        return

    command = "open" if SYSTEM == "darwin" else "xdg-open"
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"Could not find the '{command}' command required to open folders.")
    subprocess.Popen(
        [executable, str(directory)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def find_tesseract() -> Path | None:
    """Locate the Tesseract executable using PATH and platform-specific install roots."""

    return _find_executable("tesseract", _tesseract_candidates())


def find_kicad_cli() -> Path | None:
    """Locate KiCad CLI using PATH and platform-specific install roots."""

    return _find_executable("kicad-cli", _kicad_candidates())


def executable_file_types(display_name: str, executable_name: str) -> list[tuple[str, str]]:
    """Return native file-dialog filters for selecting a command-line executable."""

    filename = f"{executable_name}.exe" if SYSTEM == "windows" else executable_name
    filters = [(display_name, filename)]
    if SYSTEM == "windows":
        filters.append(("Executable files", "*.exe"))
    filters.append(("All files", "*.*"))
    return filters


def _find_executable(command: str, candidates: list[Path]) -> Path | None:
    """Return a PATH command first, then the first existing platform candidate."""

    from_path = shutil.which(command)
    if from_path:
        return Path(from_path)
    return next((path for path in candidates if path.is_file()), None)


def _tesseract_candidates() -> list[Path]:
    """Build conventional Tesseract locations for the active operating system."""

    if SYSTEM == "windows":
        candidates: list[Path] = []
        for variable, suffix in (
            ("ProgramFiles", ("Tesseract-OCR", "tesseract.exe")),
            ("ProgramFiles(x86)", ("Tesseract-OCR", "tesseract.exe")),
            ("LOCALAPPDATA", ("Programs", "Tesseract-OCR", "tesseract.exe")),
        ):
            base_directory = os.environ.get(variable)
            if base_directory:
                candidates.append(Path(base_directory).joinpath(*suffix))
        return candidates
    if SYSTEM == "darwin":
        return [
            Path("/opt/homebrew/bin/tesseract"),
            Path("/usr/local/bin/tesseract"),
            Path("/opt/local/bin/tesseract"),
        ]
    return [
        Path("/usr/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
        Path("/snap/bin/tesseract"),
    ]


def _kicad_candidates() -> list[Path]:
    """Build conventional KiCad CLI locations for the active operating system."""

    if SYSTEM == "windows":
        candidates: list[Path] = []
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            program_files = os.environ.get(variable)
            if program_files:
                candidates.extend(Path(program_files).glob("KiCad/*/bin/kicad-cli.exe"))
        return sorted(candidates, key=lambda path: path.as_posix(), reverse=True)
    if SYSTEM == "darwin":
        applications = Path("/Applications")
        candidates = [
            applications / "KiCad" / "KiCad.app" / "Contents" / "MacOS" / "kicad-cli",
            applications / "KiCad.app" / "Contents" / "MacOS" / "kicad-cli",
        ]
        candidates.extend(applications.glob("KiCad*/KiCad.app/Contents/MacOS/kicad-cli"))
        return candidates
    return [
        Path("/usr/bin/kicad-cli"),
        Path("/usr/local/bin/kicad-cli"),
        Path("/snap/bin/kicad-cli"),
    ]
