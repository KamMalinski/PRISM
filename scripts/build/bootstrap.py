from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_ENVIRONMENT = PROJECT_ROOT / ".venv-build"
PLATFORM_NAMES = {
    "windows": "windows",
    "darwin": "macos",
    "linux": "linux",
}


def main() -> None:
    """Create a clean build environment when needed and run the shared packager."""

    parser = argparse.ArgumentParser(description="Build a native PRISM package.")
    parser.add_argument("--platform", choices=("windows", "macos", "linux"), required=True)
    arguments = parser.parse_args()
    current_platform = PLATFORM_NAMES.get(platform.system().lower())
    if current_platform != arguments.platform:
        raise SystemExit(
            f"This script targets {arguments.platform}, but the current platform is "
            f"{current_platform or platform.system()}."
        )

    python = _build_environment_python()
    if not _python_is_usable(python):
        if BUILD_ENVIRONMENT.exists():
            shutil.rmtree(BUILD_ENVIRONMENT)
        venv.EnvBuilder(with_pip=True).create(BUILD_ENVIRONMENT)

    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements-build.txt")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [str(python), str(Path(__file__).with_name("package.py")), "--platform", arguments.platform],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _build_environment_python() -> Path:
    """Return the interpreter path used by virtual environments on this platform."""

    if platform.system().lower() == "windows":
        return BUILD_ENVIRONMENT / "Scripts" / "python.exe"
    return BUILD_ENVIRONMENT / "bin" / "python"


def _python_is_usable(python: Path) -> bool:
    """Check that an existing build environment still references a working interpreter."""

    if not python.is_file():
        return False
    result = subprocess.run(
        [str(python), "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


if __name__ == "__main__":
    main()
