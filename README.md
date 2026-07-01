<p align="center">
  <img src="src/schematic_generator/assets/icon.png" width="150" alt="PRISM logo">
</p>

# PRISM

**PCB Recognition and Intelligent Schematic Mapping**

PRISM reconstructs an editable KiCad schematic from aligned images of the top
and bottom sides of a PCB. It combines image processing, pad and trace
detection, OCR-assisted component recognition, connectivity analysis, manual
correction tools, and schematic optimization in one desktop application.

> PRISM is currently alpha software. Generated connectivity and component
> assignments must be reviewed before they are used for manufacturing,
> servicing, or safety-critical work.

## Features

- guided import of TOP and BOTTOM PCB images;
- automatic or manually sampled solder-mask and copper colors;
- board alignment, pad pairing, trace detection, and net reconstruction;
- OCR-assisted labels and component candidates;
- interactive correction editor for pads, traces, pairs, OCR, and components;
- text netlist and KiCad schematic export;
- optional KiCad CLI validation and schematic optimization;
- native desktop builds for Windows, macOS, and Linux.

## Requirements

- Python 3.11 or newer when running from source;
- Tk 8.6 or newer;
- Tesseract OCR for OCR features;
- KiCad with `kicad-cli` for round-trip validation and optimized previews.

PRISM searches `PATH` first and then conventional installation locations for
the current operating system. Tesseract can also be selected manually in the
interface.

On Debian or Ubuntu, install the native GUI and OpenCV runtime libraries before
running from source or building:

```bash
sudo apt-get install python3-venv python3-tk libgl1 libglib2.0-0
```

## Run From Source

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
prism
```

macOS and Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
prism
```

Windows users can alternatively run `.\run.ps1`.

## Build Native Packages

PyInstaller creates native applications and does not cross-compile. Run the
matching script on each target operating system:

```powershell
# Windows
.\scripts\build\build_windows.ps1
```

```bash
# macOS
bash scripts/build/build_macos.sh

# Linux
bash scripts/build/build_linux.sh
```

The scripts create an isolated `.venv-build` environment and use the shared
configuration in `scripts/build/package.py`.

| Platform | Output |
| --- | --- |
| Windows | `dist/PRISM/PRISM.exe` |
| macOS | `dist/PRISM.app` |
| Linux | `dist/PRISM/PRISM` |

Windows and Linux use PyInstaller's `onedir` layout. Distribute the complete
`dist/PRISM` directory, not only the executable. The macOS application is
unsigned; public releases should be code-signed and notarized by the release
owner.

## Basic Workflow

1. Select images of the top and bottom PCB sides.
2. Confirm automatic color detection or collect color samples manually.
3. Run the analysis.
4. Review detected geometry and connectivity in the correction editor.
5. Recalculate after corrections and inspect the generated netlist.
6. Open the result directory and validate the KiCad schematic.

Generated files are written under `~/PRISM/results` by default. Set the
`PRISM_OUTPUT_DIR` environment variable to use a different location. A local
`results/` directory is also excluded from version control.

## Dataset

[`Dataset/`](Dataset/README.md) contains twelve PCB examples with TOP/BOTTOM
images and reference KiCad projects. It is intended for repeatable development
and evaluation, not as a formal train/test split.

## Project Structure

```text
.
|-- Dataset/                         Test images and reference KiCad projects
|-- scripts/build/                   Native build entry points and shared build logic
|-- src/schematic_generator/
|   |-- assets/                      Application and executable icons
|   |-- correction_editor/           Manual reconstruction editor
|   |-- detection/                   Pad, trace, plane, and pair detection
|   |-- diagnostics/                 Diagnostic reports
|   |-- geometry/                    Alignment and rectification
|   |-- gui/                         Main desktop workflow
|   |-- kicad/                       KiCad serialization and routing
|   |-- netlist/                     Connectivity and component inference
|   |-- schematic/                   Validation and optimization
|   |-- platform_support.py          Operating-system integration
|   `-- __main__.py                  Application entry point
|-- pyproject.toml                  Package metadata and runtime dependencies
|-- requirements-build.txt          Build-only dependencies
`-- requirements.txt                Editable development installation
```

## Automated Builds

The GitHub Actions workflow in `.github/workflows/build.yml` runs smoke tests
and creates Windows, macOS, and Linux artifacts using the same scripts as local
builds.

## Contributing

Keep platform-independent behavior in the application modules. Operating-system
differences belong in `platform_support.py`, while packaging differences belong
in `scripts/build/package.py`. New detection or reconstruction behavior should
be checked against multiple samples from `Dataset/`.

## License

No project license has been selected yet. Add a `LICENSE` file before presenting
PRISM as open-source software or accepting external contributions.
