# PCB Test Dataset

This directory contains a curated test dataset of small PCB examples used for
evaluating PCB image analysis and schematic reconstruction workflows.

Each sample contains:

- two raster images of the PCB, one for the top side and one for the bottom
  side;
- a KiCad project directory with the reference PCB and schematic files.

## Directory Structure

The dataset is organized by sample name:

```text
Dataset/
  Cascade/
    TOP.png
    BOTTOM.png
    Cascade/
      Cascade.kicad_pcb
      Cascade.kicad_sch
      Cascade.kicad_pro
      Cascade.kicad_prl
  Connector/
    TOP.png
    BOTTOM.png
    Connector/
      Connector.kicad_pcb
      Connector.kicad_sch
      Connector.kicad_pro
      Connector.kicad_prl
  ...
```

## Samples

| Sample | Purpose / PCB feature represented | Top image | Bottom image | KiCad project folder |
| --- | --- | --- | --- | --- |
| `Cascade` | Cascaded circuit blocks and multi-stage connectivity | `TOP.png` | `BOTTOM.png` | `Cascade/` |
| `Connector` | Connector-style topology and external pin grouping | `TOP.png` | `BOTTOM.png` | `Connector/` |
| `Crossover` | Crossing or overlapping routing patterns between nets | `TOP.png` | `BOTTOM.png` | `Crossover/` |
| `Divider` | Simple divider-like circuit topology | `TOP.png` | `BOTTOM.png` | `Divider/` |
| `Elements` | Board with a larger number of components/elements | `TOP.png` | `BOTTOM.png` | `Elements/` |
| `Filter` | Filter-like circuit topology | `TOP.png` | `BOTTOM.png` | `Filter/` |
| `Floating` | Floating or weakly connected structures | `TOP.png` | `BOTTOM.png` | `Floating/` |
| `Groundplane` | Board containing a large copper groundplane/fill area | `TOP.png` | `BOTTOM.png` | `Groundplane/` |
| `Meander` | Meandered routing geometry | `TOP.png` | `BOTTOM.png` | `Meander/` |
| `Round` | Rounded or curved routing geometry | `TOP.png` | `BOTTOM.png` | `Round/` |
| `Simple` | Simple baseline board | `TOP.png` | `BOTTOM.png` | `Simple/` |
| `Transistors` | Board containing transistor-based circuitry | `TOP.png` | `BOTTOM.png` | `Transistors/` |

## File Formats

The raster images are PNG files. They should be interpreted as pixel images of
the PCB sides:

- `TOP.png` - top side image;
- `BOTTOM.png` - bottom side image.

The KiCad project directories contain:

- `*.kicad_pcb` - reference PCB layout;
- `*.kicad_sch` - reference schematic;
- `*.kicad_pro` - KiCad project metadata;
- `*.kicad_prl` - KiCad project local/session settings.

The KiCad files are intended as reference design files for validating generated
schematics, connectivity, component placement, and PCB reconstruction results.

## How to Interpret the Data

The PNG images are the primary inputs for image-processing and reconstruction
algorithms. The corresponding KiCad files are the reference data for the same
sample.

When using the dataset:

1. Treat each top-level directory as one independent PCB test case.
2. Load the top and bottom PNG images as the two observed PCB sides.
3. Use the KiCad files in the nested project directory as reference design
   data.
4. Compare generated outputs against the reference schematic or PCB project,
   depending on the evaluated task.

The dataset does not define a train/test split. If a split is needed, define it
explicitly in the experiment protocol.

## Expected Use

This dataset is suitable for:

- testing PCB side alignment;
- detecting pads, traces, holes, copper pours, and component-like structures;
- reconstructing netlists from PCB images;
- generating or validating KiCad schematic files;
- comparing algorithm behavior on boards with different routing patterns.

## Notes

- Raster image units are pixels.
- KiCad coordinate data should be interpreted according to KiCad file format
  conventions, typically in millimeters for design coordinates.
- The dataset contains PCB design examples and does not contain personal or
  human-subject data.
