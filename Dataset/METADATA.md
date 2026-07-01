# Metadata

## Dataset Description

This dataset contains 12 PCB test samples prepared for evaluating algorithms
that reconstruct PCB structure, connectivity, and schematic data from images of
the top and bottom board sides.

Each sample consists of:

- a top-side PNG image;
- a bottom-side PNG image;
- a nested KiCad project directory containing the reference PCB and schematic
  design files.

The dataset is intended for testing and benchmarking PCB image-processing,
netlist-reconstruction, and KiCad schematic-generation workflows.

## Dataset Inventory

| Sample | Top image | Top dimensions [px] | Bottom image | Bottom dimensions [px] | Reference KiCad files |
| --- | --- | ---: | --- | ---: | --- |
| `Cascade` | `TOP.png` | 765 x 884 | `BOTTOM.png` | 765 x 884 | `Cascade/*.kicad_*` |
| `Connector` | `TOP.png` | 618 x 583 | `BOTTOM.png` | 618 x 583 | `Connector/*.kicad_*` |
| `Crossover` | `TOP.png` | 1068 x 572 | `BOTTOM.png` | 1068 x 572 | `Crossover/*.kicad_*` |
| `Divider` | `TOP.png` | 746 x 233 | `BOTTOM.png` | 803 x 239 | `Divider/*.kicad_*` |
| `Elements` | `TOP.png` | 2364 x 900 | `BOTTOM.png` | 2364 x 900 | `Elements/*.kicad_*` |
| `Filter` | `TOP.png` | 803 x 744 | `BOTTOM.png` | 803 x 744 | `Filter/*.kicad_*` |
| `Floating` | `TOP.png` | 1068 x 572 | `BOTTOM.png` | 1068 x 572 | `Floating/*.kicad_*` |
| `Groundplane` | `TOP.png` | 1068 x 572 | `BOTTOM.png` | 1416 x 660 | `Groundplane/*.kicad_*` |
| `Meander` | `TOP.png` | 963 x 884 | `BOTTOM.png` | 963 x 884 | `Meander/*.kicad_*` |
| `Round` | `TOP.png` | 803 x 744 | `BOTTOM.png` | 803 x 744 | `Round/*.kicad_*` |
| `Simple` | `TOP.png` | 2364 x 900 | `BOTTOM.png` | 2364 x 900 | `Simple/*.kicad_*` |
| `Transistors` | `TOP.png` | 709 x 900 | `BOTTOM.png` | 727 x 900 | `Transistors/*.kicad_*` |

## Variables and Definitions

| Variable / field | Definition | Type | Unit / values |
| --- | --- | --- | --- |
| `sample_name` | Name of the top-level dataset directory and PCB test case. | String | One of: `Cascade`, `Connector`, `Crossover`, `Divider`, `Elements`, `Filter`, `Floating`, `Groundplane`, `Meander`, `Round`, `Simple`, `Transistors` |
| `top_image` | Raster image representing the top side of the PCB. | PNG image file | Pixel image; filename is `TOP.png` |
| `bottom_image` | Raster image representing the bottom side of the PCB. | PNG image file | Pixel image; filename is `BOTTOM.png` |
| `image_width` | Horizontal size of a raster image. | Integer | Pixels |
| `image_height` | Vertical size of a raster image. | Integer | Pixels |
| `pcb_file` | KiCad PCB layout reference file for the sample. | KiCad PCB file | `*.kicad_pcb`; design coordinates follow KiCad units |
| `schematic_file` | KiCad schematic reference file for the sample. | KiCad schematic file | `*.kicad_sch`; design coordinates follow KiCad units |
| `project_file` | KiCad project metadata file. | KiCad project file | `*.kicad_pro` |
| `local_project_file` | KiCad project local/session settings file. | KiCad local project file | `*.kicad_prl` |
| `board_side` | PCB side represented by an image. | Categorical | `TOP`, `BOTTOM` |
| `feature_category` | Main test feature represented by a sample. | Categorical description | Routing or circuit class, e.g. groundplane, meander, connector, elements |

## Units of Measurement

- Raster image dimensions are measured in pixels (`px`).
- Pixel coordinates, when derived from PNG images, should be interpreted in
  image-coordinate space with the origin at the top-left corner.
- KiCad design coordinates should be interpreted according to KiCad file format
  conventions. For PCB and schematic geometry, coordinates are typically stored
  in millimeters.
- No physical image scale, DPI calibration, or pixel-to-millimeter conversion
  factor is provided in the dataset.

## Contextual Details

The samples represent different PCB characteristics relevant to automated
analysis:

- `Cascade` includes a long chain of interconnected components
- `Connector` emphasizes connector-style grouping.
- `Crossover` emphasizes crossing or overlapping routing patterns.
- `Divider` has a PCB color similar in hue to the background, making it more difficult to isolate
- `Elements` contains a larger set of component structures.
- `Filter` has different colors on both sides of the PCB
- `Floating` includes floating or weakly connected structures
- `Groundplane` includes a large copper groundplane or fill region.
- `Meander` includes meandered traces.
- `Round` includes rounded routing geometry.
- `Simple` provides a baseline board.
- `Transistors` includes vias of different sizes and different pad configurations for transistors

The PNG images should be used as algorithm inputs. The KiCad files should be
used as reference design data for interpretation, validation, and comparison.

## Data Quality and Limitations

- Image dimensions differ between samples and sometimes between board sides.
  Processing pipelines should not assume a fixed image size.
- The dataset does not provide manual annotations as separate label files.
  Reference information is contained in the KiCad project files.
- The dataset does not include physical calibration metadata for converting
  pixels to real-world distances.

## File Count Summary

- Number of PCB samples: 12.
- Number of PNG side images: 24.
- Number of KiCad project directories: 12.
- Reference files per KiCad project: `*.kicad_pcb`, `*.kicad_sch`,
  `*.kicad_pro`, and `*.kicad_prl`.
