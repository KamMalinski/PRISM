from __future__ import annotations

from schematic_generator.netlist.elements import create_elements
from schematic_generator.netlist.io import save_devices, save_netlist
from schematic_generator.netlist.nets import build_nets

__all__ = ["build_nets", "create_elements", "save_devices", "save_netlist"]

def _patch_helper_globals() -> None:
    """Share private helpers between split modules that were originally one netlist file."""

    from schematic_generator.netlist import elements as elements
    from schematic_generator.netlist import candidate_scoring as candidate_scoring
    from schematic_generator.netlist import connectivity_helpers as connectivity_helpers
    from schematic_generator.netlist import footprint_candidates as footprint_candidates
    from schematic_generator.netlist import io as io
    from schematic_generator.netlist import nearby_devices as nearby_devices
    from schematic_generator.netlist import nets as nets
    from schematic_generator.netlist import ocr_silkscreen as ocr_silkscreen
    from schematic_generator.netlist import pad_rows_ocr as pad_rows_ocr
    from schematic_generator.netlist import silkscreen_scoring as silkscreen_scoring
    from schematic_generator.netlist import solver_candidates as solver_candidates

    modules = [
        elements,
        io,
        nets,
        solver_candidates,
        candidate_scoring,
        footprint_candidates,
        pad_rows_ocr,
        nearby_devices,
        ocr_silkscreen,
        silkscreen_scoring,
        connectivity_helpers,
    ]
    merged = {}
    for module in modules:
        merged.update({name: value for name, value in vars(module).items() if not name.startswith("__")})
    for module in modules:
        vars(module).update(merged)


_patch_helper_globals()
