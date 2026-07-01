from __future__ import annotations

from schematic_generator.detection.common import LogFn, _distance, _log, _pad_identifier
from schematic_generator.models import HolePair, Pad

def find_pairs_holes(
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    log: LogFn | None = None,
) -> list[HolePair]:
    """Pairs nearest TOP/BOTTOM pads after geometric normalization."""

    _log(log, "Hole pairs: nearest-neighbor method with tolerance max(12px, 3*radius).")
    pairs: list[HolePair] = []
    used_bottom_ids: set[str] = set()
    for pad_top in top_pads:
        candidates = [
            (pad_bottom, _distance(pad_top, pad_bottom))
            for pad_bottom in bottom_pads
            if _pad_identifier(pad_bottom) not in used_bottom_ids
        ]
        if not candidates:
            continue
        pad_bottom, distance = min(candidates, key=lambda entry: entry[1])
        radius_max = max(pad_top.radius, pad_bottom.radius)
        radius_min = min(pad_top.radius, pad_bottom.radius)
        tolerance = max(12.0, 3.0 * radius_min, 1.8 * radius_max)
        if distance <= tolerance:
            used_bottom_ids.add(_pad_identifier(pad_bottom))
            confidence = max(0.05, 1.0 - distance / tolerance)
            pairs.append(HolePair(pad_top.node, pad_bottom.node, distance, confidence))
    _log(log, f"Hole pairs: found {len(pairs)} pairs.")
    return pairs
