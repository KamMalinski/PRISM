from __future__ import annotations

import cv2
import numpy as np

from schematic_generator.mask_contacts import components_traces, label_traces_near_pad
from schematic_generator.models import Pad, HolePair


def draw_preview(
    image: np.ndarray,
    mask: np.ndarray,
    pads: list[Pad],
    pairs: list[HolePair],
    side: str = "TOP",
) -> np.ndarray:
    """Draw the trace mask, detected pads, and paired through-holes for one PCB side."""

    result = image.copy()
    color_mask = np.zeros_like(result)
    color_mask[:, :, 1] = (mask > 0).astype(np.uint8) * 180
    result = cv2.addWeighted(result, 0.82, color_mask, 0.45, 0)

    pads_by_node = {pad.node: pad for pad in pads}
    for pad in pads:
        point = (int(round(pad.x)), int(round(pad.y)))
        radius = int(round(max(3, pad.radius)))
        cv2.circle(result, point, radius, (0, 220, 255), 2)
        label = pad.name or pad.net or pad.identifier
        if pad.name and pad.net:
            label = f"{pad.name}:{pad.net}"
        cv2.putText(result, label, (point[0] + 4, point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

    for pair in pairs:
        node = pair.pad_bottom if side == "BOTTOM" else pair.pad_top
        pad = pads_by_node.get(node)
        if not pad:
            continue
        point = (int(round(pad.x)), int(round(pad.y)))
        cv2.circle(result, point, int(round(max(5, pad.radius + 3))), (255, 0, 255), 1)

    return result


def draw_thick_traces(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Overlay detected traces as thick highlighted contours on the source image."""

    result = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 6, lineType=cv2.LINE_AA)

    fill = np.zeros_like(result)
    fill[mask > 0] = color
    result = cv2.addWeighted(result, 0.88, fill, 0.35, 0)
    return result


def draw_connections_colored(
    image: np.ndarray,
    mask: np.ndarray,
    pads: list[Pad],
) -> np.ndarray:
    """Color each detected connection so traces and pads share the same net color."""

    result = image.copy()
    count, labels, _stats, _centroids = components_traces(mask)
    components_by_net: dict[str, list[int]] = {}
    for pad in pads:
        if not pad.net or pad.net == "NET?":
            continue
        label = _label_under_pad(labels, pad)
        if label and label < count:
            components_by_net.setdefault(pad.net, []).append(label)

    fill = np.zeros_like(result)
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    for net, components in sorted(components_by_net.items()):
        color = _color_net(net)
        for label in set(components):
            pixels = labels == label
            fill[pixels] = color
            contour_mask[pixels] = 255

    result = cv2.addWeighted(result, 0.72, fill, 0.75, 0)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    for pad in pads:
        if not pad.net or pad.net == "NET?":
            continue
        color = _color_net(pad.net)
        point = (int(round(pad.x)), int(round(pad.y)))
        radius = int(round(max(4, pad.radius + 2)))
        cv2.circle(result, point, radius, color, 3, lineType=cv2.LINE_AA)
        cv2.circle(result, point, max(2, radius // 3), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            result,
            pad.net,
            (point[0] + radius + 3, point[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def overlay_bottom_on_top(top_image: np.ndarray, bottom_image: np.ndarray, alpha_bottom: float = 0.5) -> np.ndarray:
    """Create an alignment preview with the BOTTOM image blended over the TOP image."""

    if bottom_image.shape[:2] != top_image.shape[:2]:
        bottom_image = cv2.resize(
            bottom_image,
            (top_image.shape[1], top_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    alpha_bottom = max(0.0, min(1.0, alpha_bottom))
    return cv2.addWeighted(top_image, 1.0 - alpha_bottom, bottom_image, alpha_bottom, 0)


def draw_alignment_holes(
    top_image: np.ndarray,
    bottom_image: np.ndarray,
    top_pads: list[Pad],
    bottom_pads: list[Pad],
    pairs: list[HolePair],
) -> np.ndarray:
    """Draw a TOP/BOTTOM overlay with lines between paired through-holes."""

    result = overlay_bottom_on_top(top_image, bottom_image, 0.5)
    top_by_node = {pad.node: pad for pad in top_pads}
    bottom_by_node = {pad.node: pad for pad in bottom_pads}

    for pair in pairs:
        top = top_by_node.get(pair.pad_top)
        bottom = bottom_by_node.get(pair.pad_bottom)
        if not top or not bottom:
            continue
        p_top = (int(round(top.x)), int(round(top.y)))
        p_bottom = (int(round(bottom.x)), int(round(bottom.y)))
        color = (0, 255, 0) if pair.distance <= max(8.0, top.radius * 1.5) else (0, 180, 255)
        cv2.line(result, p_top, p_bottom, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(result, p_top, int(round(max(4, top.radius))), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(result, p_bottom, int(round(max(4, bottom.radius))), (255, 180, 0), 2, lineType=cv2.LINE_AA)

    return result


def _label_under_pad(labels: np.ndarray, pad: Pad) -> int:
    """Return the trace-component label nearest to the pad center."""

    return label_traces_near_pad(labels, pad)


def _color_net(net: str) -> tuple[int, int, int]:
    """Choose a stable BGR preview color from the net name."""

    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (0, 0, 128),
    ]
    index = sum(ord(ch) for ch in net) % len(palette)
    r, g, b = palette[index]
    return b, g, r
