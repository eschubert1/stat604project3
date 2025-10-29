"""Strict basil-only segmentation tuned to suppress fridge leakage."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def _compute_texture(gray: np.ndarray) -> np.ndarray:
    """Return normalized texture map based on Laplacian energy."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = cv2.GaussianBlur(lap * lap, (0, 0), 3)
    tex_min, tex_max = float(tex.min()), float(tex.max())
    return (tex - tex_min) / (tex_max - tex_min + 1e-6)


def segment_basil_strict(
    bgr: np.ndarray,
    scale_max: int = 1200,
    grabcut_iters: int = 3,
    min_component_size: int = 50,
    return_overlay: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Segment basil while aggressively suppressing white fridge leakage.

    Parameters
    ----------
    bgr : np.ndarray
        Input BGR image.
    scale_max : int
        Longest edge resized to this value for processing (<= original keeps detail).
    grabcut_iters : int
        Number of GrabCut iterations when refining masks.
    min_component_size : int
        Minimum pixel area for connected components to keep.
    return_overlay : bool
        When True, also returns an RGB overlay with contours.

    Returns
    -------
    mask : np.ndarray
        uint8 mask (0/255) at original resolution.
    overlay : np.ndarray | None
        RGB overlay showing contours, only if return_overlay is True.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    scale = scale_max / max(height, width)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        small = cv2.resize(rgb, new_size, cv2.INTER_AREA)
    else:
        small = rgb.copy()
        scale = 1.0

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    chroma = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_green = cv2.inRange(
        hsv,
        np.array([35, 30, 20], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    mask_yellow_brown = cv2.inRange(
        hsv,
        np.array([15, 35, 20], dtype=np.uint8),
        np.array([45, 255, 210], dtype=np.uint8),
    )
    mask_leaf_raw = cv2.bitwise_or(mask_green, mask_yellow_brown)
    mask_leaf_raw = cv2.morphologyEx(mask_leaf_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_leaf_raw = cv2.morphologyEx(mask_leaf_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_red = cv2.inRange(
        hsv,
        np.array([0, 60, 40], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    ) | cv2.inRange(
        hsv,
        np.array([170, 60, 40], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    mask_white = ((s_channel < 30) & (v_channel > 185)).astype(np.uint8) * 255
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = cv2.bitwise_and(mask_leaf_raw, cv2.bitwise_not(mask_red))
    candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(mask_white))

    gray_small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    tex_norm = _compute_texture(gray_small)

    label_inputs = (candidates > 0).astype(np.uint8)
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(label_inputs, connectivity=8)
    keep = np.zeros(num_labels, dtype=np.uint8)

    if np.any(candidates > 0):
        tex_threshold = np.quantile(tex_norm[candidates > 0], 0.50)
        chroma_threshold = np.quantile(chroma[candidates > 0], 0.50)
    else:
        tex_threshold = chroma_threshold = 0.0

    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] < min_component_size:
            continue
        ys, xs = np.where(label_map == label_id)
        mean_tex = float(tex_norm[ys, xs].mean())
        mean_chroma = float(chroma[ys, xs].mean())
        if (mean_tex > tex_threshold * 0.9) or (mean_chroma > chroma_threshold * 0.9):
            keep[label_id] = 1

    pruned = np.zeros_like(candidates)
    pruned[keep[label_map] > 0] = 255

    small_h, small_w = small.shape[:2]
    trimap = np.full((small_h, small_w), 2, dtype=np.uint8)
    trimap[mask_white > 0] = 0

    border = 5
    trimap[:border, :] = 0
    trimap[-border:, :] = 0
    trimap[:, :border] = 0
    trimap[:, -border:] = 0

    strong_fg = cv2.erode(
        pruned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    trimap[strong_fg > 0] = 1
    trimap[(pruned > 0) & (trimap == 2)] = 3

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    gc_mask = trimap.copy()
    cv2.grabCut(small, gc_mask, None, bg_model, fg_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
    gc_foreground = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    final_mask_small = cv2.morphologyEx(
        gc_foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    final_mask_small = cv2.morphologyEx(
        final_mask_small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    mask = cv2.resize(final_mask_small, (width, height), cv2.INTER_NEAREST)

    overlay = None
    if return_overlay:
        overlay = rgb.copy()
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return mask, overlay


def batch_strict_masks(
    image_paths: Iterable[Path],
    return_overlay: bool = False,
    **kwargs,
) -> List[Tuple[str, np.ndarray, np.ndarray | None]]:
    """Run strict segmentation for multiple images."""
    results: List[Tuple[str, np.ndarray, np.ndarray | None]] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        mask, overlay = segment_basil_strict(bgr, return_overlay=return_overlay, **kwargs)
        results.append((path.stem, mask, overlay))
    return results
