"""Alternative basil-only segmentation tuned for challenging lighting and wilted leaves."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def segment_basil_only(bgr: np.ndarray, use_grabcut: bool = True) -> np.ndarray:
    """Return a binary basil mask (uint8 0/255) for the provided BGR image."""
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Gentle edge-preserving denoise
    img = cv2.bilateralFilter(rgb, d=7, sigmaColor=50, sigmaSpace=7)

    # Color features
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = [img[:, :, i].astype(np.float32) for i in range(3)]

    exg = 2 * g - r - b
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    exg_n = _normalize01(exg)
    civ_n = _normalize01(-cive)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = cv2.GaussianBlur(lap * lap, (0, 0), 3)
    tex_n = _normalize01(tex)

    bg_white = ((s < 35) & (v > 175)).astype(np.uint8)

    hue_gate = ((h >= 25) & (h <= 100)).astype(np.float32)
    score = 0.45 * exg_n + 0.35 * civ_n + 0.20 * tex_n
    score *= hue_gate

    score_for_thresh = score.copy()
    score_for_thresh[bg_white.astype(bool)] = 0.0

    score_u8 = (255 * _normalize01(score_for_thresh)).astype(np.uint8)
    threshold_value, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate = (score_u8 >= threshold_value).astype(np.uint8) * 255

    candidate[bg_white > 0] = 0
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    basil = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel3, iterations=1)
    basil = cv2.morphologyEx(basil, cv2.MORPH_CLOSE, kernel7, iterations=1)

    if not use_grabcut:
        return basil

    strong_fg = (
        (score > np.quantile(score[score > 0], 0.85))
        & (tex_n > np.quantile(tex_n, 0.55))
        & (~bg_white.astype(bool))
    ).astype(np.uint8) * 255

    strong_bg = bg_white.copy() * 255
    border = 5
    strong_bg[:border, :] = 255
    strong_bg[-border:, :] = 255
    strong_bg[:, :border] = 255
    strong_bg[:, -border:] = 255

    trimap = np.full((height, width), 2, np.uint8)
    trimap[strong_bg > 0] = 0
    trimap[strong_fg > 0] = 1
    trimap[(basil > 0) & (trimap == 2)] = 3

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    gc_mask = trimap.copy()
    cv2.grabCut(rgb, gc_mask, None, bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK)
    gc = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    gc = cv2.morphologyEx(gc, cv2.MORPH_OPEN, kernel3, iterations=1)
    gc = cv2.morphologyEx(gc, cv2.MORPH_CLOSE, kernel7, iterations=1)
    return gc


def batch_segment(image_paths: Iterable[Path], use_grabcut: bool = True) -> dict[str, np.ndarray]:
    """Helper that returns masks for each image path in the iterable."""
    masks: dict[str, np.ndarray] = {}
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        masks[path.stem] = segment_basil_only(bgr, use_grabcut=use_grabcut)
    return masks


if __name__ == "__main__":
    import sys

    inputs = [Path(p) for p in sys.argv[1:]]
    if not inputs:
        inputs = [Path("Images/P1T0D0.png"), Path("Images/P1T2D4.png"), Path("Images/P2T2D3.png")]

    for img_path in inputs:
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        mask = segment_basil_only(image, use_grabcut=True)
        out_path = img_path.with_name(f"{img_path.stem}_basil_mask.png")
        cv2.imwrite(str(out_path), mask)
        print(f"saved: {out_path}")
