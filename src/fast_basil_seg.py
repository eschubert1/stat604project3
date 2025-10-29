"""Fast basil-only segmentation tuned for speed with adjustable robustness params."""
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


def fast_basil_mask(
    bgr: np.ndarray,
    down_long: int = 640,
    s_white: int = 35,
    v_white: int = 170,
    perc: float = 0.80,
    open_sz: int = 3,
    close_sz: int = 7,
) -> np.ndarray:
    """Return a basil-only mask (uint8 0/255) at the original resolution."""
    height, width = bgr.shape[:2]
    scale = min(1.0, down_long / max(height, width))
    if scale < 1.0:
        small = cv2.resize(bgr, (int(width * scale), int(height * scale)), cv2.INTER_AREA)
    else:
        small = bgr

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]

    bg_white = (s < s_white) & (v > v_white)

    exg = 2 * g - r - b
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    exg_n = _normalize01(exg)
    civ_n = _normalize01(-cive)
    score = np.maximum(exg_n, civ_n)

    hue_gate = ((h >= 25) & (h <= 100)).astype(np.float32)
    score *= hue_gate

    score_for_thr = score.copy()
    score_for_thr[bg_white] = 0.0
    thr = float(np.quantile(score_for_thr, perc))
    candidate = (score >= thr).astype(np.uint8) * 255
    candidate[bg_white] = 0

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_sz, open_sz))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
    mask = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    if scale < 1.0:
        mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
    return mask


def batch_fast_masks(image_paths: Iterable[Path], **kwargs) -> dict[str, np.ndarray]:
    """Compute fast basil masks for multiple images."""
    masks: dict[str, np.ndarray] = {}
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        masks[path.stem] = fast_basil_mask(bgr, **kwargs)
    return masks


if __name__ == "__main__":
    import sys

    inputs = [Path(p) for p in sys.argv[1:]] or [
        Path("Images/IMG_5598.png"),
        Path("Images/P1T2D4.png"),
        Path("Images/P2T2D3.png"),
    ]

    for img_path in inputs:
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        mask = fast_basil_mask(image, down_long=640)
        out_path = img_path.with_name(f"{img_path.stem}_basil_fast_mask.png")
        cv2.imwrite(str(out_path), mask)
        print(f"saved: {out_path}")
