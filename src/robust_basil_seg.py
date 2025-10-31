"""Robust basil segmentation inspired by the CLI script provided by the user."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class BasilSegmentationMetrics:
    height: int
    width: int
    leaf_pixels: int
    coverage_pct: float
    mean_r: float
    mean_g: float
    mean_b: float


def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    mean = (mb + mg + mr) / 3.0
    b *= mean / (mb + 1e-6)
    g *= mean / (mg + 1e-6)
    r *= mean / (mr + 1e-6)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = rgb.copy()
    highlight = np.array([255, 0, 0], dtype=np.uint8)
    overlay[mask] = (alpha * highlight + (1 - alpha) * overlay[mask]).astype(np.uint8)
    return overlay


def hsv_prior(bgr_wb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    prior = (h >= 35) & (h <= 85) & (s >= 45) & (v >= 35) & (v <= 230)
    prior = prior.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    prior = cv2.morphologyEx(prior * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (prior > 127).astype(np.uint8)


def grabcut_refine(bgr_wb: np.ndarray, prior: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gc_mask = np.full(prior.shape, 2, dtype=np.uint8)
    gc_mask[prior == 1] = 3
    strong_bg = ((v > 215) & (s < 50)) | (((h < 30) | (h > 100)) & (s > 50) & (v > 140))
    gc_mask[strong_bg] = 0

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr_wb, gc_mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
    foreground = ((gc_mask == 1) | (gc_mask == 3)).astype(np.uint8)
    return (foreground & prior).astype(np.uint8)


def prune_rim_and_text(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, _, b_channel = cv2.split(lab)

    exclude = ((h < 35) | (h > 85)) | (v > 215) | (v < 60) | (b_channel > 160)
    mask = mask.astype(bool) & (~exclude)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = (cleaned > 127)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(cleaned, dtype=bool)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 700:
            continue
        component = labels[y : y + h, x : x + w] == i
        mean_sat = float(s[y : y + h, x : x + w][component].mean()) if area > 0 else 0.0
        if mean_sat >= 70:
            keep[y : y + h, x : x + w][component] = True

    return keep.astype(np.uint8)


def compute_metrics(rgb: np.ndarray, mask: np.ndarray) -> BasilSegmentationMetrics:
    height, width, _ = rgb.shape
    mask_bool = mask.astype(bool)
    leaf_pixels = int(mask_bool.sum())
    coverage = 100.0 * leaf_pixels / (height * width)
    if leaf_pixels > 0:
        mean_r = float(rgb[:, :, 0][mask_bool].mean())
        mean_g = float(rgb[:, :, 1][mask_bool].mean())
        mean_b = float(rgb[:, :, 2][mask_bool].mean())
    else:
        mean_r = mean_g = mean_b = float("nan")
    return BasilSegmentationMetrics(
        height=height,
        width=width,
        leaf_pixels=leaf_pixels,
        coverage_pct=coverage,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
    )


def segment_basil_robust(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, BasilSegmentationMetrics]:
    """Run the robust script pipeline and return mask, overlay, and metrics."""
    bgr_wb = gray_world_white_balance(bgr)
    prior = hsv_prior(bgr_wb)
    refined = grabcut_refine(bgr_wb, prior)
    final_mask = prune_rim_and_text(bgr, refined)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = overlay_mask(rgb, final_mask.astype(bool), alpha=0.55)
    metrics = compute_metrics(rgb, final_mask)
    return final_mask.astype(np.uint8), overlay, metrics


def batch_segment_robust(image_paths: Iterable[Path]) -> List[Tuple[str, np.ndarray, np.ndarray, BasilSegmentationMetrics]]:
    """Apply robust segmentation across a list of images."""
    results: List[Tuple[str, np.ndarray, np.ndarray, BasilSegmentationMetrics]] = []
    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        mask, overlay, metrics = segment_basil_robust(bgr)
        results.append((path.stem, mask, overlay, metrics))
    return results
