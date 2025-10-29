"""Pure color-based basil segmentation using white balance and KMeans clustering."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def white_balance_simple(rgb: np.ndarray) -> np.ndarray:
    """Simple per-channel white balance based on low-saturation, high-value pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white = (saturation < 25) & (value > 180)
    if white.sum() < 100:
        threshold = np.percentile(value, 98)
        white = value >= threshold
    gains = rgb[white].mean(axis=0)
    gains = gains.max() / np.maximum(gains, 1e-6)
    balanced = np.clip(rgb.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return balanced


import cv2, numpy as np
from sklearn.cluster import KMeans

def basil_mask_color_only(
    bgr: np.ndarray,
    down_long: int = 900,
    k_clusters: int = 3,
    hue_min: int = 20,
    hue_max: int = 110,
    sat_min: int = 25,
    # NEW: fridge white thresholds
    white_s: int = 25,
    white_v: int = 185,
    # NEW: cluster sanity + speed
    min_leaf_overlap: float = 0.15,   # if best cluster < 15% leafy, fall back
    sample_px: int = 12000,           # subsample pixels for KMeans
    # NEW: post-filter
    min_area: int = 120,              # drop tiny components
    bar_width_frac: float = 0.50,     # remove very wide, thin bars (fridge shelf)
    bar_height_frac: float = 0.14,
) -> np.ndarray:
    """Return a basil mask (0/255) using color clustering (Lab a,b)."""

    H, W = bgr.shape[:2]
    scale = min(1.0, down_long / max(H, W))
    small = cv2.resize(bgr, (int(W*scale), int(H*scale)), cv2.INTER_AREA) if scale < 1.0 else bgr
    rgb  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # --- white balance on fridge whites
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:,:,1], hsv[:,:,2]
    white = (s < white_s) & (v > white_v)
    if white.sum() < 100:
        white = v >= np.percentile(v, 98)
    gains = rgb[white].mean(axis=0)
    gains = gains.max() / np.maximum(gains, 1e-6)
    rgb   = np.clip(rgb.astype(np.float32)*gains, 0, 255).astype(np.uint8)

    # --- color gates
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    white_bg = (s < white_s) & (v > white_v)
    hue_gate = (h >= hue_min) & (h <= hue_max) & (s > sat_min)
    valid    = ~white_bg

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a = lab[:,:,1].astype(np.float32); b = lab[:,:,2].astype(np.float32)

    # --- subsample valid pixels for KMeans (speed)
    ys, xs = np.where(valid)
    if len(xs) < 500:
        mask_small = (hue_gate & ~white_bg).astype(np.uint8)*255
    else:
        if len(xs) > sample_px:
            idx = np.random.choice(len(xs), size=sample_px, replace=False)
            X = np.stack([a[ys[idx], xs[idx]], b[ys[idx], xs[idx]]], axis=1)
            # map-back indices later
            map_valid = None
        else:
            X = np.stack([a[valid], b[valid]], axis=1)
            map_valid = np.where(valid.ravel())[0]

        km = KMeans(n_clusters=k_clusters, n_init=5, random_state=0)
        km.fit(X)

        # assign ALL valid pixels to nearest center
        centers = km.cluster_centers_.astype(np.float32)
        ab_all  = np.stack([a[valid], b[valid]], axis=1)
        dists   = np.linalg.norm(ab_all[:,None,:] - centers[None,:,:], axis=2)
        lbl_all = dists.argmin(axis=1)

        labels = np.full(valid.shape, -1, dtype=np.int32)
        labels[valid] = lbl_all

        # pick cluster with max overlap with leaf hue gate
        best, best_score = 0, -1.0
        for c in range(k_clusters):
            cluster = (labels == c)
            if cluster.sum() == 0: 
                continue
            score = (cluster & hue_gate).sum() / (cluster.sum() + 1e-6)
            if score > best_score:
                best, best_score = c, score

        # Fallback if KMeans picked a nonsense cluster
        if best_score < min_leaf_overlap:
            mask_small = (hue_gate & ~white_bg).astype(np.uint8)*255
        else:
            mask_small = ((labels == best) & ~white_bg).astype(np.uint8)*255

    # --- morphology
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN,  k3, 1)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, k7, 1)

    # --- component post-filter (drop fridge bar + specks)
    Hs, Ws = mask_small.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_small>0).astype(np.uint8), 8)
    keep = np.zeros(num, np.uint8)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area: 
            continue
        # remove long, thin horizontal bars
        if (w > bar_width_frac*Ws) and (h < bar_height_frac*Hs):
            continue
        keep[i] = 1
    cleaned = np.zeros_like(mask_small)
    cleaned[keep[labels] > 0] = 255

    # --- upscale to original
    return cv2.resize(cleaned, (W, H), cv2.INTER_NEAREST) if scale < 1.0 else cleaned


def batch_color_masks(
    image_paths: Iterable[Path],
    **kwargs,
) -> List[Tuple[str, np.ndarray]]:
    """Run color-only segmentation across multiple images."""
    results: List[Tuple[str, np.ndarray]] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        mask = basil_mask_color_only(bgr, **kwargs)
        results.append((path.stem, mask))
    return results
