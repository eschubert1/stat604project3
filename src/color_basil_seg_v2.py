"""Refined color-based basil mask with yellow-label suppression."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import cv2
import numpy as np
from sklearn.cluster import KMeans


def white_balance_simple_bgr(bgr: np.ndarray, s_thr: int = 25, v_thr: int = 185) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = (s < s_thr) & (v > v_thr)
    if white.sum() < 100:
        white = v >= np.percentile(v, 98)
    gains = bgr[white].mean(axis=0)
    gains = gains.max() / np.maximum(gains, 1e-6)
    return np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def detect_yellow_hsv(
    bgr: np.ndarray,
    h_lo: int = 15,
    h_hi: int = 40,
    s_lo: int = 70,
    v_lo: int = 120,
    dilate_ks: int = 5,
    dilate_iter: int = 1,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h >= h_lo) & (h <= h_hi) & (s > s_lo) & (v > v_lo)
    mask = yellow.astype(np.uint8) * 255
    if dilate_ks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=dilate_iter)
    return mask


def basil_mask_color_refined(
    bgr: np.ndarray,
    down_long: int = 1000,
    k_clusters: int = 3,
    hue_min: int = 30,
    hue_max: int = 95,
    sat_min: int = 40,
    val_min: int = 40,
    yellow_params: Dict[str, Any] | None = None,
    min_area: int = 120,
) -> np.ndarray:
    height, width = bgr.shape[:2]
    balanced = white_balance_simple_bgr(bgr)
    scale = min(1.0, down_long / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(balanced, (int(width * scale), int(height * scale)), cv2.INTER_AREA)
    else:
        resized = balanced
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    hue_gate = (h >= hue_min) & (h <= hue_max) & (s > sat_min) & (v > val_min)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel, b_channel = lab[..., 1], lab[..., 2]
    ab = np.stack([a_channel.ravel(), b_channel.ravel()], axis=1)

    rng = np.random.default_rng(0)
    sample_size = min(20000, ab.shape[0])
    sample = ab[rng.choice(ab.shape[0], size=sample_size, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

    _, sample_labels, centers = cv2.kmeans(
        sample.astype(np.float32), k_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    _, full_labels, _ = cv2.kmeans(
        ab.astype(np.float32),
        k_clusters,
        None,
        criteria,
        1,
        cv2.KMEANS_USE_INITIAL_LABELS,
        centers,
    )
    full_labels = full_labels.reshape(h.shape)

    best_cluster = 0
    best_score = -1.0
    for k in range(k_clusters):
        cluster_pixels = full_labels == k
        score = (cluster_pixels & hue_gate).sum() / (cluster_pixels.sum() + 1e-6)
        if score > best_score:
            best_cluster = k
            best_score = score

    mask = ((full_labels == best_cluster) & hue_gate).astype(np.uint8) * 255

    yellow_mask = detect_yellow_hsv(resized, **(yellow_params or {}))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow_mask))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros(num_labels, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            keep[label_idx] = 1
    cleaned = np.zeros_like(mask)
    cleaned[keep[label_map] > 0] = 255

    if scale < 1.0:
        cleaned = cv2.resize(cleaned, (width, height), cv2.INTER_NEAREST)
    return cleaned


def batch_color_masks_refined(
    image_paths: Iterable[Path],
    **kwargs,
) -> List[Tuple[str, np.ndarray]]:
    results: List[Tuple[str, np.ndarray]] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        mask = basil_mask_color_refined(bgr, **kwargs)
        results.append((path.stem, mask))
    return results
