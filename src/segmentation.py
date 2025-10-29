"""
Segmentation pipeline for isolating basil leaves and a red cup from an image.

The workflow mirrors the exploratory script provided by the user but now focuses
on computing freshness-related metrics without writing image artifacts to disk.

Run as a script (single image):
    python -m src.segmentation --image Images/P1T1D1.jpg

Run in batch mode:
    python -m src.segmentation --images-dir Images --output-dir metrics
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


@dataclass
class SegmentationResult:
    """Container for in-memory segmentation artifacts."""

    image_rgb: np.ndarray
    labels: np.ndarray
    basil_mask: np.ndarray
    cup_mask: np.ndarray
    overlay: np.ndarray
    metrics: pd.Series


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return it in RGB order."""
    ext = image_path.suffix.lower()
    if ext in {".heic", ".heif"}:
        pil_img = Image.open(image_path).convert("RGB")
        return np.array(pil_img)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is not None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    return np.array(pil_img)


def prepare_processing_image(img: np.ndarray, max_dimension: int = 1024) -> Tuple[np.ndarray, float]:
    """Downscale the input image to speed up processing while keeping quality."""
    height, width = img.shape[:2]
    scale = max_dimension / max(height, width)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        img_small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
        scale = 1.0
    return img_small, scale


def build_initial_masks(img_small: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create initial HSV masks for basil (green) and the red cup."""
    hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)

    lower_green = np.array([35, 30, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red1 = np.array([0, 70, 40], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 40], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    return mask_green, mask_red


def clean_masks(mask_green: np.ndarray, mask_red: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply morphological opening/closing to clean noisy masks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_green_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green_clean = cv2.morphologyEx(mask_green_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_red_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red_clean = cv2.morphologyEx(mask_red_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_green_clean, mask_red_clean


def refine_with_grabcut(
    img_small: np.ndarray,
    mask_green: np.ndarray,
    mask_red: np.ndarray,
    iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine masks with GrabCut using the color-based masks as priors."""
    trimap = np.full(img_small.shape[:2], 2, dtype=np.uint8)  # probable background
    trimap[mask_green > 0] = 3
    trimap[mask_red > 0] = 3

    border = 5
    trimap[:border, :] = 0
    trimap[-border:, :] = 0
    trimap[:, :border] = 0
    trimap[:, -border:] = 0

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    trimap_for_gc = trimap.copy()

    cv2.grabCut(img_small, trimap_for_gc, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)

    gc_mask = np.where(
        (trimap_for_gc == cv2.GC_FGD) | (trimap_for_gc == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    mask_green_refined = cv2.bitwise_and(mask_green, gc_mask)
    mask_red_refined = cv2.bitwise_and(mask_red, gc_mask)
    return mask_green_refined, mask_red_refined


def build_label_map(
    mask_green: np.ndarray,
    mask_red: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Upscale the masks and merge them into a single label map."""
    labels_small = np.zeros(mask_green.shape[:2], dtype=np.uint8)
    labels_small[mask_green > 0] = 1
    labels_small[(labels_small == 0) & (mask_red > 0)] = 2

    height, width = output_shape
    return cv2.resize(labels_small, (width, height), interpolation=cv2.INTER_NEAREST)


def draw_overlay(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Create contour overlays for basil and cup regions."""
    overlay = img.copy()
    basil_mask = (labels == 1).astype(np.uint8) * 255
    cup_mask = (labels == 2).astype(np.uint8) * 255

    contours_basil, _ = cv2.findContours(basil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cup, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, contours_basil, -1, (255, 255, 255), 2)
    cv2.drawContours(overlay, contours_cup, -1, (0, 0, 0), 2)
    return overlay


def compute_basil_metrics(img: np.ndarray, basil_mask: np.ndarray) -> pd.Series:
    """Calculate freshness-oriented statistics from the basil mask."""
    mask_bool = basil_mask.astype(bool)
    if mask_bool.sum() == 0:
        raise ValueError("Basil mask is empty; ensure segmentation succeeded before computing metrics.")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    rgb_vals = img[mask_bool]
    hsv_vals = hsv_img[mask_bool]
    lab_vals = lab_img[mask_bool]

    total_pixels = img.shape[0] * img.shape[1]
    basil_pixels = mask_bool.sum()
    coverage = basil_pixels / total_pixels

    mean_rgb = rgb_vals.mean(axis=0)
    std_rgb = rgb_vals.std(axis=0)
    mean_hsv = hsv_vals.mean(axis=0)
    std_hsv = hsv_vals.std(axis=0)
    mean_lab = lab_vals.mean(axis=0)

    freshness_index = (
        0.4 * (mean_rgb[1] / 255.0)
        + 0.4 * (mean_hsv[1] / 255.0)
        - 0.2 * ((mean_lab[2] - 128) / 127.0)
    )

    metrics = {
        "basil_pixels": int(basil_pixels),
        "coverage_fraction": float(coverage),
        "mean_R": float(mean_rgb[0]),
        "mean_G": float(mean_rgb[1]),
        "mean_B": float(mean_rgb[2]),
        "std_R": float(std_rgb[0]),
        "std_G": float(std_rgb[1]),
        "std_B": float(std_rgb[2]),
        "mean_hue_deg": float(mean_hsv[0] * 2.0),  # OpenCV hue is 0-179
        "mean_saturation": float(mean_hsv[1]),
        "mean_value": float(mean_hsv[2]),
        "std_saturation": float(std_hsv[1]),
        "std_value": float(std_hsv[2]),
        "mean_lab_L": float(mean_lab[0]),
        "mean_lab_a": float(mean_lab[1] - 128.0),
        "mean_lab_b": float(mean_lab[2] - 128.0),
        "freshness_index": float(freshness_index),
    }
    return pd.Series(metrics)


def run_pipeline(
    image_path: Path,
    grabcut: bool = True,
    max_dimension: int = 1024,
) -> SegmentationResult:
    """Execute the segmentation pipeline for a single image."""
    img = load_image(image_path)
    img_small, _ = prepare_processing_image(img, max_dimension=max_dimension)

    mask_green, mask_red = build_initial_masks(img_small)
    mask_green, mask_red = clean_masks(mask_green, mask_red)

    if grabcut:
        mask_green, mask_red = refine_with_grabcut(img_small, mask_green, mask_red)

    labels = build_label_map(mask_green, mask_red, output_shape=img.shape[:2])

    basil_mask = (labels == 1).astype(np.uint8)
    cup_mask = (labels == 2).astype(np.uint8)
    overlay = draw_overlay(img, labels)
    metrics = compute_basil_metrics(img, basil_mask)

    return SegmentationResult(
        image_rgb=img,
        labels=labels,
        basil_mask=basil_mask,
        cup_mask=cup_mask,
        overlay=overlay,
        metrics=metrics,
    )


def save_metrics_csv(metrics: pd.Series, image_id: str, output_path: Path) -> None:
    """Persist the metrics to a CSV file with a single row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = metrics.to_frame().T
    row.insert(0, "image_id", image_id)
    row.to_csv(output_path, index=False)


def collect_image_paths(image: Path | None, images_dir: Path | None) -> Iterable[Path]:
    """Resolve which image(s) to process based on CLI arguments."""
    if image and images_dir:
        raise ValueError("Specify either --image or --images-dir, not both.")
    if image:
        if not image.exists():
            raise FileNotFoundError(image)
        return [image]
    if images_dir:
        if not images_dir.exists():
            raise FileNotFoundError(images_dir)
        matches = sorted(images_dir.glob("P*T*D*.jpg"))
        if not matches:
            raise FileNotFoundError(
                f"No images found in {images_dir} matching pattern 'P*T*D*.jpg'."
            )
        return matches
    raise ValueError("Provide --image for a single file or --images-dir for batch mode.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basil freshness metrics from segmented images.")
    parser.add_argument("--image", type=Path, help="Path to a single labeled image (e.g., Images/P1T1D1.jpg).")
    parser.add_argument("--images-dir", type=Path, help="Directory containing labeled images to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics"),
        help="Directory where per-image CSV metric files will be written.",
    )
    parser.add_argument(
        "--no-grabcut",
        action="store_true",
        help="Disable GrabCut refinement and rely solely on color masks.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Max image dimension for processing (downscale larger images for speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.image, args.images_dir)

    for image_path in image_paths:
        result = run_pipeline(
            image_path=image_path,
            grabcut=not args.no_grabcut,
            max_dimension=args.max_dimension,
        )
        csv_path = args.output_dir / f"{image_path.stem}.csv"
        save_metrics_csv(result.metrics, image_path.stem, csv_path)
        print(f"Saved metrics for {image_path.name} -> {csv_path}")


if __name__ == "__main__":
    main()
