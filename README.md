# STATS 604 Project 3, Fall 2025

## Project Structure:
The project is organized with the following structure:

- lab-notebook.md : a markdown file which documents how the herb experiment was conducted
- pre-analysis_plan.md : a markdown file which documents the pre-analysis plan
- data/
    - raw_data/ : directory for all raw data files
    - processed_data/ : directory for all cleaned data
- figures/ : directory for all plots and generated figures
- results/ : directory for all results - either intermediate or final
- src/ : directory containing all code files

Additionally, all code in the src/ directory should only require files from
data/ or results/ and any intermediate output should be saved in one of those
two locations. This will make it easier to specify a dependency structure
in the Makefile to more efficiently produce the final report.

# Basil & Cup Segmentation Metrics

Compute basil freshness metrics from color-driven segmentation of high-resolution kitchen photos. The pipeline still segments basil vs. red cup using HSV thresholds and optional GrabCut refinement, but now only emits per-image CSV metric reports.

## Project Layout

- `src/segmentation.py` &mdash; reusable pipeline and CLI entry point that returns in-memory artifacts and writes metrics CSVs.
- `notebooks/segmentation_demo.ipynb` &mdash; interactive walkthrough demonstrating segmentation, visualization, and metric export.
- `notebooks/basil_mask_probe.ipynb`, `notebooks/fast_basil_mask_probe.ipynb`, `notebooks/strict_basil_mask_probe.ipynb`, `notebooks/color_basil_mask_probe.ipynb`, `notebooks/processed_color_basil_mask_probe.ipynb`, `notebooks/processed_color_basil_mask_probe2.ipynb`, `notebooks/processed_robust_basil_mask_probe.ipynb` &mdash; optional notebooks to compare alternative basil-only segmentation strategies (including processed data top views).
- `requirements.txt` &mdash; Python dependencies for the project.
- `metrics/` &mdash; default directory for generated CSV reports (ignored by git).

## Quick Start

1. **Create / activate the virtual environment (already created as `.venv/`):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. **Run the pipeline on one image:**
   ```powershell
   python -m src.segmentation --image Images\P1T1D1.jpg --output-dir metrics
   ```
   The script writes a single-row CSV (e.g., `metrics\P1T1D1.csv`) containing coverage, RGB/HSV/Lab statistics, and the freshness index.

4. **Batch process a directory of labeled images:**
   ```powershell
   python -m src.segmentation --images-dir Images --output-dir metrics
   ```
   Files matching `P*T*D*.jpg` are processed and each receives a matching CSV in the chosen output directory.

## Notebook

Launch Jupyter and open `notebooks/segmentation_demo.ipynb` to explore the pipeline interactively:

```powershell
jupyter notebook
```

The notebook uses the same helper functions defined in `src/segmentation.py`, displays overlays for inspection, and demonstrates how to write the metrics CSV for a chosen image.

## Notes

- Update the image paths so they reflect your actual filenames under `Images/` (e.g., `P{i}T{j}D{k}.jpg`).
- Adjust HSV thresholds or GrabCut iterations inside `src/segmentation.py` if lighting conditions differ between photos.
- HEIC images are supported via Pillow/pillow-heif; place files like `P1T1D1.heic` in `Images/` and the loader will convert them on the fly.
