# LeukoBloom

**Whole-field vs DinoBloom cell-level leukemia classification on identical patient-level splits.**

Private research repository. Code only — **no patient images or data are stored here.**

## The study

Two new models are trained and tested on the **same images and the same patient-grouped
5-fold cross-validation**, so any difference in results comes from the method, not the data:

| | Model A — "LeukoAI-style" | Model B — "DinoBloom-powered" |
|---|---|---|
| Input | Whole high-power field | Each white cell, detected and cropped |
| Features | DenseNet, trained end to end | DinoBloom (frozen foundation model) |
| Patient result | Field predictions aggregated per patient | Same aggregation rule (like-for-like) and attention-based multiple-instance learning |
| Explanation | Heatmaps | Ranked cells that drove the result |

This study extends the original LeukoAI study to a similar set of patients (same centre,
microscope and data type). The published LeukoAI model stays frozen and is cited only as prior work:
Naveed S et al., *J Med Artif Intell* 2025, doi:[10.21037/jmai-24-74](https://doi.org/10.21037/jmai-24-74).

Full design, data decisions and pre-specified analyses: [`docs/STUDY_PLAN.md`](docs/STUDY_PLAN.md).

## Folder layout on the study laptop

```
Downloads\Leukobloom\
   data\                 ← images (NEVER in git)
   external\LeukemiaAttri\ ← public dataset, 5 export parts as downloaded
      training\          ALL, AML, APML, CLL, CML, NORMAL, REACTIVE CELLS
      validation\        one folder per case, e.g. AML_V001, Normal_N003
   inventory_out\        ← script outputs (NEVER in git)
      shareable\         inventory.csv, summary.txt, folds.csv  (codes only)
      PRIVATE\           keys to file names and lab IDs  (never share)
   features\             ← DinoBloom features per image (NEVER in git)
   cache\                ← prepared fields for Model A (NEVER in git)
   results\              ← model outputs per fold (NEVER in git)
   repo\                 ← this repository
```

## Steps

| # | Step | Script | Status |
|---|---|---|---|
| 1 | Data inventory: labels, patient grouping, fingerprints, duplicate and overlap checks | `scripts/inventory.py` | Done |
| 2 | Patient-grouped 5-fold split, stratified by diagnosis and camera | `scripts/make_folds.py` | Done |
| 3 | CPU benchmark (DenseNet training, DinoBloom features) | `scripts/benchmark.py` | Done |
| 3b | White-cell detector check per camera (overlays stay local) | `scripts/check_cells.py`, `leukobloom/cells.py` | Done (v2) |
| 4a | Model B: crop every cell, extract DinoBloom features (resumable) | `scripts/extract_features.py` | Done |
| 4b | Model B: gated-attention MIL, 5-fold CV and camera-shift (external) scheme | `scripts/train_mil.py`, `leukobloom/mil.py` | Done (CV) |
| 5a | Model A: prepare fields at common scale (red cell ~28 px), white-balanced | `scripts/prepare_fields.py` | Done |
| 5b | Model A: DenseNet-121 whole-field, one fold per night (resumable) | `scripts/train_densenet.py` | Running |
| 6b | Cell gallery: top-attention cells per patient for hematologist review (pictures stay local) | `scripts/cell_gallery.py` | Ready |
| 6 | Report: all pre-specified analyses, Model A vs B | `scripts/report.py`, `leukobloom/rules.py`, `leukobloom/metrics.py` | Ready |
| 6c | Publication figures: ROC with AUROC CIs, confusion matrices, paired differences, field count, calibration/deferral | `scripts/figures.py` | Ready |
| 7 | Public-data robustness test: LeukemiaAttri 100x subsets, frozen models, expert boxes for Model B | `scripts/external_leukemiaattri.py` | Ready |

## Running (Windows, Python 3.12)

```
cd %USERPROFILE%\Downloads\Leukobloom\repo
py -3.12 -m pip install -r requirements.txt
py -3.12 scripts\inventory.py
py -3.12 scripts\make_folds.py
```

Model steps also need CPU PyTorch and timm (one-time install):

```
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -3.12 -m pip install timm huggingface_hub
py -3.12 scripts\benchmark.py
```

Both scripts default to the folder layout above; use `--help` for other paths.

## Data protection

- Images are processed on the study laptop only.
- `.gitignore` blocks images, data folders, CSV files, model weights and anything named PRIVATE.
- Shareable outputs contain only codes (`IMG00001`, `P0001`, case folder names); the keys stay in `PRIVATE`.

Research use only. Not a medical device.
