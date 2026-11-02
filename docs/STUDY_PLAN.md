# Study plan — LeukoAI-style vs DinoBloom cell-level model

Last updated 2 Nov 2026. Replaces the earlier plan to re-validate the published LeukoAI on 88 cases.
Decided 30 Oct 2026: the published 64 external-validation cases cannot be reconstructed with
certainty (65 folders on disk, one dropped by the original researcher, unknown which), so exact
reproduction is dropped. Instead, a new controlled comparison on the full dataset.

## The study in one line

Two new models, trained and tested on **identical patient-level splits of the same images**:
a whole-field model built the LeukoAI way vs a DinoBloom-powered cell-segmented model.
Any difference in results comes from the method, not the data.

## Relationship to the published LeukoAI

- Published LeukoAI (v1.0 / online tool 1.1) stays **frozen** as released on GitHub and Hugging Face.
- In the new paper it is **cited only as prior work**: this study extends the original LeukoAI study
  to a similar set of patients (same centre, same capture method, same data type).
  Reference: Naveed S et al., J Med Artif Intell 2025, doi:10.21037/jmai-24-74.
- It is **not** one of the compared models and no claim is made about reproducing its published numbers.
- Optional: run it on the test patients as a descriptive reference only, noting that some of those
  patients were in its own development data.

## Models

| | Model A — "LeukoAI-style" | Model B — "DinoBloom-powered" |
|---|---|---|
| Input | Whole high-power field | Each white cell, detected and cropped |
| Image features | DenseNet, trained end to end | DinoBloom (frozen foundation model) |
| Patient result | Image predictions aggregated per patient | (a) same aggregation rule as Model A, like-for-like; (b) attention-based multiple-instance learning |
| Explanation | Heatmaps per field | Ranked cells that drove the result (cell gallery) |

Aggregation rules (published-style rule, `exp_majority`, `exp_meanprob`) are fixed in code before any
test result is seen and applied identically to both models.

## Data

| Data | Use | Restriction |
|---|---|---|
| Training class folders (ALL, AML, CLL, CML, normal+reactive) | Training folds | Images without a patient prefix (e.g. `Image_3801`) go only into training folds, never test folds |
| Validation case folders (one patient per folder; 88 cases incl. the 65 old validation folders) | Training and test folds | Renamed `DIAGNOSIS_COHORT###`; key to hospital IDs kept separately |
| Zenodo public images (2,832) | Building and testing the pipeline | Never for performance claims |
| Public datasets (e.g. Munich AML cytomorphology) | Debugging cell detection / DinoBloom | Different stain and capture |

## Steps

1. **New private repo** (e.g. `LeukoBloom`), added to the Claude GitHub app.
2. **Data inventory script** (runs on the laptop; images never leave it): patient, diagnosis,
   image count and image fingerprint for every image. Fingerprints catch duplicates between
   training and validation folders and between patients.
3. **Patient-grouped 5-fold cross-validation** over all patients with a known identity; the same folds
   for both models; every patient tested once by a model that never saw them. Folds stratified by diagnosis.
4. **Model B pipeline** first built on Zenodo/public images: cell detection → DinoBloom embeddings → MIL.
5. **Model A training** on the laptop CPU (Ryzen AI 7 350, 32 GB), fold by fold overnight; benchmark first.
6. **Run both models on the folds**; one report script for all analyses below.
7. **Hematologist review** of top-ranked cells in a sample of cases (~500–1,000 cells) and of all
   misclassified cases.

## Data decisions (fixed 2 Nov 2026, before any model training)

Inventory (`inventory.py`): 7,123 images; 143 samples with a lab ID (56 training folders, 87 validation cases:
60 "V", 27 "N"); 1,093 training images without ID; 5 teaching slides (249 images). No sample in both
training and validation; no duplicate images between samples.

- **D1** Every field takes its sample's diagnosis (8 leukemia samples had 23 fields filed as Normal/Reactive).
- **D2** Images without ID and teaching-slide images: training folds only, never tested.
- **D3** Excluded: 5 exact duplicate copies, 3 "POSSIBLE ALL" images.
- **D4** Near-duplicates kept (all within one sample).
- **Folds** (`make_folds.py`, seed 2026): 5 folds grouped by sample, stratified by diagnosis and camera;
  137 samples tested once each (ALL 11, AML 41, CLL 14, CML 10, no leukemia 61).
- **Capture:** Olympus BX53, 1,000× oil, MGG (Sysmex DI60) as in the published study; three cameras
  (1600×1200, 1920×1200, 4104×2174). Training folders are all 1600×1200; validation cases are 1920×1200
  or 4104×2174. Images are rescaled to a common physical scale (estimated from red-cell size per camera)
  and colour-normalised; results reported per camera.
- **Secondary analysis (camera shift):** train on the 1600×1200 training-folder samples only, test on the
  87 validation cases (different cameras).
- **Limitation:** lab IDs identify samples, not patients; two samples from one patient cannot be linked
  without hospital records.

## Model settings (fixed 2 Nov 2026, before any real result)

**Model B (DinoBloom + MIL)**
- Detector v2: per-field white balance; purple threshold; one red-cell size per camera (median of 40 fields);
  crops of 3 red-cell diameters, 224x224 px, from the white-balanced field. Cells touching the image edge
  excluded (sensitivity analysis: included).
- Features: DinoBloom-B (frozen, `1aurent/vit_base_patch14_224.dinobloom`); DinoBloom-S as sensitivity analysis.
- Gated-attention MIL (Ilse 2018): LayerNorm, 768->256, attention 128, dropout 0.25. Bag = all cells of a
  sample; training bags use a random 50-100% of fields and at most 512 cells. 40 epochs, AdamW lr 2e-4,
  weight decay 1e-4, one-cycle schedule, class-balanced loss, 3 seeds averaged. No tuning on test folds.
- Training-only images without ID: pseudo-patients of 10 fields per class folder.
- Field-level probabilities (each field as its own bag) for the aggregation rules shared with Model A.
- Patients with no usable cells: no prediction, counted as errors in all metrics.
- Pipeline checks on synthetic features with the real folds: learnable signal recovered; pure noise gives
  chance-level balanced accuracy (0.22 vs 0.20), i.e. no leakage between folds.

**Model A (LeukoAI-style)**: single 5-class DenseNet-121, ImageNet start, 448x448, full fine-tuning;
published-style rule applied to its probabilities (leukemia = 1 - p(no leukemia); acute = p(ALL)+p(AML)).

**Primary metrics**: patient-level balanced accuracy and macro-F1 (5-class and 3-class); accuracy reported
but not primary (dominated by AML and no-leukemia).

## Pre-specified analyses

1. **Primary:** patient-level 5-class macro-F1 and 3-class (acute/chronic/no leukemia) accuracy,
   Model A vs Model B, paired comparison (McNemar; bootstrap 95% CIs for the difference).
2. **Like-for-like:** Model B with Model A's aggregation rule, to separate the gain from better
   features vs better patient-level aggregation.
3. **Aggregation rules:** published-style vs `exp_majority` vs `exp_meanprob`, no tuning on test folds.
4. **Field count:** resample 1, 3, 5, 10, 20 fields per patient; accuracy vs number of fields.
5. **Deferral:** accuracy when the least certain X% of patients are referred to a hematologist.
6. **Failure and stain analysis:** e.g. Zenodo AML fields `Image_3816`–`3829` (typical blasts, classed CLL by v1.0).
7. Per-subtype results with 95% CIs; small classes (ALL, CLL, CML) stated plainly as a limitation.

## Positioning

Foundation-model MIL on smears has been published with slide scanners (e.g. CAREMIL 2026,
cAItomorph 2025). Contributions here: manually captured high-power fields (no scanner); a Gulf
population; a controlled head-to-head of whole-field vs cell-level methods on identical splits;
fully documented image inventory; practical guidance on field count and deferral.

## Open items

| Item | Owner |
|---|---|
| Create private repo | Dr Naveed |
| Describe folder layout (training + validation, example names, IDs replaced) | Dr Naveed |
| Rename case folders to `DIAGNOSIS_COHORT###` | Done |
| Inventory script | Done |
| IRB amendment | SSMC co-investigator |
| MBZUAI/SSMC written agreement | To obtain |

## IRB amendment — points to cover

- Original approval number; extension of the retrospective LeukoAI study to a similar set of patients.
- Cases outside the original period (number to confirm); same data type (de-identified smear images + final diagnosis).
- New aims: development and comparison of two models (whole-field vs DinoBloom cell-level) on the existing data.
- Personnel: SSMC co-investigator as site lead; Dr Naveed as external collaborator (CCAD).
- Data handling: images stay at SSMC/on the study laptop, or transfer under a data-sharing agreement;
  GPU training at MBZUAI covered by that agreement.
- Continued waiver of consent (no patient contact).
- The IRB member who is a co-investigator should recuse from the vote.
