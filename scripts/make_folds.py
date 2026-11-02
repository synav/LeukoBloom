"""LeukoBloom: build the patient-grouped cross-validation folds.

Input : inventory_out/shareable/inventory.csv  (from inventory.py; no names or IDs)
Output: inventory_out/shareable/folds.csv          one row per image: label, fold, include/exclude
        inventory_out/shareable/folds_summary.txt  patients and images per fold

Decisions applied (agreed 2 Nov 2026, fixed before any model is trained):
  D1  Every field takes its SAMPLE's diagnosis. A sample with leukemia images and a few
      fields filed as Normal/Reactive is labelled with the leukemia class.
  D2  Images without a patient ID and teaching-slide images go to TRAINING folds only
      (fold = -1: added to the training data of every fold, never tested).
  D3  Excluded: extra exact copies (first of each identical group kept) and images whose
      file name marks the diagnosis as uncertain ("POSSIBLE ...").
  D4  Near-duplicates are kept (all within one patient, so they cannot leak across folds).

Folds: 5 folds, grouped by patient, stratified by diagnosis AND capture type (camera /
image size), so each fold has a similar mix of both. Fixed seed -> same folds every run,
used by BOTH models.

Usage:  py -3.12 make_folds.py   [--inv path\\to\\inventory.csv] [--k 5] [--seed 2026]
Needs:  pandas, numpy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

LEUK = ["ALL", "AML", "CLL", "CML"]
THREE = {"ALL": "ACUTE", "AML": "ACUTE", "CLL": "CHRONIC", "CML": "CHRONIC", "NO_LEUKEMIA": "NO_LEUKEMIA"}


def capture_type(w: str, h: str) -> str:
    """Capture group from image size (one camera per size on the same Olympus BX53)."""
    return {"1600x1200": "C1600", "1920x1200": "C1920", "4104x2174": "C4104"}.get(f"{w}x{h}", "OTHER")


def patient_label(classes: set[str]) -> str:
    leuk = [c for c in classes if c in LEUK]
    if len(set(leuk)) > 1:
        return "CONFLICT"
    return leuk[0] if leuk else "NO_LEUKEMIA"


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", type=Path, default=home / "Downloads/Leukobloom/inventory_out/shareable/inventory.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    d = pd.read_csv(a.inv, dtype=str).fillna("")
    d["capture"] = [capture_type(w, h) for w, h in zip(d.width, d.height)]
    d["include"] = True
    d["exclude_reason"] = ""

    # D3: exclusions
    unc = d["flags"].str.contains("uncertain_label_in_name")
    d.loc[unc, ["include", "exclude_reason"]] = [False, "uncertain_diagnosis_in_name"]
    dup = d[d.exact_dup_group != ""].sort_values("image_id")
    extra = dup[dup.duplicated("exact_dup_group", keep="first")].index
    d.loc[extra, ["include", "exclude_reason"]] = [False, "exact_duplicate_copy"]

    # D1: label = sample diagnosis
    inc = d[d.include]
    plab = inc[inc.patient_code != ""].groupby("patient_code").class5.apply(lambda s: patient_label(set(s)))
    if (plab == "CONFLICT").any():
        raise SystemExit(f"Two leukemia classes in one sample: {list(plab[plab == 'CONFLICT'].index)}")
    d["label5"] = d.patient_code.map(plab).where(d.patient_code != "", d.class5)
    d["relabelled"] = d.include & (d.label5 != d.class5)
    d["label3"] = d.label5.map(THREE)

    # D2: training-only images
    teach = d.patient_source == "teaching"
    noid = d.patient_code == ""
    d["role"] = np.where(teach | noid, "train_only", "cv")

    # patient table for fold assignment
    cv = d[d.include & (d.role == "cv")]
    pats = cv.groupby("patient_code").agg(label5=("label5", "first"),
                                          capture=("capture", lambda s: s.value_counts().index[0]),
                                          n_img=("image_id", "size")).reset_index()
    rng = np.random.default_rng(a.seed)
    pats = pats.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    # strata: biggest first; within stratum, fill the fold with fewest patients of that label, then fewest images
    fold_of = {}
    n_lab = np.zeros((a.k, len(pats.label5.unique())), int)
    lab_idx = {l: i for i, l in enumerate(sorted(pats.label5.unique()))}
    n_img = np.zeros(a.k, int)
    strata = pats.groupby(["label5", "capture"]).size().sort_values(ascending=False).index
    start = 0
    for lab, cap in strata:
        grp = pats[(pats.label5 == lab) & (pats.capture == cap)].sort_values("n_img", ascending=False)
        for _, p in grp.iterrows():
            li = lab_idx[lab]
            order = sorted(range(a.k), key=lambda f: (n_lab[f, li], n_img[f], (f - start) % a.k))
            f = order[0]
            fold_of[p.patient_code] = f
            n_lab[f, li] += 1
            n_img[f] += p.n_img
        start = (start + 1) % a.k
    d["fold"] = d.patient_code.map(fold_of).fillna(-1).astype(int)
    d.loc[d.role == "train_only", "fold"] = -1

    out = a.inv.parent
    cols = ["image_id", "set", "patient_code", "class7", "class5", "label5", "label3", "relabelled",
            "capture", "role", "fold", "include", "exclude_reason"]
    d[cols].to_csv(out / "folds.csv", index=False)

    # summary
    L = ["LeukoBloom folds — summary", f"k={a.k}  seed={a.seed}", ""]
    L.append(f"Images: {len(d)}  included: {d.include.sum()}  excluded: {(~d.include).sum()}")
    L.append("  " + ", ".join(f"{k}={v}" for k, v in d[~d.include].exclude_reason.value_counts().items()))
    L.append(f"Relabelled to sample diagnosis: {d.relabelled.sum()} images in "
             f"{d[d.relabelled].patient_code.nunique()} samples")
    L.append(f"Training-only images (no ID or teaching slide): {(d.include & (d.role == 'train_only')).sum()}")
    L.append("")
    L.append("Patients per fold and diagnosis (tested once each):")
    L.append(pd.crosstab(pats.patient_code.map(fold_of).rename("fold"), pats.label5, margins=True).to_string())
    L.append("")
    L.append("Patients per fold and capture type:")
    L.append(pd.crosstab(pats.patient_code.map(fold_of).rename("fold"), pats.capture, margins=True).to_string())
    L.append("")
    inc_cv = d[d.include & (d.role == "cv")]
    L.append("Test images per fold and diagnosis:")
    L.append(pd.crosstab(inc_cv.fold, inc_cv.label5, margins=True).to_string())
    L.append("")
    L.append("Capture type by diagnosis, all included images (patients in CV + training-only images):")
    L.append(pd.crosstab(d[d.include].capture, d[d.include].label5, margins=True).to_string())
    txt = "\n".join(L)
    (out / "folds_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
