"""LeukoBloom post hoc analysis: can the LeukemiaAttri images support diagnosis when the model learns from them?

Secondary, post hoc (decided 5 Nov 2026, after the frozen external test). DinoBloom-B stays frozen; the cell
features saved by external_leukemiaattri.py --model b are reused. Only the MIL step is trained, FROM SCRATCH,
with exactly the SSMC settings (train_mil.py: 40 epochs, AdamW lr 2e-4, 3 seeds, class-balanced loss,
at most 512 cells and 50-100% of fields per training bag). Nothing from the SSMC models is used.

Folds: 5, grouped by PATIENT (diagnosis + number; all four microscope/camera images of a patient in the same
fold), stratified by diagnosis, seed 2026. A bag = one patient under one condition. Every patient-condition is
tested once by models that never saw that patient under any condition.

Scheme --scheme cross-microscope: train on the high-cost microscope (H_*) only, test on the low-cost (L_*),
same patient-grouped folds.

Usage:  py -3.12 scripts\\leukemiaattri_cv.py
        py -3.12 scripts\\leukemiaattri_cv.py --scheme cross-microscope
Output: results\\leukemiaattri_cv[_cross-microscope]\\patients.csv, summary.md
        (summary also copied to inventory_out\\shareable\\leukemiaattri_cv[...].md; codes and numbers only)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from leukobloom.metrics import bootstrap, confusion  # noqa: E402
from leukobloom.mil import CLASSES  # noqa: E402
from external_leukemiaattri import LABELS4, fmt_ci, method_block  # noqa: E402
from train_mil import SEEDS, predict, train_one  # noqa: E402

N_FOLDS, SEED = 5, 2026


def assign_folds(cases: pd.DataFrame) -> dict:
    """Stratified by diagnosis: shuffle the cases of each class, deal them round-robin over the folds."""
    rng = np.random.default_rng(SEED)
    fold, start = {}, 0
    for lab, g in cases.groupby("label5"):
        ids = list(g.case)
        rng.shuffle(ids)
        for i, c in enumerate(ids):
            fold[c] = (start + i) % N_FOLDS
        start += len(ids)
    return fold


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--scheme", choices=["within", "cross-microscope"], default="within")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=0, help=argparse.SUPPRESS)   # testing only
    a = ap.parse_args()
    if a.epochs:
        import train_mil
        train_mil.EPOCHS = a.epochs

    idx = pd.read_csv(a.root / "results" / "external_leukemiaattri" / "index.csv")
    fdir = a.root / "features" / "leukemiaattri_dinobloom_b"
    fields = {}
    for fid in idx.field_id:
        z = np.load(fdir / f"{fid}.npz")
        f = z["feats"].astype(np.float32)
        fields[fid] = f[~z["touches_border"].astype(bool)] if len(f) else f
    idx["case"] = idx.patient.str.split("/").str[1]
    bags = (idx.groupby(["subset", "patient", "case", "label5"]).field_id.apply(list).rename("image_ids")
            .reset_index())
    bags["n_cells"] = [sum(len(fields[i]) for i in ids) for ids in bags.image_ids]
    fold = assign_folds(bags.drop_duplicates("case")[["case", "label5"]])
    bags["fold"] = bags.case.map(fold)
    is_train_sub = bags.subset.str.startswith("H_") if a.scheme == "cross-microscope" else bags.subset.notna()
    is_test_sub = bags.subset.str.startswith("L_") if a.scheme == "cross-microscope" else bags.subset.notna()
    print(f"{bags.case.nunique()} patients, {len(bags)} patient-condition bags; scheme {a.scheme}; "
          f"folds: " + ", ".join(f"{k}: {(pd.Series(fold) == k).sum()}" for k in range(N_FOLDS)), flush=True)

    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 1))
    rows, t0 = [], time.time()
    for k in range(N_FOLDS):
        tr = bags[(bags.fold != k) & is_train_sub & (bags.n_cells > 0)].reset_index(drop=True)
        te = bags[(bags.fold == k) & is_test_sub]
        models = [train_one(tr, fields, 768, s) for s in SEEDS]
        for r in te.itertuples():
            p = predict(models, r.image_ids, fields)[0] if r.n_cells else np.full(len(CLASSES), np.nan)
            rows.append(dict(subset=r.subset, patient=r.patient, case=r.case, label5=r.label5, fold=k,
                             n_cells=r.n_cells, **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
        print(f"  fold {k}: trained on {len(tr)} bags ({tr.case.nunique()} patients), tested {len(te)} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = a.root / "results" / ("leukemiaattri_cv" + ("" if a.scheme == "within" else "_" + a.scheme))
    out.mkdir(parents=True, exist_ok=True)
    pt = pd.DataFrame(rows)
    P = pt[[f"p_{c}" for c in CLASSES]].values
    pt["pred"] = np.where(np.isnan(P).any(1), "NO_CELLS", [CLASSES[i] for i in np.nan_to_num(P).argmax(1)])
    pt.to_csv(out / "patients.csv", index=False)

    title = ("trained and tested within LeukemiaAttri" if a.scheme == "within"
             else "trained on the high-cost microscope, tested on the low-cost microscope")
    L = [f"# LeukemiaAttri patient-grouped 5-fold CV — {title}", "",
         "Post hoc, secondary. Frozen DinoBloom-B features (expert boxes); MIL trained from scratch with the SSMC "
         "settings; folds grouped by patient across all four conditions. Compare with the frozen SSMC models in "
         "external_leukemiaattri.md. 95% CIs: patient bootstrap.", "",
         "| Subset | n | Accuracy | Balanced accuracy | Macro-F1 | Acute/chronic | Called leukemia |",
         "|---|---|---|---|---|---|---|"]
    for sub, g in pt.groupby("subset"):
        y, p = g.label5.values, g.pred.values

        def fn(ix, y=y, p=p):
            lm, ac, le = method_block(y[ix], p[ix])
            return dict(acc=lm["acc"], bal=lm["bal_acc"], f1=lm["macro_f1"], ac=ac, le=le)
        ci = bootstrap(fn, len(y), B=a.boot)
        L.append(f"| {sub} | {len(y)} | {fmt_ci(ci['acc'])} | {fmt_ci(ci['bal'])} | {fmt_ci(ci['f1'])} | "
                 f"{fmt_ci(ci['ac'])} | {fmt_ci(ci['le'])} |")
    L += ["", "## Confusion matrices (rows true, columns predicted)", ""]
    for sub, g in pt.groupby("subset"):
        m, cols = confusion(list(g.label5), list(g.pred), LABELS4)
        L += [f"**{sub}**", "", "| True \\ Pred | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        L += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in zip(LABELS4, m) if row.sum()]
        L.append("")
    cons = pt.groupby("case").pred.nunique().eq(1).mean()
    L += [f"Same prediction for a patient in every tested condition: {cons:.2f} "
          f"({pt.case.nunique()} patients).", ""]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    shr = a.root / "inventory_out" / "shareable" / f"{out.name}.md"
    shutil.copy(out / "summary.md", shr)
    print("\n".join(L) + f"\nSend to Claude: {shr}")


if __name__ == "__main__":
    main()
