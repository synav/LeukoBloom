"""LeukoBloom exploratory analysis: combining Model A (whole field) and Model B (cells).

Rules FIXED on 5 Nov 2026 before any combined result was computed (both single-model results were known,
so this is exploratory and needs prospective confirmation). No model is retrained or rerun: the saved
patient probabilities are combined. Model A enters through its mean field probability (A-meanprob), its
pre-specified probabilistic output; Model B through B-MIL.

  1. SEQUENCE  ("field first, then cells", as a hematologist reads a smear)
       Model A decides the group: acute (p_ALL+p_AML) vs chronic (p_CLL+p_CML) vs no leukemia (p_NO),
       largest wins (ties: acute, chronic, no leukemia). Within acute or chronic, Model B picks the subtype
       (larger of its two probabilities; tie -> ALL / CLL). If Model B has no usable cells, Model A's subtype.
  2. AVERAGE   equal-weight mean of the two probability vectors, top class (Model A alone if B has no cells).
  3. AGREEMENT both top classes equal -> accept that diagnosis; otherwise refer to a hematologist.
               Reported: share accepted, accuracy on accepted, accuracy of each model on referred patients.

Data: SSMC 5-fold CV (137 patients; results\\report_cv\\tables\\patient_predictions.csv) and the LeukemiaAttri
frozen external test per subset (results\\external_leukemiaattri\\patients_B.csv, fields_A_ensemble.csv).

Usage:  py -3.12 scripts\\combine_models.py
Output: results\\combined\\summary.md (+ copy inventory_out\\shareable\\combined_models.md), patient tables.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom import metrics as M  # noqa: E402
from leukobloom.rules import CLASSES, THREE  # noqa: E402

PC = [f"p_{c}" for c in CLASSES]
LABELS4 = ["ALL", "AML", "CLL", "CML"]


# ------------------------------------------------------------------------------ the three fixed rules
def top(P):
    return np.array([CLASSES[i] for i in P.argmax(1)])


def sequence(PA, PB):
    out = []
    for a, b in zip(PA, PB):
        g = int(np.argmax([a[0] + a[1], a[2] + a[3], a[4]]))      # argmax: first on ties
        if g == 2:
            out.append("NO_LEUKEMIA")
            continue
        src = a if np.isnan(b).any() else b
        i, j = (0, 1) if g == 0 else (2, 3)
        out.append(CLASSES[i] if src[i] >= src[j] else CLASSES[j])
    return np.array(out)


def average(PA, PB):
    P = np.where(np.isnan(PB).any(1, keepdims=True), PA, (PA + PB) / 2)
    return top(P)


def agreement(PA, PB):
    pa, pb = top(PA), np.where(np.isnan(PB).any(1), "NO_CELLS", top(np.nan_to_num(PB)))
    return pa == pb, pb


# ------------------------------------------------------------------------------ metrics
def fmt(t):
    return f"{t[0]:.2f} ({t[1]:.2f}–{t[2]:.2f})"


def block(y, p, labels, boot):
    def fn(ix):
        lm = M.label_metrics(list(y[ix]), list(p[ix]), labels)
        y3, p3 = [THREE[v] for v in y[ix]], [THREE.get(v, v) for v in p[ix]]
        d = dict(acc=lm["acc"], bal=lm["bal_acc"], f1=lm["macro_f1"],
                 acc3=float(np.mean([u == v for u, v in zip(y3, p3)])))
        yl = y[ix] != "NO_LEUKEMIA"
        pl = (p[ix] != "NO_LEUKEMIA") & (p[ix] != "NO_CELLS")
        d["sens"] = (pl & yl).sum() / max(yl.sum(), 1)
        if (~yl).any():
            d["spec"] = (~pl & ~yl).sum() / (~yl).sum()
        return d
    return M.bootstrap(fn, len(y), B=boot)


def section(title, y, PA, PB, singles, labels, boot, with_spec):
    preds = dict(singles)
    preds["1. Sequence (A field → B cells)"] = sequence(PA, PB)
    preds["2. Average"] = average(PA, PB)
    cols = ["Accuracy", "Balanced accuracy", "Macro-F1", "Acute/chronic/none", "Leukemia sensitivity"] + \
           (["Leukemia specificity"] if with_spec else [])
    keys = ["acc", "bal", "f1", "acc3", "sens"] + (["spec"] if with_spec else [])
    L = [f"### {title} (n = {len(y)})", "", "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, p in preds.items():
        ci = block(y, p, labels, boot)
        L.append(f"| {name} | " + " | ".join(fmt(ci[k]) for k in keys) + " |")
    bm = preds["Model B (B-MIL)"] == y
    L += [""]
    for name in ["1. Sequence (A field → B cells)", "2. Average"]:
        n01, n10, pv = M.mcnemar_exact(bm, preds[name] == y)
        L.append(f"- {name} vs Model B, McNemar exact: combined right / B wrong = {n01}, "
                 f"B right / combined wrong = {n10}, p = {pv:.3f}")
    agree, pb = agreement(PA, PB)
    ya, pa_ = y[agree], pb[agree]
    acc_ok = M.label_metrics(list(ya), list(pa_), labels)["acc"] if agree.any() else np.nan
    lo, hi = M.wilson(int((ya == pa_).sum()), int(agree.sum()))
    ref = ~agree
    L += ["", f"**3. Agreement:** accepted {agree.sum()}/{len(y)} ({agree.mean():.0%}); accuracy on accepted "
              f"{(ya == pa_).sum()}/{agree.sum()} = {acc_ok:.2f} (Wilson {lo:.2f}–{hi:.2f}); referred {ref.sum()} "
              f"— on these, Model B right {int((preds['Model B (B-MIL)'][ref] == y[ref]).sum())}, "
              f"Model A right {int((top(PA)[ref] == y[ref]).sum())}.", ""]
    return L, preds, agree


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    res = a.root / "results"
    out = res / "combined"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Combining Model A (whole field) and Model B (cells) — exploratory", "",
         "Rules fixed before any combined result was computed (see scripts/combine_models.py). Model A = mean field "
         "probability; Model B = MIL. 95% CIs: patient bootstrap. Exploratory: confirm prospectively.", ""]

    # ---- SSMC cross-validation
    cv = pd.read_csv(res / "report_cv" / "tables" / "patient_predictions.csv")
    y = cv.label5.values
    PA = cv[[f"A-meanprob:{c}" for c in PC]].values.astype(float)
    PB = cv[[f"B-MIL:{c}" for c in PC]].values.astype(float)
    singles = {"Model B (B-MIL)": cv["B-MIL"].values.astype(str), "Model A (A-meanprob)": top(PA),
               "Model A (A-published)": cv["A-published"].values.astype(str)}
    L += ["## SSMC, 5-fold cross-validation", ""]
    Ls, preds, agree = section("All cameras", y, PA, PB, singles, CLASSES, a.boot, True)
    L += Ls
    tab = cv[["patient_code", "label5", "camera"]].copy()
    for k, v in preds.items():
        tab[k] = v
    tab["agree"] = agree
    tab.to_csv(out / "patients_cv.csv", index=False)
    L += ["By camera (5-class accuracy): " + "; ".join(
        f"{cam} (n={int(m.sum())}): B {np.mean(preds['Model B (B-MIL)'][m] == y[m]):.2f}, "
        f"A {np.mean(preds['Model A (A-meanprob)'][m] == y[m]):.2f}, "
        f"sequence {np.mean(preds['1. Sequence (A field → B cells)'][m] == y[m]):.2f}, "
        f"average {np.mean(preds['2. Average'][m] == y[m]):.2f}"
        for cam in sorted(cv.camera.dropna().unique()) for m in [cv.camera.values == cam]), ""]

    # ---- LeukemiaAttri external (frozen)
    ext = res / "external_leukemiaattri"
    if (ext / "patients_B.csv").exists() and (ext / "fields_A_ensemble.csv").exists():
        pb = pd.read_csv(ext / "patients_B.csv")
        fa = pd.read_csv(ext / "fields_A_ensemble.csv").groupby("patient")[PC].mean()
        pub = pd.read_csv(ext / "patient_predictions.csv")
        pub = pub[pub.method == "A-published"].set_index("patient").pred
        L += ["## LeukemiaAttri external test (frozen models), per subset", "",
              "Leukemia patients only: specificity not defined; balanced accuracy and macro-F1 over ALL/AML/CLL/CML.",
              ""]
        rows = []
        for sub, g in pb.groupby("subset"):
            g = g[g.patient.isin(fa.index)]
            y = g.label5.values
            PA = fa.loc[g.patient, PC].values.astype(float)
            PB = g[PC].values.astype(float)
            mb = np.where(np.isnan(PB).any(1), "NO_CELLS", top(np.nan_to_num(PB)))
            singles = {"Model B (B-MIL)": mb, "Model A (A-meanprob)": top(PA),
                       "Model A (A-published)": pub.reindex(g.patient).fillna("NO_CELLS").values.astype(str)}
            Ls, preds, agree = section(sub, y, PA, PB, singles, LABELS4, a.boot, False)
            L += Ls
            t = g[["subset", "patient", "label5"]].copy()
            for k, v in preds.items():
                t[k] = v
            t["agree"] = agree
            rows.append(t)
        pd.concat(rows).to_csv(out / "patients_leukemiaattri.csv", index=False)
    else:
        L += ["LeukemiaAttri: results not found (run external_leukemiaattri.py --model b and --model a)."]

    txt = "\n".join(L) + "\n"
    (out / "summary.md").write_text(txt, encoding="utf-8")
    shr = a.root / "inventory_out" / "shareable" / "combined_models.md"
    shutil.copy(out / "summary.md", shr)
    print(txt + f"\nSend to Claude: {shr}")


if __name__ == "__main__":
    main()
