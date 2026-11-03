"""LeukoBloom report: every pre-specified analysis, Model A vs Model B, from the saved predictions.

Reads (no retraining):
  results\\model_b_b_<scheme>\\patients.csv, fields.csv      Model B (DinoBloom-B + MIL)
  results\\model_a_<scheme>\\fields_*.csv                     Model A (LeukoAI-style DenseNet)
  inventory_out\\shareable\\folds.csv                          camera per field

Methods compared (patient level):
  B-MIL            Model B attention-MIL over all cells of the patient        (Model B primary)
  A-published      Model A fields through the published LeukoAI rule, T = 3  (Model A primary)
  A-/B-<rule>      both models' FIELD probabilities through the same rules (like-for-like):
                   published, exp_majority, exp_meanprob, meanprob (leukobloom/rules.py)
Probability-based analyses (AUROC, calibration, deferral, decision curve) need a patient probability:
  B-MIL = its own output; Model A (and B-meanprob) = mean of field probabilities.
Leukemia screening: rule methods -> the rule's call; probabilistic methods -> p(leukemia) >= 0.5.

Outputs (Leukobloom\\results\\report_<scheme>\\, codes only): report.md (also copied to shareable),
tables\\*.csv, figures\\*.png. 95% CIs: bootstrap over patients (2,000 resamples, seed 2026); Wilson for
leukemia sensitivity/specificity; paired bootstrap and exact McNemar for A vs B.

Usage:  py -3.12 scripts\\report.py                 (cross-validation, all 137 patients)
        py -3.12 scripts\\report.py --scheme external
        py -3.12 scripts\\report.py --b-only        (Model B alone, before Model A is finished)
Needs:  matplotlib   (py -3.12 -m pip install matplotlib)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom import metrics as M  # noqa: E402
from leukobloom.rules import CLASSES, RULES, THREE, apply_rule  # noqa: E402

PC = [f"p_{c}" for c in CLASSES]
FIELD_COUNTS = [1, 3, 5, 10, 20]
DEFER = [0, 5, 10, 15, 20, 25, 30]


def fmt(t):
    return f"{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})"


def load(root: Path, scheme: str, b_only: bool):
    res = root / "results"
    bp = pd.read_csv(res / f"model_b_b_{scheme}" / "patients.csv")
    bf = pd.read_csv(res / f"model_b_b_{scheme}" / "fields.csv")
    af = None
    if not b_only:
        parts = sorted((res / f"model_a_{scheme}").glob("fields_fold*.csv" if scheme == "cv" else "fields_external.csv"))
        parts = [p for p in parts if "smoke" not in p.name]
        if (scheme == "cv" and len(parts) < 5) or not parts:
            raise SystemExit(f"Model A not finished ({len(parts)} fold files). Use --b-only for Model B alone.")
        af = pd.concat([pd.read_csv(p) for p in parts])
    folds = pd.read_csv(root / "inventory_out" / "shareable" / "folds.csv")
    return bp, bf, af, folds


def patient_table(bp, bf, af, folds):
    """One row per patient: truth, camera, and each method's label (+ probabilities where defined)."""
    pats = bp[["patient_code", "fold", "label5"] + PC].rename(columns={c: f"B-MIL:{c}" for c in PC})
    cam = folds.groupby("patient_code").capture.agg(lambda s: s.value_counts().index[0])
    pats["camera"] = pats.patient_code.map(cam)
    pats["B-MIL"] = np.where(pats[[f"B-MIL:{c}" for c in PC]].isna().any(axis=1), "NO_CELLS",
                             [CLASSES[i] for i in np.nan_to_num(pats[[f"B-MIL:{c}" for c in PC]].values).argmax(1)])
    for tag, f in (("B", bf), ("A", af)):
        if f is None:
            continue
        g = {k: v[PC].values for k, v in f.groupby("patient_code")}
        for rule in RULES:
            pats[f"{tag}-{rule}"] = [apply_rule(g[p], rule) if p in g else "NO_CELLS" for p in pats.patient_code]
        mean = {k: np.nanmean(v, 0) if (~np.isnan(v).any(1)).any() else np.full(5, np.nan) for k, v in g.items()}
        for i, c in enumerate(PC):
            pats[f"{tag}-meanprob:{c}"] = [mean.get(p, np.full(5, np.nan))[i] for p in pats.patient_code]
    return pats


def methods(pats):
    order = ["B-MIL"] + [f"B-{r}" for r in RULES] + [f"A-{r}" for r in RULES]
    return [m for m in order if m in pats.columns]


def prob(pats, meth):
    """Patient probability matrix for probabilistic methods (None for pure rules)."""
    base = "B-MIL" if meth == "B-MIL" else meth[0] + "-meanprob"
    if meth != "B-MIL" and meth.split("-", 1)[1] != "meanprob":
        return None
    return pats[[f"{base}:{c}" for c in PC]].values


def leuk_call(pats, meth):
    P = prob(pats, meth)
    if P is not None:
        return np.where(np.isnan(P).any(1), False, (1 - np.nan_to_num(P)[:, 4]) >= 0.5)
    return (pats[meth] != "NO_LEUKEMIA").values & (pats[meth] != "NO_CELLS").values


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--scheme", choices=["cv", "external"], default="cv")
    ap.add_argument("--b-only", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bp, bf, af, folds = load(a.root, a.scheme, a.b_only)
    pats = patient_table(bp, bf, af, folds)
    out = a.root / "results" / f"report_{a.scheme}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    pats.to_csv(out / "tables" / "patient_predictions.csv", index=False)
    y5 = pats.label5.values
    n = len(pats)
    meths = methods(pats)
    primary = ["B-MIL"] + (["A-published"] if "A-published" in pats else [])
    L = [f"# LeukoBloom report — scheme: {a.scheme}", "",
         f"Patients: {n} ({', '.join(f'{c} {int((y5 == c).sum())}' for c in CLASSES)}). "
         f"95% CI by bootstrap over patients ({a.boot} resamples) unless stated. "
         "Primary methods: B-MIL (Model B) and A-published (Model A). Codes only; no images.", ""]

    # 1. headline table ------------------------------------------------------------------------
    rows = []
    for m in meths:
        lc = leuk_call(pats, m)
        p5 = pats[m].values
        r = M.bootstrap(lambda idx: M.headline(y5[idx], p5[idx], lc[idx]), n, a.boot)
        rows.append((m, r))
    keys = ["5c_macro_f1", "5c_bal_acc", "3c_bal_acc", "3c_acc", "5c_acc", "leuk_sens", "leuk_spec"]
    L += ["## 1. Patient-level performance, all methods", "",
          "| Method | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for m, r in rows:
        L.append(f"| {'**' + m + '**' if m in primary else m} | " + " | ".join(fmt(r[k]) for k in keys) + " |")
    pd.DataFrame([{"method": m, **{f"{k}{s}": r[k][i] for k in keys for i, s in enumerate(['', '_lo', '_hi'])}}
                  for m, r in rows]).to_csv(out / "tables" / "headline.csv", index=False)
    L += ["", "Leukemia screening (Wilson 95% CI):", ""]
    yl = y5 != "NO_LEUKEMIA"
    for m in primary:
        lc = leuk_call(pats, m)
        tp, fn_, tn, fp = (lc & yl).sum(), (~lc & yl).sum(), (~lc & ~yl).sum(), (lc & ~yl).sum()
        s, sp = M.wilson(tp, tp + fn_), M.wilson(tn, tn + fp)
        L.append(f"- {m}: sensitivity {tp}/{tp + fn_} = {tp / (tp + fn_):.3f} ({s[0]:.3f}–{s[1]:.3f}); "
                 f"specificity {tn}/{tn + fp} = {tn / (tn + fp):.3f} ({sp[0]:.3f}–{sp[1]:.3f})")
    L.append("")

    # 2. primary comparison ---------------------------------------------------------------------
    if "A-published" in pats:
        L += ["## 2. Primary comparison: B-MIL vs A-published (paired)", ""]
        lb, la = leuk_call(pats, "B-MIL"), leuk_call(pats, "A-published")
        pb, pa = pats["B-MIL"].values, pats["A-published"].values

        def diff(idx):
            hb, ha = M.headline(y5[idx], pb[idx], lb[idx]), M.headline(y5[idx], pa[idx], la[idx])
            return {k: hb[k] - ha[k] for k in keys}
        d = M.bootstrap(diff, n, a.boot)
        L += ["| Difference (B − A) | Estimate (95% CI) |", "|---|---|"] + [f"| {k} | {fmt(d[k])} |" for k in keys]
        for name, tb, ta in (("5-class", pb == y5, pa == y5),
                             ("3-class", np.array([THREE[v] for v in pb]) == np.array([THREE[v] for v in y5]),
                              np.array([THREE[v] for v in pa]) == np.array([THREE[v] for v in y5]))):
            n01, n10, p = M.mcnemar_exact(ta, tb)
            L.append(f"\nMcNemar exact, {name}: B right / A wrong = {n01}, A right / B wrong = {n10}, p = {p:.3f}")
        L.append("")

    # 3. confusion matrices -----------------------------------------------------------------------
    L += ["## 3. Confusion matrices (rows = true, columns = predicted)", ""]
    for m in primary:
        for lab, yy, pp in (("5-class", list(y5), list(pats[m])), ("3-class", [THREE[v] for v in y5], [THREE[v] for v in pats[m]])):
            labels = CLASSES if lab == "5-class" else M.C3
            cm, cols = M.confusion(yy, pp, labels)
            L += [f"**{m}, {lab}**", "", "| true \\ pred | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
            L += [f"| {labels[i]} | " + " | ".join(str(v) for v in cm[i]) + " |" for i in range(len(labels))] + [""]

    # 4. per class ---------------------------------------------------------------------------------
    L += ["## 4. Per class: sensitivity, specificity, F1; AUROC (one-vs-rest)", ""]
    pc_rows = []
    for m in primary + [x for x in meths if x.split("-", 1)[1] == "meanprob"]:
        P = prob(pats, m)
        lm = M.label_metrics(list(y5), list(pats[m]), CLASSES)
        for i, c in enumerate(CLASSES):
            au = (np.nan,) * 3
            if P is not None:
                ok = ~np.isnan(P).any(1)
                au = M.bootstrap(lambda idx: {"a": M.auroc(P[ok][idx, i], (y5[ok] == c)[idx])}, ok.sum(), a.boot)["a"]
            pc_rows.append(dict(method=m, cls=c, n=int(lm["n_true"][i]), sens=lm["sens"][i], spec=lm["spec"][i],
                                f1=lm["f1"][i], auroc=au[0], auroc_lo=au[1], auroc_hi=au[2]))
    pcd = pd.DataFrame(pc_rows)
    pcd.to_csv(out / "tables" / "per_class.csv", index=False)
    L += ["| Method | Class | n | Sens | Spec | F1 | AUROC (95% CI) |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r.method} | {r.cls} | {r.n} | {r.sens:.3f} | {r.spec:.3f} | {r.f1:.3f} | "
          + (f"{r.auroc:.3f} ({r.auroc_lo:.3f}–{r.auroc_hi:.3f})" if not np.isnan(r.auroc) else "—") + " |"
          for r in pcd.itertuples()] + [""]

    # 5. calibration + 6. deferral + decision curve (probabilistic methods) ----------------------------------
    probm = [m for m in ["B-MIL", "A-meanprob", "B-meanprob"] if m in meths]
    L += ["## 5. Calibration (patient level, top-label)", "", "| Method | Brier (95% CI) | ECE |", "|---|---|---|"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for m in probm:
        P = prob(pats, m)
        ok = ~np.isnan(P).any(1)
        b = M.bootstrap(lambda idx: {"b": M.brier(P[ok][idx], y5[ok][idx])}, ok.sum(), a.boot)["b"]
        bins, ece = M.calibration_bins(P[ok], y5[ok])
        L.append(f"| {m} | {fmt(b)} | {ece:.3f} |")
        ax.plot([r[2] for r in bins], [r[3] for r in bins], marker="o", label=m)
        pd.DataFrame(bins, columns=["lo", "hi", "confidence", "accuracy", "n"]).to_csv(out / "tables" / f"calibration_{m}.csv", index=False)
    ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1), title="Reliability")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "calibration.png", dpi=150); plt.close(fig)

    L += ["", "## 6. Deferral: refer the least confident X% to a hematologist", "",
          "| Method | " + " | ".join(f"{x}%" for x in DEFER) + " |", "|" + "---|" * (len(DEFER) + 1)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m in probm:
        P = np.nan_to_num(prob(pats, m), nan=0.0)
        order = np.argsort(P.max(1))
        pred = np.array([CLASSES[i] for i in P.argmax(1)])
        accs = []
        for x in DEFER:
            keep = order[int(round(n * x / 100)):]
            accs.append(M.label_metrics(list(y5[keep]), list(pred[keep]), CLASSES)["bal_acc"])
        L.append(f"| {m} (5-class balanced accuracy) | " + " | ".join(f"{v:.3f}" for v in accs) + " |")
        ax.plot(DEFER, accs, marker="o", label=m)
    ax.set(xlabel="Patients referred (%)", ylabel="Balanced accuracy, retained", title="Deferral")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "deferral.png", dpi=150); plt.close(fig)

    L += ["", "Decision curve, leukemia vs no leukemia (net benefit; treat-all in brackets):", ""]
    th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    L += ["| Method | " + " | ".join(f"t={t}" for t in th) + " |", "|" + "---|" * (len(th) + 1)]
    for m in probm:
        P = np.nan_to_num(prob(pats, m), nan=0.0)
        nb = M.net_benefit(1 - P[:, 4], yl, th)
        L.append(f"| {m} | " + " | ".join(f"{v:.3f} ({w:.3f})" for _, v, w in nb) + " |")

    # 7. field count (field-level rules and mean, resampled) --------------------------------------------------
    L += ["", "## 7. Number of fields per patient (200 resamples per patient)", "",
          "5-class macro-F1 / 3-class balanced accuracy when only k fields are available:", ""]
    fc_rows = []
    rng = np.random.default_rng(2026)
    fsets = {tag: {k: v[PC].values for k, v in f.groupby("patient_code")} for tag, f in (("A", af), ("B", bf)) if f is not None}
    for tag, g in fsets.items():
        for rule in ["published", "meanprob"]:
            for k in FIELD_COUNTS:
                f1s, b3s = [], []
                for _ in range(200):
                    p5 = [apply_rule(g[p][rng.choice(len(g[p]), min(k, len(g[p])), replace=False)], rule)
                          if p in g else "NO_CELLS" for p in pats.patient_code]
                    h = M.headline(y5, p5, [v not in ("NO_LEUKEMIA", "NO_CELLS") for v in p5])
                    f1s.append(h["5c_macro_f1"]); b3s.append(h["3c_bal_acc"])
                fc_rows.append(dict(method=f"{tag}-{rule}", k=k, macro_f1=np.mean(f1s), bal_acc3=np.mean(b3s)))
    fcd = pd.DataFrame(fc_rows)
    fcd.to_csv(out / "tables" / "field_count.csv", index=False)
    L += ["| Method | " + " | ".join(f"k={k}" for k in FIELD_COUNTS) + " |", "|" + "---|" * (len(FIELD_COUNTS) + 1)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, g in fcd.groupby("method", sort=False):
        L.append(f"| {m} | " + " | ".join(f"{r.macro_f1:.2f} / {r.bal_acc3:.2f}" for r in g.itertuples()) + " |")
        ax.plot(g.k, g.macro_f1, marker="o", label=m)
    ax.set(xlabel="Fields per patient", ylabel="5-class macro-F1", xscale="log", title="Field count")
    ax.set_xticks(FIELD_COUNTS, [str(k) for k in FIELD_COUNTS])
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "field_count.png", dpi=150); plt.close(fig)
    L.append("\nThe published rule needs more than 3 leukemic fields, so with k ≤ 3 it calls every patient 'no leukemia'.")

    # 8. by camera --------------------------------------------------------------------------------------------
    L += ["", "## 8. By camera (descriptive)", "", "| Camera | n | " + " | ".join(f"{m} 5c macro-F1 / 3c bal-acc" for m in primary) + " |",
          "|" + "---|" * (len(primary) + 2)]
    for cam, g in pats.groupby("camera"):
        cells = []
        for m in primary:
            h = M.headline(g.label5.values, g[m].values, leuk_call(g, m))
            cells.append(f"{h['5c_macro_f1']:.2f} / {h['3c_bal_acc']:.2f}")
        L.append(f"| {cam} | {len(g)} | " + " | ".join(cells) + " |")

    txt = "\n".join(L)
    (out / "report.md").write_text(txt, encoding="utf-8")
    (a.root / "inventory_out" / "shareable" / f"report_{a.scheme}.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nFigures and tables: {out}\nSend to Claude: inventory_out\\shareable\\report_{a.scheme}.md (+ figures if you like)")


if __name__ == "__main__":
    main()
