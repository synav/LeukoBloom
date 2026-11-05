"""LeukoBloom: publication figures from the saved report tables (no model is rerun).

Reads results\\report_<scheme>\\tables\\ (written by report.py) and draws:
  fig_roc            ROC curves, one-vs-rest per class + acute leukemia + any leukemia (screening), with
                     AUROC (95% bootstrap CI); B-MIL vs A-meanprob (the pre-specified probabilistic methods),
                     A-published and B-MIL operating points marked on the screening panel
  fig_confusion      patient-level confusion matrices, B-MIL vs A-published, 5-class and 3-class
  fig_differences    primary comparison B-MIL − A-published with paired bootstrap 95% CIs
  fig_fields         performance vs number of fields per patient
  fig_calibration    reliability diagram and deferral curve
Each as 300-dpi PNG and vector PDF in results\\figures_<scheme>\\ (aggregate numbers only: shareable),
plus figure_stats.md (AUROCs, paired AUROC differences) copied to inventory_out\\shareable\\.

Usage:  py -3.12 scripts\\figures.py            (after report.py)
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
C_B, C_A = "#2a78d6", "#eb6834"          # Model B blue, Model A orange (validated categorical pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"B-MIL": "Model B (DinoBloom + MIL)", "A-meanprob": "Model A (whole-field CNN)",
         "A-published": "Model A, published rule", "B-meanprob": "Model B, mean of fields"}
LABEL = {"ALL": "ALL", "AML": "AML", "CLL": "CLL", "CML": "CML", "NO_LEUKEMIA": "No leukemia",
         "ACUTE": "Acute", "CHRONIC": "Chronic"}


def style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7, "figure.dpi": 100,
        "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(fig, out, name):
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(out / f"{name}.{ext}", **kw)


def roc_curve(score, pos):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    o = np.argsort(-score, kind="mergesort")
    s, p = score[o], pos[o]
    tps, fps = np.cumsum(p), np.cumsum(~p)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]          # one point per distinct threshold
    tpr = np.r_[0, tps[last] / max(p.sum(), 1)]
    fpr = np.r_[0, fps[last] / max((~p).sum(), 1)]
    return fpr, tpr


def auc_ci(scores, pos, B, seed=2026):
    """AUROC per score set + 95% CI, and paired difference of the first two (same resamples)."""
    rng = np.random.default_rng(seed)
    n = len(pos)
    pt = [M.auroc(s, pos) for s in scores]
    draws = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if pos[i].all() or not pos[i].any():
            continue
        draws.append([M.auroc(s[i], pos[i]) for s in scores])
    d = np.array(draws)
    ci = [(pt[k], *np.percentile(d[:, k], [2.5, 97.5])) for k in range(len(scores))]
    diff = (pt[0] - pt[1], *np.percentile(d[:, 0] - d[:, 1], [2.5, 97.5])) if len(scores) > 1 else None
    return ci, diff


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--scheme", choices=["cv", "external"], default="cv")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    tab = a.root / "results" / f"report_{a.scheme}" / "tables"
    out = a.root / "results" / f"figures_{a.scheme}"
    out.mkdir(parents=True, exist_ok=True)
    pats = pd.read_csv(tab / "patient_predictions.csv")
    y5 = pats.label5.values
    PB = pats[[f"B-MIL:{c}" for c in PC]].values.astype(float)
    PA = pats[[f"A-meanprob:{c}" for c in PC]].values.astype(float)
    PB, PA = np.nan_to_num(PB, nan=0.0), np.nan_to_num(PA, nan=0.0)   # no-cell patients: score 0 (counted as misses)
    S = ["# LeukoBloom figure statistics", "",
         f"Patients: {len(pats)}. AUROC with 95% CI (patient bootstrap, {a.boot} resamples, seed 2026); "
         "difference = Model B (B-MIL) − Model A (A-meanprob), paired.", "",
         "| Task | n positive | Model B AUROC | Model A AUROC | Difference B − A |", "|---|---|---|---|---|"]

    # ---------------------------------------------------------------- ROC
    tasks = [(LABEL[c], y5 == c, PB[:, i], PA[:, i]) for i, c in enumerate(CLASSES[:4])]
    tasks.append(("Acute (ALL + AML)", np.isin(y5, ["ALL", "AML"]), PB[:, 0] + PB[:, 1], PA[:, 0] + PA[:, 1]))
    tasks.append(("Any leukemia", y5 != "NO_LEUKEMIA", 1 - PB[:, 4], 1 - PA[:, 4]))
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, (title, pos, sb, sa) in zip(axs.flat, tasks):
        ci, diff = auc_ci([sb, sa], pos, a.boot)
        ax.plot([0, 1], [0, 1], ls=":", c=INK2, lw=0.8)
        for s, c, ls, (v, lo, hi), lab in ((sa, C_A, "--", ci[1], "Model A"), (sb, C_B, "-", ci[0], "Model B")):
            fx, ty = roc_curve(s, pos)
            ax.plot(fx, ty, c=c, lw=1.8, ls=ls, label=f"{lab}  {v:.2f} ({lo:.2f}–{hi:.2f})",
                    drawstyle="steps-post")
        if title == "Any leukemia":
            for col, c, mk, lab in (("A-published", C_A, "s", "Model A, published rule"),
                                    (None, C_B, "o", "Model B at p ≥ 0.5")):
                call = (pats[col].values != "NO_LEUKEMIA") & (pats[col].values != "NO_CELLS") if col \
                    else (1 - PB[:, 4]) >= 0.5
                ax.plot((call & ~pos).sum() / (~pos).sum(), (call & pos).sum() / pos.sum(), mk, ms=6, mfc=c,
                        mec="white", mew=1.2, ls="none", label=lab, zorder=5)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title=f"{title}  (n = {int(pos.sum())})",
               xlabel="1 − specificity", ylabel="Sensitivity", aspect="equal")
        ax.legend(loc="lower right", title="AUROC (95% CI)", title_fontsize=7, handlelength=2.2)
        S.append(f"| {title} | {int(pos.sum())} | {ci[0][0]:.3f} ({ci[0][1]:.3f}–{ci[0][2]:.3f}) | "
                 f"{ci[1][0]:.3f} ({ci[1][1]:.3f}–{ci[1][2]:.3f}) | {diff[0]:+.3f} ({diff[1]:+.3f} to {diff[2]:+.3f}) |")
    fig.suptitle("Patient-level ROC curves, 5-fold cross-validation", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig_roc")
    plt.close(fig)
    S += ["", "Operating points (any leukemia): " +
          "; ".join(f"{NAMES[m]}: sensitivity {np.mean(c[y5 != 'NO_LEUKEMIA']):.3f}, specificity "
                    f"{1 - np.mean(c[y5 == 'NO_LEUKEMIA']):.3f}"
                    for m, c in (("A-published", np.isin(pats['A-published'].values, ['NO_LEUKEMIA', 'NO_CELLS'], invert=True)),
                                 ("B-MIL", (1 - PB[:, 4]) >= 0.5))), ""]

    # ---------------------------------------------------------------- confusion matrices
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    C3 = ["ACUTE", "CHRONIC", "NO_LEUKEMIA"]
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.0), gridspec_kw=dict(height_ratios=[5, 3.4]))
    for col, meth in enumerate(["B-MIL", "A-published"]):
        p5 = pats[meth].values.astype(str)
        for row, (labs, y, p) in enumerate(((CLASSES, y5, p5),
                                            (C3, [THREE[v] for v in y5], [THREE.get(v, v) for v in p5]))):
            m, cols = M.confusion(list(y), list(p), list(labs))
            m = m[:, :len(labs)] if cols[:len(labs)] == list(labs) else m
            frac = m / np.maximum(m.sum(1, keepdims=True), 1)
            ax = axs[row, col]
            ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=8,
                            color="white" if frac[i, j] > 0.55 else INK)
            ax.set_xticks(range(len(labs)), [LABEL[v] for v in labs], rotation=30, ha="right")
            ax.set_yticks(range(len(labs)), [LABEL[v] for v in labs])
            ax.grid(False)
            ax.set(xlabel="Predicted", ylabel="True" if col == 0 else "")
            ax.set_title(f"{'Model B (DinoBloom + MIL)' if col == 0 else 'Model A, published rule'}"
                         f" — {'5' if row == 0 else '3'} classes", fontsize=8.5)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.text(0.01, 0.005, "Numbers = patients; shading = share of the true class (row).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, "fig_confusion")
    plt.close(fig)

    # ---------------------------------------------------------------- paired differences (forest)
    lb = (1 - PB[:, 4]) >= 0.5
    la = np.isin(pats["A-published"].values, ["NO_LEUKEMIA", "NO_CELLS"], invert=True)
    pb, pa = pats["B-MIL"].values.astype(str), pats["A-published"].values.astype(str)

    def diff(ix):
        hb, ha = M.headline(y5[ix], pb[ix], lb[ix]), M.headline(y5[ix], pa[ix], la[ix])
        return {k: hb[k] - ha[k] for k in hb}
    D = M.bootstrap(diff, len(pats), a.boot)
    keys = [("5c_macro_f1", "5-class macro-F1"), ("5c_bal_acc", "5-class balanced accuracy"),
            ("5c_acc", "5-class accuracy"), ("3c_bal_acc", "3-class balanced accuracy"),
            ("3c_acc", "3-class accuracy"), ("leuk_sens", "Leukemia sensitivity"),
            ("leuk_spec", "Leukemia specificity")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for i, (k, lab) in enumerate(keys[::-1]):
        v, lo, hi = D[k]
        c = C_B if lo > 0 else (C_A if hi < 0 else INK2)
        ax.plot([lo, hi], [i, i], c=c, lw=2, solid_capstyle="round")
        ax.plot(v, i, "o", ms=6, mfc=c, mec="white", mew=1.2)
        ax.text(1.03, i, f"{v:+.3f} ({lo:+.3f} to {hi:+.3f})", va="center", fontsize=7, color=INK,
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0, c=INK2, lw=0.8)
    ax.set_yticks(range(len(keys)), [lab for _, lab in keys[::-1]])
    ax.set(xlabel="Difference, Model B − Model A (published rule)", xlim=(-0.25, 0.4))
    ax.grid(axis="y", visible=False)
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(-0.01, 1.02, "← favours Model A", transform=tr, fontsize=7, color=INK2, ha="right")
    ax.text(0.01, 1.02, "favours Model B →", transform=tr, fontsize=7, color=INK2, ha="left")
    ax.text(1.03, 1.02, "Difference (95% CI)", transform=ax.transAxes, fontsize=7, color=INK2, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_differences")
    plt.close(fig)

    # ---------------------------------------------------------------- field count
    fc = pd.read_csv(tab / "field_count.csv")
    look = {"B-meanprob": (C_B, "-", "o"), "B-published": (C_B, ":", "s"),
            "A-meanprob": (C_A, "--", "o"), "A-published": (C_A, ":", "s")}
    lab = {"B-meanprob": "Model B, mean of fields", "B-published": "Model B, published rule",
           "A-meanprob": "Model A, mean of fields", "A-published": "Model A, published rule"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (col, yl) in zip(axs, (("macro_f1", "5-class macro-F1"), ("bal_acc3", "3-class balanced accuracy"))):
        for m, g in fc.groupby("method", sort=False):
            c, ls, mk = look.get(m, (INK2, "-", "o"))
            ax.plot(g.k, g[col], c=c, ls=ls, marker=mk, ms=4.5, mec="white", mew=0.8, lw=1.6, label=lab.get(m, m))
        ax.set(xscale="log", xlabel="Fields per patient", ylabel=yl, ylim=(0, 1))
        ax.set_xticks(sorted(fc.k.unique()), [str(k) for k in sorted(fc.k.unique())])
        ax.minorticks_off()
    axs[0].legend(loc="lower right")
    fig.text(0.01, -0.02, "Published rule: leukemia only if more than 3 fields are leukemic, so with 1–3 fields "
                          "every patient is called 'no leukemia'.", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig_fields")
    plt.close(fig)

    # ---------------------------------------------------------------- calibration + deferral
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], ls=":", c=INK2, lw=0.8)
    n_bins = 6   # equal-count bins (~23 patients each): stable with small samples
    for m, c, ls in (("A-meanprob", C_A, "--"), ("B-MIL", C_B, "-")):
        P = PB if m == "B-MIL" else PA
        conf = P.max(1)
        ok = conf == conf
        correct = P.argmax(1) == np.array([CLASSES.index(v) for v in y5])
        order = np.argsort(conf[ok])
        groups = np.array_split(order, n_bins)
        xs = [conf[ok][g].mean() for g in groups]
        ys = [correct[ok][g].mean() for g in groups]
        ns = [len(g) for g in groups]
        b = M.brier(P, y5)
        ax.plot(xs, ys, c=c, ls=ls, lw=1.6, label=f"{NAMES[m]} (Brier {b:.2f})")
        ax.scatter(xs, ys, s=[8 + 1.5 * n for n in ns], c=c, edgecolors="white", linewidths=0.8, zorder=3)
    ax.text(0.03, 0.97, f"{n_bins} bins with equal patient counts", transform=ax.transAxes, fontsize=6.5,
            color=INK2, va="top")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Predicted confidence (top class)", ylabel="Observed accuracy",
           title="Calibration", aspect="equal")
    ax.legend(loc="lower right")
    ax = axs[1]
    defer = [0, 5, 10, 15, 20, 25, 30]
    for m, P, c, ls in (("A-meanprob", PA, C_A, "--"), ("B-MIL", PB, C_B, "-")):
        order = np.argsort(P.max(1))
        pred = np.array([CLASSES[i] for i in P.argmax(1)])
        acc = [M.label_metrics(list(y5[order[int(round(len(y5) * x / 100)):]]),
                               list(pred[order[int(round(len(y5) * x / 100)):]]), CLASSES)["bal_acc"] for x in defer]
        ax.plot(defer, acc, c=c, ls=ls, lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8, label=NAMES[m])
    ax.set(xlabel="Least-confident patients referred to a hematologist (%)", ylabel="5-class balanced accuracy",
           title="Deferral", ylim=(0.5, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_calibration")
    plt.close(fig)

    txt = "\n".join(S) + "\n"
    (out / "figure_stats.md").write_text(txt, encoding="utf-8")
    shr = a.root / "inventory_out" / "shareable"
    shutil.copy(out / "figure_stats.md", shr / f"figure_stats_{a.scheme}.md")
    print(txt + f"\nFigures (PNG 300 dpi + PDF): {out}\nShareable: the figures (aggregate only) and {shr / f'figure_stats_{a.scheme}.md'}")


if __name__ == "__main__":
    main()
