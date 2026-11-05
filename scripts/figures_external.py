"""LeukoBloom: figure for the LeukemiaAttri external test (frozen vs combined vs locally retrained).

Reads saved patient predictions only (nothing is rerun):
  results\\combined\\patients_leukemiaattri.csv            frozen Model A (published rule), frozen Model B,
                                                          frozen A -> B sequence  (combine_models.py)
  results\\leukemiaattri_cv\\patients.csv                   Model B with the MIL step retrained locally
  results\\leukemiaattri_cv_cross-microscope\\patients.csv  retrained on the high-cost microscope only
Draws, per microscope/camera subset, 4-class accuracy and acute-vs-chronic accuracy with 95% bootstrap CIs, plus
the share of patients given the same diagnosis under every condition. PNG (300 dpi) + PDF in
results\\figures_external\\, and a numbers table (external_figure_stats.md, also copied to shareable).

Usage:  py -3.12 scripts\\figures_external.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import INK, INK2, save, style  # noqa: E402
from leukobloom.rules import THREE  # noqa: E402

SUBSETS = ["H_100X_C1", "H_100X_C2", "L_100X_C1", "L_100X_C2"]
SUB_LABEL = {"H_100X_C1": "Cam 1", "H_100X_C2": "Cam 2", "L_100X_C1": "Cam 1", "L_100X_C2": "Cam 2"}
SHORT = {"A": "A frozen", "B": "B frozen", "SEQ": "A → B frozen", "XM": "B high-cost", "LOCAL": "B local"}
# (key, label, colour, marker) — fixed order; colours validated (dataviz validator), shapes as second encoding
METHODS = [("A", "Model A, frozen (published rule)", "#eb6834", "s"),
           ("B", "Model B, frozen", "#2a78d6", "o"),
           ("SEQ", "Frozen A → B (field, then cells)", "#6250d6", "D"),
           ("XM", "Model B, MIL retrained on high-cost microscope", "#eda100", "v"),
           ("LOCAL", "Model B, MIL retrained locally", "#1baf7a", "^")]


def ci(values, B, seed=2026):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    d = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return v.mean(), *np.percentile(d, [2.5, 97.5])


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    res = a.root / "results"
    out = res / "figures_external"
    out.mkdir(parents=True, exist_ok=True)

    comb = pd.read_csv(res / "combined" / "patients_leukemiaattri.csv")
    rows = []
    for key, col in (("A", "Model A (A-published)"), ("B", "Model B (B-MIL)"),
                     ("SEQ", "1. Sequence (A field → B cells)")):
        rows.append(comb[["subset", "patient", "label5"]].assign(method=key, pred=comb[col].astype(str)))
    for key, folder in (("LOCAL", "leukemiaattri_cv"), ("XM", "leukemiaattri_cv_cross-microscope")):
        f = res / folder / "patients.csv"
        if f.exists():
            d = pd.read_csv(f)
            rows.append(d[["subset", "patient", "label5", "pred"]].assign(method=key))
        else:
            print(f"Not found (skipped): {f}")
    P = pd.concat(rows, ignore_index=True)
    P["case"] = P.patient.str.split("/").str[1]
    P["ok"] = P.pred == P.label5
    P["ok3"] = [THREE[y] == THREE.get(p, p) for y, p in zip(P.label5, P.pred)]
    present = [m for m in METHODS if m[0] in set(P.method)]

    stats = []
    for m in present:
        for s in SUBSETS:
            g = P[(P.method == m[0]) & (P.subset == s)]
            if len(g):
                stats.append(dict(method=m[0], subset=s, n=len(g), acc=ci(g.ok, a.boot), acc3=ci(g.ok3, a.boot)))
    S = pd.DataFrame(stats)
    cons = {m[0]: P[P.method == m[0]].groupby("case").pred.nunique().eq(1).mean() for m in present}
    cons_n = {m[0]: P[P.method == m[0]].subset.nunique() for m in present}

    fig = plt.figure(figsize=(8.0, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 2.2], wspace=0.42)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    k = len(present)
    off = np.linspace(-0.3, 0.3, k)
    for ax, col, title in ((axs[0], "acc", "Correct diagnosis (ALL / AML / CLL / CML)"),
                           (axs[1], "acc3", "Correct acute vs chronic")):
        for j, m in enumerate(present):
            g = S[S.method == m[0]].set_index("subset")
            xs = [SUBSETS.index(s) + off[j] for s in g.index]
            v = np.array([t for t in g[col]])
            ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=m[3], ms=5.5, color=m[2],
                        mec="white", mew=0.8, elinewidth=1.4, capsize=0, label=m[1])
        for x in (0.5, 1.5, 2.5):
            ax.axvline(x, color="#e4e3df", lw=0.6)
        ax.axvline(1.5, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(range(4), [SUB_LABEL[s] for s in SUBSETS], fontsize=7)
        for x, t in ((0.5, "High-cost\nmicroscope"), (2.5, "Low-cost\nmicroscope")):
            ax.text(x, -0.09, t, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=INK)
        ax.set(ylim=(0, 1.06), title=title, xlim=(-0.5, 3.5))
        ax.grid(axis="x", visible=False)
        if ax is axs[0]:
            ax.set_ylabel("Share of 47 patients")
    ax = axs[2]
    for i, m in enumerate(present):
        ax.barh(i, cons[m[0]], color=m[2], height=0.62, edgecolor="white", linewidth=2)
        ax.text(cons[m[0]] + 0.03, i, f"{cons[m[0]]:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(k), [SHORT[m[0]] for m in present], fontsize=7)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.15), title="Same diagnosis\nin every condition")
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0, 0.5, 1])
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), fontsize=7, handletextpad=0.3,
               columnspacing=1.2)
    fig.suptitle("External test on LeukemiaAttri (47 patients, 4 microscope/camera conditions)", x=0.01, y=1.06,
                 ha="left", fontsize=10, color=INK)
    fig.text(0.01, -0.25, "Points = share correct, bars = 95% bootstrap CI. Retrained models: patient-grouped 5-fold "
                          "cross-validation within LeukemiaAttri (DinoBloom frozen). High-cost-trained model tested on "
                          "low-cost subsets only.", fontsize=6.5, color=INK2)
    save(fig, out, "fig_external")
    plt.close(fig)

    names = {m[0]: m[1] for m in METHODS}
    L = ["# External test figure — numbers", "", "Share correct (95% bootstrap CI), 47 patients per subset.", "",
         "| Method | Subset | Correct diagnosis | Correct acute vs chronic |", "|---|---|---|---|"]
    for r in S.itertuples():
        L.append(f"| {names[r.method]} | {r.subset} | {r.acc[0]:.2f} ({r.acc[1]:.2f}–{r.acc[2]:.2f}) | "
                 f"{r.acc3[0]:.2f} ({r.acc3[1]:.2f}–{r.acc3[2]:.2f}) |")
    L += ["", "Same diagnosis for a patient in every condition tested: " +
          "; ".join(f"{names[k_]} {v:.2f} ({cons_n[k_]} conditions)" for k_, v in cons.items())]
    (out / "external_figure_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    shr = a.root / "inventory_out" / "shareable" / "external_figure_stats.md"
    shutil.copy(out / "external_figure_stats.md", shr)
    print("\n".join(L) + f"\n\nFigure: {out / 'fig_external.png'}\nShareable: the figure and {shr}")


if __name__ == "__main__":
    main()
