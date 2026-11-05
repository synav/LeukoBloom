"""LeukoBloom Figure 1: study design (no data; a drawn schematic).

Journal style: white background, black text, thin grey outlines, Arial-type font; colour only as light tints matching
the model colours used in Figures 2-5 (Model A orange, Model B blue).
Usage:  py -3.12 scripts\\fig_design.py [--out results\\figures_cv]   -> fig_design.png (300 dpi) + fig_design.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, EDGE, GREY = "#000000", "#4d4d4d", "#f2f2f2"
A_EDGE, A_FILL = "#c2410c", "#fde8dc"      # Model A: orange family (Figures 2-5 use #eb6834)
B_EDGE, B_FILL = "#1d5fae", "#dce9f8"      # Model B: blue family (Figures 2-5 use #2a78d6)


def box(ax, x, y, w, h, lines, fill="white", edge=EDGE, lw=0.8, bold_first=False, size=7.2, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=lw,
                                ls=(0, (4, 2)) if dashed else "-"))
    n = len(lines)
    step = 2.55 if size < 7.5 else 2.9
    y0 = y + h / 2 + (n - 1) * step / 2
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y0 - i * step, t, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if (bold_first and i == 0) else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, side="right"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + (1.2 if side == "right" else -1.2), (y1 + y2) / 2, text, fontsize=6.3,
                color="#333333", ha="left" if side == "right" else "right", va="center", style="italic")


def draw(out: Path):
    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.axis("off")

    # data and splits
    box(ax, 12, 91, 76, 10, ["Single-centre development data", "7,123 manually captured fields · 1,000× oil · "
        "May-Grünwald-Giemsa · 3 cameras", "137 patients tested: ALL 11, AML 41, CLL 14, CML 10, "
        "no leukemia 61"], fill=GREY, bold_first=True)
    arrow(ax, 50, 91, 50, 87.5)
    box(ax, 22, 80.5, 56, 7, ["Patient-grouped 5-fold cross-validation", "stratified by diagnosis and camera; "
        "identical folds for both models"], bold_first=True)
    # split to the two models
    ax.plot([50, 50], [80.5, 78.5], color=EDGE, lw=0.8)
    ax.plot([26, 74], [78.5, 78.5], color=EDGE, lw=0.8)
    arrow(ax, 26, 78.5, 26, 76)
    arrow(ax, 74, 78.5, 74, 76)

    # Model A column
    xa, xb, w = 6, 54, 40
    box(ax, xa, 69, w, 7, ["Model A — whole field", "LeukoAI-style"], fill=A_FILL, edge=A_EDGE, lw=1.1,
        bold_first=True, size=7.6)
    box(ax, xa, 59.5, w, 7, ["Field: white balance, rescaled to", "common scale (red cell ≈ 28 px)"])
    box(ax, xa, 50, w, 7, ["DenseNet-121, ImageNet start,", "fully fine-tuned (448 px crops)"])
    box(ax, xa, 40.5, w, 7, ["Field probabilities (5 classes)", "→ published LeukoAI rule"])
    for y1, y2 in ((69, 66.5), (59.5, 57), (50, 47.5)):
        arrow(ax, xa + w / 2, y1, xa + w / 2, y2)

    # Model B column
    box(ax, xb, 69, w, 7, ["Model B — cell level", "foundation model + MIL"], fill=B_FILL, edge=B_EDGE, lw=1.1,
        bold_first=True, size=7.6)
    box(ax, xb, 59.5, w, 7, ["White-cell detection; crops of 3", "red-cell diameters (224 px)"])
    box(ax, xb, 50, w, 7, ["DinoBloom-B, frozen", "768 features per cell"])
    box(ax, xb, 40.5, w, 7, ["Gated-attention MIL over all", "cells of a patient (3 seeds)"])
    for y1, y2 in ((69, 66.5), (59.5, 57), (50, 47.5)):
        arrow(ax, xb + w / 2, y1, xb + w / 2, y2)

    # outputs and comparison
    box(ax, xa, 32, w, 5.5, ["Patient diagnosis"], fill=A_FILL, edge=A_EDGE)
    box(ax, xb, 32, w, 5.5, ["Patient diagnosis"], fill=B_FILL, edge=B_EDGE)
    arrow(ax, xa + w / 2, 40.5, xa + w / 2, 37.5)
    arrow(ax, xb + w / 2, 40.5, xb + w / 2, 37.5)
    ax.plot([26, 26], [32, 30], color=EDGE, lw=0.8)
    ax.plot([74, 74], [32, 30], color=EDGE, lw=0.8)
    ax.plot([26, 74], [30, 30], color=EDGE, lw=0.8)
    arrow(ax, 50, 30, 50, 28)
    box(ax, 14, 20, 72, 8, ["Paired comparison (pre-specified, TRIPOD+AI)", "macro-F1, balanced accuracy, AUROC, "
        "McNemar · aggregation rules · fields per patient", "calibration and referral · "
        "exploratory model combination"], bold_first=True)

    # external test band
    arrow(ax, 50, 20, 50, 16.5, text="models frozen")
    ax.add_patch(FancyBboxPatch((3, 1), 94, 15.5, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=EDGE,
                                lw=0.8, ls=(0, (4, 2))))
    ax.text(5, 14.3, "External test — LeukemiaAttri (public): 47 leukemia patients, each imaged with 2 microscopes "
            "× 2 cameras; expert cell boxes", fontsize=7, fontweight="bold", color=INK, va="center")
    box(ax, 6, 3, 40, 8.5, ["Frozen models", "Model A · Model B · Model A then B", "per microscope and camera"],
        fill=GREY)
    box(ax, 54, 3, 40, 8.5, ["Post hoc: MIL step retrained", "patient-grouped 5-fold; DinoBloom unchanged",
        "also high-cost → low-cost microscope"], fill=GREY)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_design.png", dpi=300, facecolor="white")
    fig.savefig(out / "fig_design.pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out / 'fig_design.png'} and .pdf")


if __name__ == "__main__":
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=home / "Downloads" / "Leukobloom" / "results" / "figures_cv")
    draw(ap.parse_args().out)
