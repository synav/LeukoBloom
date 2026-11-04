"""LeukoBloom: cell gallery for hematologist review (explanation analysis; no retraining).

For each chosen patient it shows the cells Model B weighted most (attention), cropped from the original
fields exactly as the model saw them (white-balanced, same physical scale), with:
  rank, attention relative to an average cell (x1.0 = average; x10 = ten times the average weight),
  the field code, and whether the cell touched the image edge.
A strip of the LEAST weighted cells is added for contrast.

It also writes a numbers-only summary (cells detected per field for each listed patient, against the
median for correctly classified patients of the same diagnosis and camera), to check for missed cells.

Pictures contain patient images: they stay on the laptop (results\\report_cv\\gallery\\). The summary
is codes and counts only and can be shared.

Usage:
  py -3.12 scripts\\cell_gallery.py --misclassified           (all patients Model B got wrong)
  py -3.12 scripts\\cell_gallery.py --patients P0113,P0101     (any patients)
  py -3.12 scripts\\cell_gallery.py --misclassified --top 36
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.cells import Cell, crop_cells, white_balance_image  # noqa: E402
from leukobloom.rules import CLASSES  # noqa: E402

TILE, COLS = 168, 6


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--result", default="model_b_b_cv", help="results folder of the Model B run")
    ap.add_argument("--patients", default="", help="comma-separated patient codes")
    ap.add_argument("--misclassified", action="store_true")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--bottom", type=int, default=6)
    a = ap.parse_args()

    res = a.root / "results" / a.result
    cfg = json.loads((res / "config.json").read_text()) if (res / "config.json").exists() else {}
    keep_border = bool(cfg.get("keep_border", False))
    feat_dir = a.root / "features" / cfg.get("features", "dinobloom_b")
    pats = pd.read_csv(res / "patients.csv")
    P = pats[[f"p_{c}" for c in CLASSES]].values
    pats["pred"] = np.where(np.isnan(P).any(1), "NO_CELLS", [CLASSES[i] for i in np.nan_to_num(P).argmax(1)])
    pats["conf"] = np.nan_to_num(P).max(1)
    folds = pd.read_csv(a.root / "inventory_out" / "shareable" / "folds.csv")
    folds = folds[folds.include.astype(str).str.lower() == "true"]
    key = pd.read_csv(a.root / "inventory_out" / "PRIVATE" / "image_key.csv").set_index("image_id").relative_path
    rbc = json.loads((a.root / "features" / "rbc_px_per_camera.json").read_text())
    cam_of = folds.set_index("image_id").capture

    # cells per field (non-edge, as the model used them) for every patient: to check for missed cells
    def n_used(image_id):
        z = np.load(feat_dir / f"{image_id}.npz")
        return int(len(z["x"]) if keep_border else (~z["touches_border"].astype(bool)).sum()) if len(z["x"]) else 0
    cpf = (folds[folds.patient_code.isin(pats.patient_code)]
           .assign(n=lambda d: [n_used(i) for i in d.image_id])
           .groupby("patient_code").agg(cells_per_field=("n", "mean"), fields=("n", "size"),
                                        camera=("capture", lambda s: s.value_counts().index[0])))
    pats = pats.join(cpf, on="patient_code")

    if a.misclassified:
        chosen = pats[pats.pred != pats.label5]
    else:
        codes = [c.strip() for c in a.patients.split(",") if c.strip()]
        if not codes:
            raise SystemExit("Give --misclassified or --patients P0001,P0002")
        chosen = pats[pats.patient_code.isin(codes)]
    out = a.root / "results" / "gallery" / a.result
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"Cell gallery summary ({a.result}); cells per field = cells the model used (edge cells "
             f"{'included' if keep_border else 'excluded'}). Reference = median of correctly classified "
             "patients with the same diagnosis and camera.", "",
             "| Patient | True | Predicted | Confidence | Camera | Fields | Cells/field | Reference cells/field |",
             "|---|---|---|---|---|---|---|---|"]
    correct = pats[pats.pred == pats.label5]
    for r in chosen.sort_values(["label5", "patient_code"]).itertuples():
        ref = correct[(correct.label5 == r.label5) & (correct.camera == r.camera)].cells_per_field
        lines.append(f"| {r.patient_code} | {r.label5} | {r.pred} | {r.conf:.2f} | {r.camera} | {r.fields} | "
                     f"{r.cells_per_field:.1f} | {ref.median():.1f} (n={len(ref)}) |" if len(ref) else
                     f"| {r.patient_code} | {r.label5} | {r.pred} | {r.conf:.2f} | {r.camera} | {r.fields} | "
                     f"{r.cells_per_field:.1f} | — (no correct patient) |")

        # ---- gallery picture
        att_p = res / "attention" / f"{r.patient_code}.npz"
        if not att_p.exists():
            continue
        z = np.load(att_p, allow_pickle=True)
        img_ids, cell_idx, att = z["image_id"], z["cell"], z["attention"].astype(float)
        if len(att) == 0:
            continue
        order = np.argsort(-att)
        pick = list(order[:a.top]) + ([] if len(att) <= a.top else list(order[-a.bottom:]))
        cache, tiles = {}, []
        for k in pick:
            iid = str(img_ids[k])
            if iid not in cache:
                f = np.load(feat_dir / f"{iid}.npz")
                used = np.arange(len(f["x"])) if keep_border else np.flatnonzero(~f["touches_border"].astype(bool))
                cache[iid] = (white_balance_image(Image.open(a.root / "data" / key[iid])), f, used)
            im, f, used = cache[iid]
            j = int(used[int(cell_idx[k])])
            c = Cell(x=float(f["x"][j]), y=float(f["y"][j]), area_rbc=0, touches_border=bool(f["touches_border"][j]),
                     clump=False, score=0)
            crop = crop_cells(im, [c], rbc[cam_of[iid]], out_px=TILE)[0]
            d = ImageDraw.Draw(crop)
            rel = att[k] * len(att)
            rank = int(np.where(order == k)[0][0]) + 1
            d.rectangle([0, 0, TILE, 15], fill=(255, 255, 255))
            d.text((3, 2), f"#{rank}  x{rel:.1f}  {iid}{'  edge' if c.touches_border else ''}", fill=(0, 0, 0))
            tiles.append(crop)
        n_top = min(a.top, len(tiles))
        rows_top = int(np.ceil(n_top / COLS))
        rows_bot = int(np.ceil((len(tiles) - n_top) / COLS))
        H = 56 + rows_top * TILE + (28 + rows_bot * TILE if rows_bot else 0) + 8
        canvas = Image.new("RGB", (COLS * TILE, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), f"{r.patient_code}   true {r.label5}   predicted {r.pred} ({r.conf:.2f})   camera {r.camera}",
               fill=(0, 0, 0))
        d.text((8, 24), f"{len(att)} cells used from {r.fields} fields ({r.cells_per_field:.1f}/field).  "
                        f"Top {n_top} by attention" + (f"; last row: {len(tiles) - n_top} lowest" if rows_bot else ""),
               fill=(80, 80, 80))
        for i, t in enumerate(tiles[:n_top]):
            canvas.paste(t, ((i % COLS) * TILE, 48 + (i // COLS) * TILE))
        if rows_bot:
            y0 = 48 + rows_top * TILE + 20
            d.text((8, y0 - 16), "Lowest attention (for contrast):", fill=(80, 80, 80))
            for i, t in enumerate(tiles[n_top:]):
                canvas.paste(t, ((i % COLS) * TILE, y0 + (i // COLS) * TILE))
        canvas.save(out / f"{r.patient_code}_{r.label5}_as_{r.pred}.png")
        print(f"  {r.patient_code}: {r.label5} -> {r.pred}", flush=True)

    txt = "\n".join(lines)
    shr = a.root / "inventory_out" / "shareable" / f"gallery_summary_{a.result}.md"
    shr.write_text(txt, encoding="utf-8")
    print("\n" + txt + f"\n\nPictures (local only): {out}\nSend to Claude: {shr}")


if __name__ == "__main__":
    main()
