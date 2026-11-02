"""LeukoBloom: check white-cell detection on YOUR images, per camera, before full extraction.

For each camera (image size) it:
  * estimates the red-cell diameter from up to 40 fields (sets the physical crop size),
  * runs the detector on a few fields per diagnosis and saves overlay pictures
    (green box = cell, orange = touches the image edge) plus a gallery of crops.

Pictures are saved LOCALLY in  Leukobloom\\inventory_out\\cell_check\\  for you to look at;
they contain patient images, so do not upload them. The printed summary has numbers only.

Usage:  py -3.12 scripts\\check_cells.py
Needs:  scikit-image scipy pandas pillow   (py -3.12 -m pip install scikit-image scipy)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.cells import CROP_RBC, DETECTOR_VERSION, crop_cells, detect_cells, estimate_rbc_diameter  # noqa: E402


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    root = home / "Downloads" / "Leukobloom"
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--per-class", type=int, default=2, help="overlay fields per diagnosis per camera")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--camera", default="", help="only this camera, e.g. C1920 (same fields as a full run)")
    a = ap.parse_args()

    out_dir = a.root / "inventory_out" / "shareable"
    folds = pd.read_csv(out_dir / "folds.csv")
    key = pd.read_csv(a.root / "inventory_out" / "PRIVATE" / "image_key.csv")
    d = folds.merge(key, on="image_id")
    d = d[d.include.astype(str).str.lower() == "true"]
    check = a.root / "inventory_out" / "cell_check"
    check.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    lines = []
    for cam, g in sorted(d.groupby("capture")):
        if cam == "OTHER":
            continue
        if a.camera and cam != a.camera:
            for _ in g.label5.unique():   # keep the random draws identical -> same fields as a full run
                rng.integers(1e9)
            continue
        # 1. scale from up to 40 fields
        sample = g.sample(min(40, len(g)), random_state=a.seed)
        diam = []
        for rel in sample.relative_path:
            dm, n = estimate_rbc_diameter(Image.open(a.root / "data" / rel))
            if n >= 8:
                diam.append(dm)
        rbc = float(np.median(diam)) if diam else None
        if rbc is None:
            lines.append(f"{cam}: could not estimate red-cell size (too few isolated red cells)")
            print(lines[-1])
            continue

        # 2. overlays + crops on a few fields per diagnosis
        n_cells, n_fields, crops, per_field = [], 0, [], []
        tiles = []
        for lab, gl in sorted(g.groupby("label5")):
            for _, r in gl.sample(min(a.per_class, len(gl)), random_state=int(rng.integers(1e9))).iterrows():
                im = Image.open(a.root / "data" / r.relative_path).convert("RGB")
                cells, _ = detect_cells(im, rbc)
                n_cells.append(len(cells))
                per_field.append(f"   picture {len(tiles) + 1}: {lab:11s} {r.image_id}  cells={len(cells)}")
                n_fields += 1
                crops += crop_cells(im, cells[:6], rbc, out_px=112)
                dr = ImageDraw.Draw(im)
                s = CROP_RBC * rbc / 2
                for c in cells:
                    col = (230, 120, 0) if c.touches_border else (0, 170, 0)
                    dr.rectangle([c.x - s, c.y - s, c.x + s, c.y + s], outline=col, width=max(3, im.width // 300))
                dr.text((15, 15), f"{len(tiles) + 1}. {lab}  {r.image_id}  cells={len(cells)}", fill=(0, 0, 0))
                tiles.append(im.resize((640, round(640 * im.height / im.width))))
        th = max(t.height for t in tiles)
        grid = Image.new("RGB", (640 * 2, th * ((len(tiles) + 1) // 2)), "white")
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * 640, (i // 2) * th))
        grid.save(check / f"{cam}_overlays_{DETECTOR_VERSION}.jpg", quality=85)
        if crops:
            gal = Image.new("RGB", (112 * 12, 112 * ((len(crops) + 11) // 12)), "white")
            for i, c in enumerate(crops):
                gal.paste(c, ((i % 12) * 112, (i // 12) * 112))
            gal.save(check / f"{cam}_crops_{DETECTOR_VERSION}.jpg", quality=85)
        um = 7.5 / rbc
        lines.append(f"{cam}: red cell ~{rbc:.0f} px (~{um:.3f} um/px if RBC = 7.5 um; from {len(diam)} fields) | "
                     f"cells/field mean {np.mean(n_cells):.1f} (min {min(n_cells)}, max {max(n_cells)}) over {n_fields} fields")
        print(lines[-1])
        lines += per_field
        print("\n".join(per_field))

    lines.insert(0, f"detector {DETECTOR_VERSION}")
    (out_dir / "cell_check_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPictures: {check}   (local only)\nSummary : {out_dir / 'cell_check_summary.txt'}  (numbers only; send to Claude)")


if __name__ == "__main__":
    main()
