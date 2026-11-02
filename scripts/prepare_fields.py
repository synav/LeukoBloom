"""LeukoBloom Model A, step 1: prepare every field once at a common physical scale.

For each included image: white-balance (same as Model B), then resize so a red cell is TARGET_RBC_PX
pixels wide, using the per-camera red-cell size measured in step 4a (features\\rbc_px_per_camera.json).
TARGET_RBC_PX = 28 reproduces the published LeukoAI input scale (1600x1200 field padded to square and
resized to 448). Fields are saved as high-quality JPEG in  Leukobloom\\cache\\fields_rbc28\\  (local only).

Usage:  py -3.12 scripts\\prepare_fields.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.cells import white_balance_image  # noqa: E402

TARGET_RBC_PX = 28.0


def work(job):
    src, dst, factor = job
    try:
        im = white_balance_image(Image.open(src))
        w, h = im.size
        im = im.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.LANCZOS)
        im.save(dst, quality=95)
        return ""
    except Exception as e:
        return f"{Path(dst).stem}: {type(e).__name__}: {e}"


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    rbc = json.loads((a.root / "features" / "rbc_px_per_camera.json").read_text())
    folds = pd.read_csv(a.root / "inventory_out" / "shareable" / "folds.csv")
    key = pd.read_csv(a.root / "inventory_out" / "PRIVATE" / "image_key.csv")
    d = folds.merge(key, on="image_id")
    d = d[d.include.astype(str).str.lower() == "true"]
    out = a.root / "cache" / f"fields_rbc{int(TARGET_RBC_PX)}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(a.root / "data" / r.relative_path), str(out / f"{r.image_id}.jpg"), TARGET_RBC_PX / rbc[r.capture])
            for r in d.itertuples() if not (out / f"{r.image_id}.jpg").exists()]
    print(f"{len(d)} fields; {len(d) - len(jobs)} already prepared; {len(jobs)} to do "
          f"(scale factors: " + ", ".join(f"{k} x{TARGET_RBC_PX / v:.3f}" for k, v in rbc.items()) + ")")
    t0, errs = time.time(), []
    with Pool(a.workers) as pool:
        for i, e in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            if e:
                errs.append(e)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if errs:
        print(f"{len(errs)} failed:\n" + "\n".join(errs[:20]))
    sizes = pd.Series([Image.open(p).size for p in list(out.glob("*.jpg"))[:2000]]).value_counts().head(6)
    print("Most common prepared sizes (w, h):\n" + sizes.to_string())
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
