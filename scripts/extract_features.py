"""LeukoBloom Model B, step 1: find every white cell and extract DinoBloom features.

For each included image (folds.csv):
  1. white-balance the field (removes camera colour casts),
  2. detect white cells (leukobloom.cells, detector v2) using ONE red-cell size per camera,
     estimated once from 40 fields, so crops have the same physical size on every camera,
  3. crop each cell (3 red-cell diameters, 224x224 px),
  4. embed every crop with DinoBloom (frozen) and save, per image:
       features/<model>/<image_id>.npz   feats (n_cells x dim, float16), x, y, area_rbc,
                                         touches_border, clump, score
Images already done are skipped, so the run can be stopped and restarted at any time.

Outputs stay on the laptop (features are derived from patient images; never upload them):
  Leukobloom\\features\\dinobloom_b\\IMG00001.npz ...
  Leukobloom\\inventory_out\\shareable\\features_summary.txt   (counts only; send to Claude)

Usage:
  py -3.12 scripts\\extract_features.py                 # DinoBloom-B (main model)
  py -3.12 scripts\\extract_features.py --model s       # DinoBloom-S (sensitivity check)
  py -3.12 scripts\\extract_features.py --limit 50      # quick test on 50 images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.cells import (DETECTOR_VERSION, crop_cells, detect_cells,  # noqa: E402
                              estimate_rbc_diameter, white_balance_image)

MODELS = {"b": "vit_base_patch14_224.dinobloom", "s": "vit_small_patch14_224.dinobloom"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_RBC = {}


def _init(rbc_by_cam):
    _RBC.update(rbc_by_cam)


def detect_and_crop(job):
    """Worker: one image -> (image_id, crops uint8 n x 224 x 224 x 3, cell table)."""
    image_id, path, cam = job
    try:
        im = white_balance_image(Image.open(path))
        rbc = _RBC[cam]
        cells, _ = detect_cells(im, rbc)
        crops = crop_cells(im, cells, rbc, out_px=224)
        arr = np.stack([np.asarray(c) for c in crops]) if crops else np.zeros((0, 224, 224, 3), np.uint8)
        meta = {k: np.array([getattr(c, k) for c in cells]) for k in
                ("x", "y", "area_rbc", "touches_border", "clump", "score")}
        return image_id, arr, meta, ""
    except Exception as e:  # unreadable image etc.
        return image_id, None, None, f"{type(e).__name__}: {e}"


def load_model(name: str, allow_random: bool):
    import timm
    try:
        m = timm.create_model(f"hf-hub:1aurent/{name}", pretrained=True)
        return m.eval(), True
    except Exception as e:
        if not allow_random:
            raise SystemExit(f"Could not download DinoBloom weights ({type(e).__name__}: {e}).\n"
                             "Check the internet connection and run again.")
        print(f"WARNING: random weights (test mode only): {type(e).__name__}")
        arch = name.replace(".dinobloom", "").replace("patch14_224", "patch14_dinov2")
        return timm.create_model(arch, pretrained=False, img_size=224).eval(), False


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    root0 = home / "Downloads" / "Leukobloom"
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=root0)
    ap.add_argument("--model", choices=list(MODELS), default="b")
    ap.add_argument("--workers", type=int, default=4, help="processes for detection/cropping")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (testing)")
    ap.add_argument("--allow-random-weights", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    import torch
    shr = a.root / "inventory_out" / "shareable"
    folds = pd.read_csv(shr / "folds.csv")
    key = pd.read_csv(a.root / "inventory_out" / "PRIVATE" / "image_key.csv")
    d = folds.merge(key, on="image_id")
    d = d[d.include.astype(str).str.lower() == "true"].sort_values("image_id")
    if a.limit:
        per_cam = max(1, a.limit // d.capture.nunique())
        d = d[d.groupby("capture").cumcount() < per_cam]

    out = a.root / "features" / f"dinobloom_{a.model}"
    out.mkdir(parents=True, exist_ok=True)

    # one red-cell size per camera (fixed, saved for reproducibility)
    scale_file = a.root / "features" / "rbc_px_per_camera.json"
    rbc = json.loads(scale_file.read_text()) if scale_file.exists() else {}
    for cam, g in d.groupby("capture"):
        if cam in rbc:
            continue
        vals = []
        for rel in g.sample(min(40, len(g)), random_state=7).relative_path:
            dm, n = estimate_rbc_diameter(Image.open(a.root / "data" / rel))
            if n >= 8:
                vals.append(dm)
        if not vals:  # e.g. a handful of odd-sized images: scale from the 1600-px camera by width
            ref = rbc.get("C1600", 100.0)
            w = Image.open(a.root / "data" / g.relative_path.iloc[0]).width
            vals = [ref * w / 1600]
        rbc[cam] = float(np.median(vals))
        print(f"{cam}: red cell = {rbc[cam]:.1f} px (from {len(vals)} fields)")
    scale_file.write_text(json.dumps(rbc, indent=1))

    todo = [(r.image_id, str(a.root / "data" / r.relative_path), r.capture)
            for r in d.itertuples() if not (out / f"{r.image_id}.npz").exists()]
    print(f"Images: {len(d)} included, {len(d) - len(todo)} already done, {len(todo)} to do")
    if not todo:
        write_summary(a, d, out, shr)
        return

    torch.set_num_threads(max(1, (os.cpu_count() or 8) - a.workers))
    model, pretrained = load_model(MODELS[a.model], a.allow_random_weights)
    print(f"DinoBloom-{a.model.upper()} loaded ({'pretrained' if pretrained else 'RANDOM - test only'}), "
          f"{torch.get_num_threads()} threads; detection in {a.workers} processes")

    t0, n_img, n_cells, errors = time.time(), 0, 0, []
    with Pool(a.workers, initializer=_init, initargs=(rbc,)) as pool:
        for image_id, crops, meta, err in pool.imap(detect_and_crop, todo, chunksize=2):
            if err:
                errors.append(f"{image_id}: {err}")
                continue
            feats = []
            with torch.inference_mode():
                for i in range(0, len(crops), a.batch):
                    x = (crops[i:i + a.batch].astype(np.float32) / 255.0 - MEAN) / STD
                    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
                    feats.append(model(x).numpy())
            dim = model.num_features
            f = np.concatenate(feats).astype(np.float16) if feats else np.zeros((0, dim), np.float16)
            np.savez_compressed(out / f"{image_id}.npz", feats=f, pretrained=pretrained,
                                detector=DETECTOR_VERSION, **meta)
            n_img += 1
            n_cells += len(f)
            if n_img % 50 == 0 or n_img == len(todo):
                el = time.time() - t0
                eta = el / n_img * (len(todo) - n_img)
                print(f"  {n_img}/{len(todo)} images, {n_cells} cells  "
                      f"({el/60:.0f} min elapsed, ~{eta/60:.0f} min left)", flush=True)
    if errors:
        (shr / "features_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"{len(errors)} images could not be processed (features_errors.txt)")
    write_summary(a, d, out, shr)


def write_summary(a, d, out, shr):
    rows = []
    for r in d.itertuples():
        p = out / f"{r.image_id}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rows.append((r.capture, r.label5, len(z["feats"]), int(z["touches_border"].sum()) if len(z["feats"]) else 0,
                     bool(z["pretrained"])))
    t = pd.DataFrame(rows, columns=["camera", "label5", "cells", "border", "pretrained"])
    L = [f"Features: DinoBloom-{a.model.upper()}  ({out.name}), detector {DETECTOR_VERSION}",
         f"Images done: {len(t)} of {len(d)}   cells: {t.cells.sum()}   (touching edge: {t.border.sum()})",
         f"Pretrained weights: {'yes' if t.pretrained.all() else 'NO - test run'}",
         f"Fields with 0 cells: {(t.cells == 0).sum()}", "",
         "Mean cells per field:",
         t.pivot_table(index="camera", columns="label5", values="cells", aggfunc="mean").round(1).to_string(), "",
         "Total cells:",
         t.pivot_table(index="camera", columns="label5", values="cells", aggfunc="sum", margins=True).to_string()]
    (shr / f"features_summary_{a.model}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nSend to Claude: {shr / f'features_summary_{a.model}.txt'}")


if __name__ == "__main__":
    main()
