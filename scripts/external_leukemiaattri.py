"""LeukoBloom: cross-site robustness test on the public LeukemiaAttri dataset (frozen models, no training).

Data: LeukemiaAttri (Rehman et al., MICCAI 2024), 100x subsets only: H_100X_C1, H_100X_C2, L_100X_C1,
L_100X_C2 (H/L = high-/low-cost microscope, C1/C2 = camera). Each subset is analysed on its own.
Labels: diagnosis from the file name {patient}_{field}_1000_{DX}.png; APML counted as AML; no normal controls,
so only leukemia patients (a "no leukemia" prediction is an error). Patient = subset + diagnosis + patient
number; the dataset's own train/test split is ignored (nothing is trained here) and both parts are used.

Fixed before any result was seen (4 Nov 2026):
  * Images are stored as 640x640 squares squeezed from wider photos. The width is restored per subset from
    the red-cell shape (median height/width of red cells, snapped to 1, 4:3 or 16:9); measured values saved.
  * Scale: red-cell diameter measured without colour rules (estimate_rbc_diameter_any_stain), expert
    white-cell boxes excluded; one value per patient (median of its fields), subset median as fallback.
  * Model B: one cell per EXPERT box (the dataset's annotations; our detector is not used), crop of 3 red-cell
    diameters at 224 px from the white-balanced, width-restored field, as in training; cells touching the
    image edge left out (as in training). Frozen DinoBloom-B, then the 15 frozen CV MIL models
    (5 folds x 3 seeds) averaged. Field probabilities = each field as its own bag.
  * Model A: field white-balanced, width restored, resized so a red cell is 28 px, 448 crops covering it,
    each available frozen CV fold model applied and the probabilities averaged (fold models listed in the
    summary). Patient result by the published rule (T = 3) and by mean probability.
  * Ensemble of the CV fold models (no new training on all data), so the external test uses exactly the
    models whose CV results are reported.

Outputs (Leukobloom\\results\\external_leukemiaattri\\; public data, but kept out of git like all results):
  index.csv            one row per field: subset, patient, diagnosis, boxes, red-cell size
  scale.json           aspect and red-cell size per subset
  fields_B.csv, patients_B.csv, fields_A_fold<k>.csv, patients_A.csv
  qc\\                  pictures: boxes, crops and the scale used, 2 fields per subset (check these first)
  summary.md           also copied to inventory_out\\shareable\\external_leukemiaattri.md
Resumable: features (features\\leukemiaattri_dinobloom_b\\) and Model A field results are kept; rerun to continue.

Usage:
  py -3.12 scripts\\external_leukemiaattri.py --qc            (index + scale + QC pictures only; minutes)
  py -3.12 scripts\\external_leukemiaattri.py --model b       (Model B; about 2 h)
  py -3.12 scripts\\external_leukemiaattri.py --model a       (Model A, all fold models found; several h)
  py -3.12 scripts\\external_leukemiaattri.py --summary       (rebuild summary from saved results)
  add --limit 2 for a quick test (2 patients per subset)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from leukobloom.cells import (Cell, crop_cells, estimate_rbc_diameter_any_stain,  # noqa: E402
                              red_cell_aspect, white_balance_image)
from leukobloom.metrics import bootstrap, confusion, label_metrics  # noqa: E402
from leukobloom.rules import CLASSES, THREE, apply_rule  # noqa: E402

SUBSETS = ["H_100X_C1", "H_100X_C2", "L_100X_C1", "L_100X_C2"]
DX5 = {"ALL": "ALL", "AML": "AML", "APML": "AML", "APL": "AML", "CLL": "CLL", "CML": "CML"}
LABELS4 = ["ALL", "AML", "CLL", "CML"]
ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "16:9": 16 / 9}
TARGET_RBC_A = 28.0
NAME_RE = re.compile(r"^(\d+)_(\d+)_1000_([A-Za-z]+)\.png$", re.I)
EDGE_PX = 2


# ------------------------------------------------------------------------------ index and labels
def find_subset_dirs(ds: Path):
    """All copies of each 100x subset across the export parts."""
    out = defaultdict(list)
    for p in ds.rglob("*"):
        if p.is_dir() and p.name in SUBSETS:
            out[p.name].append(p)
    return out


def read_labels(dirs):
    """Expert boxes per (split, file_name) from COCO json (loose or zipped); YOLO txt as fallback."""
    coco, yolo = {}, {}

    def add_coco(j, split):
        names = {im["id"]: im["file_name"] for im in j["images"]}
        boxes = defaultdict(list)
        for an in j["annotations"]:
            boxes[an["image_id"]].append(an["bbox"])
        for iid, fn in names.items():
            coco.setdefault((split, Path(fn).name), boxes.get(iid, []))

    def add_yolo(text, split, fn):
        b = []
        for line in text.splitlines():
            v = line.split()
            if len(v) >= 5:
                cx, cy, w, h = map(float, v[1:5])
                b.append([(cx - w / 2) * 640, (cy - h / 2) * 640, w * 640, h * 640])
        yolo.setdefault((split, fn), b)

    for d in dirs:
        for jf in d.glob("json_labels/*.json"):
            add_coco(json.loads(jf.read_text(encoding="utf-8")), jf.stem.lower())
        for zf in d.glob("*.zip"):
            with zipfile.ZipFile(zf) as z:
                for n in z.namelist():
                    parts = n.lower().split("/")
                    if n.endswith(".json") and "json_labels" in parts:
                        add_coco(json.loads(z.read(n)), Path(n).stem.lower())
                    elif n.endswith(".txt") and ("train" in parts or "test" in parts):
                        split = "train" if "train" in parts else "test"
                        add_yolo(z.read(n).decode("utf-8", "ignore"), split, Path(n).stem + ".png")
    return coco, yolo


def build_index(ds: Path, limit: int):
    rows = []
    for sub, dirs in sorted(find_subset_dirs(ds).items()):
        coco, yolo = read_labels(dirs)
        seen = set()
        for d in dirs:
            for img_dir in [p for p in d.iterdir() if p.is_dir() and p.name.lower() == "images"]:
                for split_dir in [p for p in img_dir.iterdir() if p.is_dir()]:
                    split = split_dir.name.lower()
                    for f in sorted(split_dir.glob("*.png")):
                        m = NAME_RE.match(f.name)
                        if not m or (split, f.name) in seen or m.group(3).upper() not in DX5:
                            continue
                        seen.add((split, f.name))
                        pid, field, dx = m.group(1), m.group(2), m.group(3).upper()
                        if (split, f.name) in coco:
                            boxes, src = coco[(split, f.name)], "json"
                        elif (split, f.name) in yolo:
                            boxes, src = yolo[(split, f.name)], "txt"
                        else:
                            boxes, src = None, "none"
                        rows.append(dict(field_id=f"{sub}_{split}_{f.stem}", subset=sub, split=split,
                                         patient=f"{sub}/{dx}_{pid}", dx_file=dx, label5=DX5[dx],
                                         path=str(f.relative_to(ds)), label_src=src,
                                         n_boxes=(np.nan if boxes is None else len(boxes)),
                                         boxes=json.dumps(boxes) if boxes is not None else ""))
    idx = pd.DataFrame(rows)
    if idx.empty:
        raise SystemExit(f"No 100x images found under {ds}")
    if limit:
        keep = idx.groupby("subset").patient.apply(lambda s: list(dict.fromkeys(s))[:limit]).explode()
        idx = idx[idx.patient.isin(set(keep))]
    return idx.sort_values(["subset", "patient", "field_id"]).reset_index(drop=True)


# ------------------------------------------------------------------------------ geometry and scale
def load_field(ds, r, k):
    """Width-restored, white-balanced field and boxes in its coordinates."""
    im = Image.open(ds / r.path).convert("RGB")
    w, h = im.size
    boxes = json.loads(r.boxes) if r.boxes else []
    if abs(k - 1) > 1e-6:
        im = im.resize((round(w * k), h), Image.Resampling.LANCZOS)
        boxes = [[x * k, y, bw * k, bh] for x, y, bw, bh in boxes]
    edge = [x <= EDGE_PX or y <= EDGE_PX or x + bw >= im.width - EDGE_PX or y + bh >= im.height - EDGE_PX
            for x, y, bw, bh in boxes]
    return white_balance_image(im), boxes, edge


def measure_scale(ds, idx, overrides):
    scale = {}
    for sub, g in idx.groupby("subset"):
        smp = g.sample(min(40, len(g)), random_state=7)
        q = []
        for r in smp.itertuples():
            boxes = json.loads(r.boxes) if r.boxes else []
            a, n = red_cell_aspect(Image.open(ds / r.path).convert("RGB"), boxes)
            if n >= 5:
                q.append(a)
        measured = float(np.median(q)) if q else float("nan")
        if sub in overrides:
            name, k = "override", overrides[sub]
        elif np.isnan(measured):
            name, k = "1:1 (not measurable)", 1.0
        else:
            name = min(ASPECTS, key=lambda n_: abs(np.log(measured) - np.log(ASPECTS[n_])))
            k = ASPECTS[name]
        scale[sub] = dict(aspect_measured=measured, n_fields=len(q), width_factor=k, aspect=name)
        print(f"  {sub}: red-cell height/width {measured:.2f} -> width x{k:.3f} ({name})", flush=True)
    rbc, nrbc = [], []
    for r in idx.itertuples():
        im, boxes, _ = load_field(ds, r, scale[r.subset]["width_factor"])
        d, n = estimate_rbc_diameter_any_stain(im, boxes)
        rbc.append(d)
        nrbc.append(n)
    idx["rbc_field"], idx["rbc_n"] = rbc, nrbc
    ok = idx.rbc_n >= 8
    sub_med = idx[ok].groupby("subset").rbc_field.median()
    pat_med = idx[ok].groupby("patient").rbc_field.agg(["median", "size"])
    idx["rbc_px"] = [pat_med.loc[p, "median"] if p in pat_med.index and pat_med.loc[p, "size"] >= 3
                     else sub_med.get(s, np.nan) for p, s in zip(idx.patient, idx.subset)]
    idx["rbc_source"] = ["patient" if p in pat_med.index and pat_med.loc[p, "size"] >= 3 else "subset"
                         for p in idx.patient]
    for sub in scale:
        g = idx[idx.subset == sub]
        scale[sub].update(rbc_subset_median=float(sub_med.get(sub, np.nan)),
                          rbc_patient_iqr=[float(v) for v in g.groupby("patient").rbc_px.first().quantile([.25, .75])],
                          fields_measured=int((g.rbc_n >= 8).sum()), fields=int(len(g)))
    if idx.rbc_px.isna().any():
        raise SystemExit("Red-cell size could not be measured for some subsets: " +
                         ", ".join(idx[idx.rbc_px.isna()].subset.unique()))
    return idx, scale


def qc_pictures(ds, idx, scale, out):
    out.mkdir(parents=True, exist_ok=True)
    for sub, g in idx[idx.n_boxes > 0].groupby("subset"):
        for r in g.sample(min(2, len(g)), random_state=3).itertuples():
            im, boxes, edge = load_field(ds, r, scale[sub]["width_factor"])
            vis = im.copy()
            d = ImageDraw.Draw(vis)
            for (x, y, w, h), e in zip(boxes, edge):
                d.rectangle([x, y, x + w, y + h], outline=(255, 140, 0) if e else (0, 200, 0), width=3)
            s = r.rbc_px
            d.ellipse([10, 10, 10 + s, 10 + s], outline=(255, 0, 0), width=3)   # one red-cell diameter
            d.rectangle([10, vis.height - 20, 10 + 3 * s, vis.height - 10], fill=(255, 0, 0))  # crop side
            cells = [Cell(x=x + w / 2, y=y + h / 2, area_rbc=0, touches_border=e, clump=False, score=0)
                     for (x, y, w, h), e in zip(boxes, edge)]
            crops = crop_cells(im, cells, s, out_px=160)
            W = max(vis.width, 160 * max(1, len(crops)))
            canvas = Image.new("RGB", (W, vis.height + 190), (255, 255, 255))
            canvas.paste(vis, (0, 0))
            for i, c in enumerate(crops):
                canvas.paste(c, (160 * i, vis.height + 25))
            ImageDraw.Draw(canvas).text(
                (5, vis.height + 5), f"{r.field_id}  width x{scale[sub]['width_factor']:.2f}  red cell "
                f"{s:.0f}px (circle; bar = crop side)  orange = edge, left out", fill=(0, 0, 0))
            canvas.save(out / f"{r.field_id}.jpg", quality=90)


# ------------------------------------------------------------------------------ Model B
def run_model_b(a, ds, idx, scale, res):
    import torch
    from extract_features import MEAN, MODELS, STD, load_model
    from leukobloom.mil import GatedAttentionMIL
    fdir = a.root / "features" / "leukemiaattri_dinobloom_b"
    fdir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in idx.itertuples() if not (fdir / f"{r.field_id}.npz").exists()]
    print(f"Model B features: {len(idx)} fields, {len(idx) - len(todo)} done, {len(todo)} to do", flush=True)
    if todo:
        model, pretrained = load_model(MODELS["b"], a.allow_random_weights)
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            im, boxes, edge = load_field(ds, r, scale[r.subset]["width_factor"])
            cells = [Cell(x=x + w / 2, y=y + h / 2, area_rbc=0, touches_border=e, clump=False, score=0)
                     for (x, y, w, h), e in zip(boxes, edge)]
            crops = crop_cells(im, cells, r.rbc_px, out_px=224)
            if crops:
                x = (np.stack([np.asarray(c) for c in crops]).astype(np.float32) / 255.0 - MEAN) / STD
                with torch.inference_mode():
                    f = model(torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()).numpy().astype(np.float16)
            else:
                f = np.zeros((0, model.num_features), np.float16)
            np.savez_compressed(fdir / f"{r.field_id}.npz", feats=f, touches_border=np.array(edge, bool),
                                x=np.array([c.x for c in cells]), y=np.array([c.y for c in cells]),
                                pretrained=pretrained)
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} fields ({el / 60:.0f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                      flush=True)

    mdir = a.root / "results" / "model_b_b_cv" / "models"
    paths = sorted(mdir.glob("fold*_seed*.pt"))
    if not paths:
        raise SystemExit(f"No Model B fold models in {mdir}")
    models = []
    for p in paths:
        m = GatedAttentionMIL(768)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.eval())
    feats = {}
    for r in idx.itertuples():
        z = np.load(fdir / f"{r.field_id}.npz")
        feats[r.field_id] = z["feats"].astype(np.float32)[~z["touches_border"].astype(bool)] \
            if len(z["feats"]) else z["feats"].astype(np.float32)

    @torch.inference_mode()
    def pred(x):
        if len(x) == 0:
            return np.full(5, np.nan)
        x = torch.from_numpy(x)
        return np.mean([torch.softmax(m(x)[0], 0).numpy() for m in models], 0)

    frows = [dict(field_id=r.field_id, subset=r.subset, patient=r.patient, label5=r.label5,
                  n_cells=len(feats[r.field_id]), **dict(zip([f"p_{c}" for c in CLASSES], pred(feats[r.field_id]))))
             for r in idx.itertuples()]
    prows = []
    for (sub, pat, lab), g in idx.groupby(["subset", "patient", "label5"]):
        x = np.concatenate([feats[i] for i in g.field_id]) if len(g) else np.zeros((0, 768), np.float32)
        prows.append(dict(subset=sub, patient=pat, label5=lab, fields=len(g), n_cells=len(x),
                          **dict(zip([f"p_{c}" for c in CLASSES], pred(x)))))
    pd.DataFrame(frows).to_csv(res / "fields_B.csv", index=False)
    pd.DataFrame(prows).to_csv(res / "patients_B.csv", index=False)
    (res / "models_B.txt").write_text("\n".join(p.name for p in paths))
    print(f"Model B: {len(prows)} patients, {len(paths)} MIL models averaged", flush=True)


# ------------------------------------------------------------------------------ Model A
def run_model_a(a, ds, idx, scale, res):
    import torch
    import torchvision
    from train_densenet import CFG, test_crops, to_tensor
    mdir = a.root / "results" / "model_a_cv" / "models"
    folds = sorted(int(p.stem[4:]) for p in mdir.glob("fold*.pt") if re.fullmatch(r"fold\d+", p.stem))
    if not folds:
        raise SystemExit(f"No finished Model A fold models in {mdir}")
    print(f"Model A fold models found: {folds}", flush=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 1))
    for k in folds:
        out = res / f"fields_A_fold{k}.csv"
        done = set(pd.read_csv(out).field_id) if out.exists() else set()
        todo = [r for r in idx.itertuples() if r.field_id not in done]
        print(f"  fold {k}: {len(done)} fields done, {len(todo)} to do", flush=True)
        if not todo:
            continue
        m = torchvision.models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, len(CLASSES))
        m.load_state_dict(torch.load(mdir / f"fold{k}.pt", map_location="cpu"))
        m.eval()
        t0, buf = time.time(), []
        for i, r in enumerate(todo, 1):
            im, _, _ = load_field(ds, r, scale[r.subset]["width_factor"])
            f = TARGET_RBC_A / r.rbc_px
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.Resampling.LANCZOS)
            b = io.BytesIO()
            im.save(b, "JPEG", quality=95)          # as the training cache (prepare_fields.py)
            im = Image.open(b).convert("RGB")
            crops = test_crops(im, CFG["img"])
            with torch.inference_mode():
                p = torch.softmax(m(torch.stack([to_tensor(c) for c in crops])), 1).mean(0).numpy()
            buf.append(dict(field_id=r.field_id, n_crops=len(crops), **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
            if len(buf) >= 100 or i == len(todo):
                pd.DataFrame(buf).to_csv(out, mode="a", header=not out.exists(), index=False)
                buf = []
                el = time.time() - t0
                print(f"    {i}/{len(todo)} ({el / 60:.0f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                      flush=True)


def model_a_fields(idx, res):
    files = sorted(res.glob("fields_A_fold*.csv"))
    if not files:
        return None, []
    P, used = [], []
    for f in files:
        d = pd.read_csv(f).drop_duplicates("field_id").set_index("field_id")
        if set(idx.field_id) <= set(d.index):
            P.append(d.loc[idx.field_id, [f"p_{c}" for c in CLASSES]].values)
            used.append(f.stem.replace("fields_A_", ""))
    if not P:
        return None, []
    return pd.DataFrame(np.mean(P, 0), columns=[f"p_{c}" for c in CLASSES]).assign(
        field_id=idx.field_id.values, subset=idx.subset.values, patient=idx.patient.values,
        label5=idx.label5.values), used


# ------------------------------------------------------------------------------ summary
def patient_table(fields, pats_b=None):
    """Patient predictions for every method available."""
    cols = [f"p_{c}" for c in CLASSES]
    rows = []
    for (sub, pat, lab), g in fields.groupby(["subset", "patient", "label5"]):
        P = g[cols].values.astype(float)
        rows.append(dict(subset=sub, patient=pat, label5=lab,
                         **{f"pred_{r}": apply_rule(P, r) for r in ["published", "meanprob"]}))
    t = pd.DataFrame(rows)
    if pats_b is not None:
        P = pats_b[cols].values.astype(float)
        mil = np.where(np.isnan(P).any(1), "NO_CELLS", [CLASSES[i] for i in np.nan_to_num(P).argmax(1)])
        t = t.merge(pats_b[["patient"]].assign(pred_mil=mil), on="patient", how="left")
    return t


def method_block(y, p):
    lm = label_metrics(list(y), list(p), LABELS4)
    y3, p3 = [THREE[v] for v in y], [THREE.get(v, v) for v in p]
    acute_chronic = np.mean([a == b for a, b in zip(y3, p3)])
    leuk = np.mean([v not in ("NO_LEUKEMIA", "NO_CELLS") for v in p])
    return lm, acute_chronic, leuk


def fmt_ci(t):
    return f"{t[0]:.2f} ({t[1]:.2f}–{t[2]:.2f})"


def write_summary(a, idx, scale, res):
    L = ["# LeukemiaAttri external robustness test (frozen LeukoBloom models)", "",
         "Public dataset, 100x subsets, each analysed separately. Leukemia patients only (no normal controls); "
         "APML counted as AML. Model B uses the expert white-cell boxes. Patient-level results; 95% CIs by "
         "bootstrap over patients (2,000 resamples). Small subsets: read CIs, not point estimates.", ""]
    L += ["## Data and scale", "",
          "| Subset | Patients (ALL/AML/CLL/CML) | Fields | Fields with labels | Expert boxes/field | "
          "Red-cell h/w measured | Width restored | Red cell px (subset median) |", "|---|---|---|---|---|---|---|---|"]
    for sub, g in idx.groupby("subset"):
        pc = g.groupby("patient").label5.first().value_counts()
        s = scale[sub]
        L.append(f"| {sub} | {g.patient.nunique()} ({'/'.join(str(pc.get(c, 0)) for c in LABELS4)}) | {len(g)} | "
                 f"{int((g.label_src != 'none').sum())} | {g.n_boxes.mean():.1f} | {s['aspect_measured']:.2f} | "
                 f"x{s['width_factor']:.3f} ({s['aspect']}) | {s['rbc_subset_median']:.1f} |")
    dx_files = idx.groupby("dx_file").patient.nunique()
    L += ["", "Diagnoses in file names (patients): " + ", ".join(f"{k} {v}" for k, v in dx_files.items()), ""]

    methods = {}
    if (res / "fields_B.csv").exists():
        fb, pb = pd.read_csv(res / "fields_B.csv"), pd.read_csv(res / "patients_B.csv")
        tb = patient_table(fb, pb)
        methods["B-MIL"] = tb[["subset", "patient", "label5", "pred_mil"]].rename(columns={"pred_mil": "pred"})
        methods["B-published"] = tb[["subset", "patient", "label5", "pred_published"]].rename(
            columns={"pred_published": "pred"})
        L += [f"Model B: {len((res / 'models_B.txt').read_text().split())} MIL models averaged; "
              f"cells used per field (edge cells left out): "
              + ", ".join(f"{s} {g.n_cells.mean():.1f}" for s, g in fb.groupby("subset"))
              + f"; fields with no usable cell: {int((fb.n_cells == 0).sum())}.", ""]
    fa, used = model_a_fields(idx, res)
    if fa is not None:
        ta = patient_table(fa)
        methods["A-published"] = ta[["subset", "patient", "label5", "pred_published"]].rename(
            columns={"pred_published": "pred"})
        methods["A-meanprob"] = ta[["subset", "patient", "label5", "pred_meanprob"]].rename(
            columns={"pred_meanprob": "pred"})
        L += [f"Model A: fold models averaged: {', '.join(used)}.", ""]
        fa.to_csv(res / "fields_A_ensemble.csv", index=False)
    if not methods:
        L += ["No model results yet."]
    else:
        L += ["## Patient-level results", "",
              "Accuracy = correct diagnosis among ALL/AML/CLL/CML; balanced accuracy and macro-F1 over the "
              "classes present; acute/chronic = correct 3-class group; called leukemia = share not predicted "
              "'no leukemia' (sensitivity for leukemia).", "",
              "| Subset | Method | n | Accuracy | Balanced accuracy | Macro-F1 | Acute/chronic | Called leukemia |",
              "|---|---|---|---|---|---|---|---|"]
        allp = []
        for name, t in methods.items():
            allp.append(t.assign(method=name))
        allp = pd.concat(allp)
        allp.to_csv(res / "patient_predictions.csv", index=False)
        for sub in sorted(allp.subset.unique()):
            for name in methods:
                t = allp[(allp.subset == sub) & (allp.method == name)]
                y, p = t.label5.values, t.pred.values

                def fn(ix, y=y, p=p):
                    lm, ac, le = method_block(y[ix], p[ix])
                    return dict(acc=lm["acc"], bal=lm["bal_acc"], f1=lm["macro_f1"], ac=ac, le=le)
                ci = bootstrap(fn, len(y), B=a.boot)
                L.append(f"| {sub} | {name} | {len(y)} | {fmt_ci(ci['acc'])} | {fmt_ci(ci['bal'])} | "
                         f"{fmt_ci(ci['f1'])} | {fmt_ci(ci['ac'])} | {fmt_ci(ci['le'])} |")
        L += ["", "## Confusion matrices (rows true, columns predicted)", ""]
        for sub in sorted(allp.subset.unique()):
            for name in methods:
                t = allp[(allp.subset == sub) & (allp.method == name)]
                m, cols = confusion(list(t.label5), list(t.pred), LABELS4)
                L += [f"**{sub} — {name}**", "", "| True \\ Pred | " + " | ".join(cols) + " |",
                      "|---|" + "---|" * len(cols)]
                L += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in zip(LABELS4, m)
                      if row.sum()]
                L.append("")
        # same patient under the four microscope/camera conditions (patient key = diagnosis + number)
        allp["case"] = allp.patient.str.split("/").str[1]
        L += ["## Consistency across microscopes and cameras", "",
              "Same patients imaged under each condition. Consistent = identical prediction in every subset the "
              "patient appears in; majority = most frequent prediction across subsets (ties: none counted correct).",
              "", "| Method | Patients | Consistent | Majority correct |", "|---|---|---|---|"]
        for name in methods:
            t = allp[allp.method == name]
            cons, maj = [], []
            for _, g in t.groupby("case"):
                vc = g.pred.value_counts()
                cons.append(len(vc) == 1)
                top = vc.index[0] if (len(vc) == 1 or vc.iloc[0] > vc.iloc[1]) else None
                maj.append(top == g.label5.iloc[0])
            L.append(f"| {name} | {len(cons)} | {np.mean(cons):.2f} | {np.mean(maj):.2f} |")
        L.append("")
        L += ["## Patient predictions (codes are the dataset's own patient numbers)", "",
              "| Patient | True | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
        wide = allp.pivot_table(index=["patient", "label5"], columns="method", values="pred", aggfunc="first")
        for (pat, lab), row in wide.iterrows():
            L.append(f"| {pat} | {lab} | " + " | ".join(str(row.get(m_, "")) for m_ in methods) + " |")
    txt = "\n".join(L) + "\n"
    (res / "summary.md").write_text(txt, encoding="utf-8")
    shr = a.root / "inventory_out" / "shareable"
    shr.mkdir(parents=True, exist_ok=True)
    shutil.copy(res / "summary.md", shr / "external_leukemiaattri.md")
    print(f"\nSummary: {res / 'summary.md'}\nSend to Claude: {shr / 'external_leukemiaattri.md'}")


# ------------------------------------------------------------------------------ main
def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--dataset", type=Path, default=None, help="default: <root>\\external\\LeukemiaAttri")
    ap.add_argument("--model", choices=["a", "b"], default=None)
    ap.add_argument("--qc", action="store_true", help="index, scale and QC pictures only")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary")
    ap.add_argument("--limit", type=int, default=0, help="patients per subset (quick test)")
    ap.add_argument("--aspect", default="", help="override width factor, e.g. H_100X_C1=1.778")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--allow-random-weights", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    ds = a.dataset or a.root / "external" / "LeukemiaAttri"
    res = a.root / "results" / ("external_leukemiaattri" + (f"_limit{a.limit}" if a.limit else ""))
    res.mkdir(parents=True, exist_ok=True)

    if (res / "index.csv").exists() and (res / "scale.json").exists():
        idx = pd.read_csv(res / "index.csv", keep_default_na=False, na_values=[""])
        idx["boxes"] = idx.boxes.fillna("")
        scale = json.loads((res / "scale.json").read_text())
        print(f"Index loaded: {len(idx)} fields, {idx.patient.nunique()} patients")
    else:
        print("Indexing 100x images and expert labels ...", flush=True)
        idx = build_index(ds, a.limit)
        print(f"  {len(idx)} fields, {idx.patient.nunique()} patients; labels: "
              + ", ".join(f"{k} {v}" for k, v in idx.label_src.value_counts().items()), flush=True)
        print("Measuring image shape and red-cell size ...", flush=True)
        overrides = {k: float(v) for k, v in (s.split("=") for s in a.aspect.split(",") if s)}
        idx, scale = measure_scale(ds, idx, overrides)
        idx.to_csv(res / "index.csv", index=False)
        (res / "scale.json").write_text(json.dumps(scale, indent=1))
        qc_pictures(ds, idx, scale, res / "qc")
        print(f"QC pictures: {res / 'qc'}")
    if a.model == "b":
        run_model_b(a, ds, idx, scale, res)
    elif a.model == "a":
        run_model_a(a, ds, idx, scale, res)
    write_summary(a, idx, scale, res)


if __name__ == "__main__":
    main()
