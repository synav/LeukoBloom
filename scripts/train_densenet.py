"""LeukoBloom Model A: LeukoAI-style whole-field DenseNet-121, one fold per run.

Pre-specified settings (fixed before any result; do not tune on test folds):
  input   prepared fields (common scale, red cell ~28 px, white-balanced; prepare_fields.py)
          training: random 448x448 crop (padded with background where the field is smaller),
          flips, 90-degree rotations, mild colour jitter; testing: 448 crops covering the field, averaged
  model   DenseNet-121, ImageNet weights, new 5-class head, all layers trained
  train   10 epochs, batch 8, AdamW lr 1e-4, weight decay 1e-4, one-cycle schedule,
          class-weighted cross-entropy (field level), seed 0
  label   every field takes its sample's diagnosis (decision D1)
  patient mean of field probabilities -> 5-class prediction; 3-class from summed probabilities;
          other aggregation rules applied later to fields.csv by the report script

Resumable: a checkpoint is written every --save-min minutes and at the end of every epoch; rerun the same
command to continue. Each fold writes its own files, so folds can run on different nights.

Outputs (Leukobloom\\results\\model_a_<scheme>\\, codes only, no images):
  models\\fold<k>.pt           final weights          models\\fold<k>_ckpt.pt   resume checkpoint
  fields_fold<k>.csv          test-field probabilities
  log_fold<k>.txt             loss per epoch
  patients.csv + summary.txt  written once all folds are done (summary also copied to shareable)

Usage:
  py -3.12 scripts\\train_densenet.py --fold 0          (one fold; about 8 h on the study laptop)
  py -3.12 scripts\\train_densenet.py --fold 0 --smoke  (2-minute test of the whole pipeline)
  py -3.12 scripts\\train_densenet.py --scheme external
  py -3.12 scripts\\train_densenet.py --summary         (only rebuild patients.csv / summary.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.mil import CLASSES  # noqa: E402

THREE = {"ALL": "ACUTE", "AML": "ACUTE", "CLL": "CHRONIC", "CML": "CHRONIC", "NO_LEUKEMIA": "NO_LEUKEMIA"}
C3 = ["ACUTE", "CHRONIC", "NO_LEUKEMIA"]
CFG = dict(img=448, epochs=10, batch=8, lr=1e-4, wd=1e-4, seed=0, cache="fields_rbc28")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BG = (240, 240, 240)  # white-balanced background


# ------------------------------------------------------------------------------ data
def pad_to(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    if w >= size and h >= size:
        return im
    canvas = Image.new("RGB", (max(w, size), max(h, size)), BG)
    canvas.paste(im, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def to_tensor(im: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD


def test_crops(im: Image.Image, size: int):
    """448 crops covering the whole field (1 for a 1600-camera field, 2 for wider fields)."""
    im = pad_to(im, size)
    w, h = im.size
    xs = [0] if w == size else list(np.linspace(0, w - size, int(np.ceil(w / size))).round().astype(int))
    ys = [0] if h == size else list(np.linspace(0, h - size, int(np.ceil(h / size))).round().astype(int))
    return [im.crop((x, y, x + size, y + size)) for y in ys for x in xs]


class FieldSet(torch.utils.data.Dataset):
    def __init__(self, paths, labels, size):
        self.paths, self.labels, self.size = list(paths), list(labels), size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import random
        im = pad_to(Image.open(self.paths[i]).convert("RGB"), self.size)
        w, h = im.size
        x, y = random.randint(0, w - self.size), random.randint(0, h - self.size)
        im = im.crop((x, y, x + self.size, y + self.size))
        if random.random() < 0.5:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im = im.rotate(90 * random.randint(0, 3))
        t = to_tensor(im)
        # mild colour jitter: brightness/contrast +-10%, per-channel gain +-3%
        t = t * (1 + 0.1 * (2 * torch.rand(1) - 1)) + 0.1 * (2 * torch.rand(1) - 1)
        t = t * (1 + 0.03 * (2 * torch.rand(3, 1, 1) - 1))
        return t, self.labels[i]


class EpochOrder(torch.utils.data.Sampler):
    """Fixed shuffled order per epoch (seeded), starting at `start`: lets a run resume mid-epoch."""

    def __init__(self, n, seed, epoch, start):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        self.order = torch.randperm(n, generator=g)[start:].tolist()

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


# ------------------------------------------------------------------------------ model
def build_model(allow_random: bool):
    import torchvision
    try:
        m = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        pre = True
    except Exception as e:
        if not allow_random:
            raise SystemExit(f"Could not download ImageNet weights ({type(e).__name__}). Check internet and rerun.")
        print("WARNING: random weights (test mode only)")
        m, pre = torchvision.models.densenet121(weights=None), False
    m.classifier = nn.Linear(m.classifier.in_features, len(CLASSES))
    return m, pre


# ------------------------------------------------------------------------------ train / test one split
def run_split(a, name, train_df, test_df, out, cache):
    ckpt_path, final_path = out / "models" / f"{name}_ckpt.pt", out / "models" / f"{name}.pt"
    log_path = out / f"log_{name}.txt"
    cfg = dict(CFG)
    if a.smoke:
        cfg.update(epochs=1, batch=4)
        train_df = train_df.groupby("label5").head(4)
        test_df = test_df.head(6)

    torch.manual_seed(cfg["seed"])
    model, pretrained = build_model(a.allow_random_weights)
    y = np.array([CLASSES.index(l) for l in train_df.label5])
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    ds = FieldSet([cache / f"{i}.jpg" for i in train_df.image_id], y.tolist(), cfg["img"])
    steps_per_epoch = int(np.ceil(len(ds) / cfg["batch"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    total = cfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total,
                                                pct_start=max(0.1, 2.0 / total))
    epoch, step_in_epoch = 0, 0
    if final_path.exists() and not a.smoke:
        model.load_state_dict(torch.load(final_path, map_location="cpu"))
        epoch = cfg["epochs"]
        print(f"{name}: already trained, testing only")
    elif ckpt_path.exists() and not a.smoke:
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        epoch, step_in_epoch, pretrained = c["epoch"], c["step"], c["pretrained"]
        print(f"{name}: resuming at epoch {epoch + 1}, batch {step_in_epoch + 1}/{steps_per_epoch}")

    def save_ckpt(ep, st):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep, step=st, pretrained=pretrained), ckpt_path)

    print(f"{name}: {len(train_df)} training fields, {len(test_df)} test fields "
          f"({test_df.patient_code.nunique()} patients); {cfg['epochs']} epochs x {steps_per_epoch} batches; "
          f"{'ImageNet' if pretrained else 'RANDOM'} start; {torch.get_num_threads()} threads")
    t0, done0 = time.time(), epoch * steps_per_epoch + step_in_epoch
    last_save = time.time()
    while epoch < cfg["epochs"]:
        model.train()
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], num_workers=a.workers,
                                         sampler=EpochOrder(len(ds), cfg["seed"], epoch, step_in_epoch * cfg["batch"]),
                                         persistent_workers=False)
        run_loss, n = 0.0, 0
        for x, t in dl:
            loss = F.cross_entropy(model(x), t, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step_in_epoch += 1
            run_loss += loss.item(); n += 1
            done = epoch * steps_per_epoch + step_in_epoch
            if step_in_epoch % 50 == 0 or step_in_epoch == steps_per_epoch:
                el = time.time() - t0
                left = el / max(done - done0, 1) * (cfg["epochs"] * steps_per_epoch - done)
                print(f"  epoch {epoch + 1}/{cfg['epochs']} batch {step_in_epoch}/{steps_per_epoch}  "
                      f"loss {run_loss / n:.3f}  (~{left / 3600:.1f} h left)", flush=True)
            if time.time() - last_save > a.save_min * 60 and not a.smoke:
                save_ckpt(epoch, step_in_epoch); last_save = time.time()
        if not a.smoke:
            with open(log_path, "a") as fh:
                fh.write(f"epoch {epoch + 1} loss {run_loss / max(n, 1):.4f} {time.strftime('%Y-%m-%d %H:%M')}\n")
        epoch, step_in_epoch = epoch + 1, 0
        if not a.smoke:
            save_ckpt(epoch, 0); last_save = time.time()
    if not a.smoke:
        torch.save(model.state_dict(), final_path)

    # ---- test
    model.eval()
    rows = []
    with torch.inference_mode():
        for r in test_df.itertuples():
            crops = test_crops(Image.open(cache / f"{r.image_id}.jpg").convert("RGB"), cfg["img"])
            p = torch.softmax(model(torch.stack([to_tensor(c) for c in crops])), 1).mean(0).numpy()
            rows.append(dict(image_id=r.image_id, patient_code=r.patient_code, fold=r.fold, set=r.set,
                             capture=r.capture, label5=r.label5, n_crops=len(crops),
                             **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
    f = pd.DataFrame(rows)
    f.to_csv(out / f"fields_{name}{'_smoke' if a.smoke else ''}.csv", index=False)
    print(f"{name}: tested {len(f)} fields")
    return pretrained


# ------------------------------------------------------------------------------ summary
def summarise(a, out, names, shr):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_mil import summarise as mil_summary  # same metrics as Model B
    parts = [out / f"fields_{n}.csv" for n in names]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        print(f"Summary waits for: {', '.join(missing)}")
        return
    f = pd.concat([pd.read_csv(p) for p in parts])
    pc = [f"p_{c}" for c in CLASSES]
    pat = f.groupby("patient_code").agg(fold=("fold", "first"), set=("set", "first"), label5=("label5", "first"),
                                        n_fields=("image_id", "size"), **{c: (c, "mean") for c in pc}).reset_index()
    P = pat[pc].values
    pat["pred5"] = [CLASSES[i] for i in P.argmax(1)]
    p3 = np.stack([P[:, 0] + P[:, 1], P[:, 2] + P[:, 3], P[:, 4]], 1)
    pat["pred3"] = [C3[i] for i in p3.argmax(1)]
    pat["label3"] = pat.label5.map(THREE)
    pat.to_csv(out / "patients.csv", index=False)
    txt = mil_summary(pat, f"Model A (LeukoAI-style DenseNet-121, whole field), scheme={a.scheme}, "
                           f"patient = mean of field probabilities, epochs={CFG['epochs']}")
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    (shr / f"model_a_{a.scheme}_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt + f"\n\nSend to Claude: inventory_out\\shareable\\model_a_{a.scheme}_summary.txt")


# ------------------------------------------------------------------------------ main
def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--scheme", choices=["cv", "external"], default="cv")
    ap.add_argument("--fold", type=int, default=None, help="fold to train (cv); omit with --summary")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--save-min", type=float, default=20.0, help="checkpoint interval in minutes")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline (nothing kept)")
    ap.add_argument("--summary", action="store_true", help="only build patients.csv and summary")
    ap.add_argument("--allow-random-weights", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - a.workers))

    shr = a.root / "inventory_out" / "shareable"
    out = a.root / "results" / f"model_a_{a.scheme}"
    (out / "models").mkdir(parents=True, exist_ok=True)
    cache = a.root / "cache" / CFG["cache"]
    d = pd.read_csv(shr / "folds.csv")
    d = d[d.include.astype(str).str.lower() == "true"].copy()
    d["patient_code"] = d.patient_code.fillna("")
    names = [f"fold{k}" for k in sorted(d[d.role == "cv"].fold.unique())] if a.scheme == "cv" else ["external"]

    if a.summary:
        summarise(a, out, names, shr)
        return
    missing = [i for i in d.image_id if not (cache / f"{i}.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} fields not prepared yet: run scripts\\prepare_fields.py first")

    if a.scheme == "cv":
        if a.fold is None:
            raise SystemExit("Give --fold 0..4 (one fold per run), or --summary")
        k = a.fold
        tr = d[(d.role != "cv") | (d.fold != k)]
        te = d[(d.role == "cv") & (d.fold == k)]
        name = f"fold{k}"
    else:
        tr = d[d.set == "training"]
        te = d[(d.set == "validation") & (d.role == "cv")]
        name = "external"
    (out / "config.json").write_text(json.dumps(dict(CFG, model="densenet121", scheme=a.scheme,
                                                     torch=torch.__version__), indent=1))
    run_split(a, name, tr, te, out, cache)
    if not a.smoke:
        summarise(a, out, names, shr)


if __name__ == "__main__":
    main()
