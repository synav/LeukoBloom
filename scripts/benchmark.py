"""LeukoBloom: CPU benchmark before committing to training runs.

Times, on THIS laptop:
  1. Model A: DenseNet-121 training steps at 448x448 (input size of the published LeukoAI),
     full fine-tuning and last-block-only.
  2. Model B: DinoBloom-S and DinoBloom-B feature extraction on 224x224 crops.
Then estimates the full 5-fold runs from the image counts in folds.csv.

Uses ~32 of your own training images, read locally; nothing leaves the laptop.
Weights are downloaded once (ImageNet DenseNet from PyTorch; DinoBloom from Hugging Face).

Usage:  py -3.12 scripts\\benchmark.py
Install first (CPU PyTorch):
  py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  py -3.12 -m pip install timm huggingface_hub pandas pillow
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_images(data: Path, n: int, seed: int = 0):
    files = [p for p in (data / "training").rglob("*") if p.suffix.lower() in IMG_EXT]
    random.Random(seed).shuffle(files)
    imgs = []
    for p in files:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if len(imgs) == n:
            break
    if not imgs:
        raise SystemExit(f"No images found under {data / 'training'}")
    return imgs


def to_tensor(imgs, size):
    import torchvision.transforms.functional as F
    return torch.stack([F.normalize(F.to_tensor(F.resize(im, [size, size])),
                                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for im in imgs])


def random_crops(imgs, n, size=224):
    """Stand-in for cell crops: random 224x224 patches (real crops are the same size)."""
    import torchvision.transforms.functional as F
    out = []
    rng = random.Random(1)
    while len(out) < n:
        im = imgs[rng.randrange(len(imgs))]
        w, h = im.size
        s = min(w, h) // 4                      # roughly one-cell-sized square
        x, y = rng.randrange(0, w - s), rng.randrange(0, h - s)
        out.append(F.resize(im.crop((x, y, x + s, y + s)), [size, size]))
    return out


def time_train(model, x, y, steps, params):
    opt = torch.optim.AdamW(params, lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for i in range(steps + 1):                 # first step = warm-up, not timed
        if i == 1:
            t0 = time.perf_counter()
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    return (time.perf_counter() - t0) / (steps * len(x))   # seconds per image


@torch.inference_mode()
def time_infer(model, x, reps):
    model.eval()
    model(x[:2])                                # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    return (time.perf_counter() - t0) / (reps * len(x))


def fmt_h(sec):
    return f"{sec/3600:.1f} h" if sec >= 3600 else f"{sec/60:.0f} min"


def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    root = home / "Downloads" / "Leukobloom"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--folds", type=Path, default=root / "inventory_out/shareable/folds.csv")
    ap.add_argument("--epochs", type=int, default=15, help="planned DenseNet epochs per fold")
    ap.add_argument("--cells-per-field", type=float, default=5.0, help="assumed white cells per field")
    ap.add_argument("--skip-dinobloom", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} CPU threads")

    # image counts for the estimates
    n_train_per_fold, n_fields = 5900, 7115
    if a.folds.exists():
        import pandas as pd
        f = pd.read_csv(a.folds)
        f = f[f.include.astype(str).str.lower() == "true"]
        n_fields = len(f)
        cv = f[f.fold >= 0]
        n_train_per_fold = int((f.fold == -1).sum() + len(cv) * 4 / 5)
    print(f"Fields: {n_fields}; training images per fold: ~{n_train_per_fold}\n")

    print("Loading 32 local training images ...")
    imgs = load_images(a.data, 32)
    lines = []

    # ---- Model A: DenseNet-121 -------------------------------------------------------
    import torchvision
    print("Model A: DenseNet-121 (downloading ImageNet weights on first run) ...")
    try:
        m = torchvision.models.densenet121(weights="IMAGENET1K_V1")
    except Exception as e:  # offline: same speed without the weights
        print(f"  (ImageNet weights not downloaded: {type(e).__name__}; timing with random weights)")
        m = torchvision.models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 5)
    for size, bs in ((448, 8), (224, 16)):
        x = to_tensor(imgs[:bs], size)
        y = torch.randint(0, 5, (bs,))
        for p in m.parameters():
            p.requires_grad = True
        full = time_train(m, x, y, 3, m.parameters())
        for p in m.parameters():
            p.requires_grad = False
        last = list(m.features.denseblock4.parameters()) + list(m.features.norm5.parameters()) + list(m.classifier.parameters())
        for p in last:
            p.requires_grad = True
        part = time_train(m, x, y, 3, last)
        for name, s in (("full fine-tuning", full), ("last block only", part)):
            epoch = s * n_train_per_fold
            total = epoch * a.epochs * 5
            lines.append(f"Model A {size}x{size} {name:17s}: {s:.2f} s/image | epoch {fmt_h(epoch)} | "
                         f"{a.epochs} epochs x 5 folds = {fmt_h(total)}")
            print("  " + lines[-1])

    # ---- Model B: DinoBloom ------------------------------------------------------------
    if not a.skip_dinobloom:
        import timm
        crops = random_crops(imgs, 64)
        n_cells = n_fields * a.cells_per_field
        for name in ("vit_small_patch14_224.dinobloom", "vit_base_patch14_224.dinobloom"):
            print(f"Model B: DinoBloom {name.split('_')[1]} (downloading weights on first run) ...")
            try:
                db = timm.create_model(f"hf-hub:1aurent/{name}", pretrained=True)
            except Exception as e:  # offline: same architecture, random weights, same speed
                print(f"  (DinoBloom weights not downloaded: {type(e).__name__}; timing with random weights)")
                arch = name.replace(".dinobloom", "").replace("patch14_224", "patch14_dinov2")
                db = timm.create_model(arch, pretrained=False, img_size=224)
            x = to_tensor(crops, 224)   # 224x224, ImageNet mean/std (DinoBloom's input)
            s = time_infer(db, x, 2)
            lines.append(f"Model B DinoBloom-{name.split('_')[1][0].upper()} features: {s*1000:.0f} ms/cell | "
                         f"~{n_cells:,.0f} cells (assumed {a.cells_per_field}/field) = {fmt_h(s * n_cells)} (once) | "
                         f"embedding size {db.num_features}")
            print("  " + lines[-1])

    out = root / "inventory_out" / "benchmark.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads\n" + "\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out}  (no patient data; send it to Claude)")


if __name__ == "__main__":
    main()
