"""LeukoBloom Model B, step 2: attention-MIL on DinoBloom cell features.

Schemes
  --scheme cv        (main)  5-fold patient-grouped cross-validation from folds.csv; every CV patient
                             is tested once by a model that never saw them.
  --scheme external          train on the TRAINING folders only (camera C1600 + training-only images),
                             test on the 87 VALIDATION cases (other cameras): the camera-shift analysis.

Pre-specified settings (fixed before any result is seen; do not tune on test folds):
  cells touching the image edge excluded; bags of up to 512 cells during training, with a random
  subset of each patient's fields; 40 epochs; AdamW lr 2e-4, weight decay 1e-4, cosine schedule;
  class-balanced loss; 3 seeds per fold, probabilities averaged.
Training-only images without a patient ID are grouped into pseudo-patients of 10 fields per class.

Outputs (Leukobloom\\results\\model_b_<features>_<scheme>\\, codes only, no images):
  patients.csv   one row per tested patient: true label, 5 class probabilities, prediction
  fields.csv     one row per tested field: same, from that field's cells alone (for the field-
                 level aggregation rules shared with Model A)
  attention\\     per-patient cell attention (which cells drove the result), for the cell gallery
  summary.txt    accuracy, balanced accuracy, macro-F1, confusion matrices (also copied to shareable)

Usage:  py -3.12 scripts\\train_mil.py                 (DinoBloom-B, 5-fold CV)
        py -3.12 scripts\\train_mil.py --scheme external
        py -3.12 scripts\\train_mil.py --features s
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.mil import CLASSES, GatedAttentionMIL  # noqa: E402

THREE = {"ALL": "ACUTE", "AML": "ACUTE", "CLL": "CHRONIC", "CML": "CHRONIC", "NO_LEUKEMIA": "NO_LEUKEMIA"}
C3 = ["ACUTE", "CHRONIC", "NO_LEUKEMIA"]
EPOCHS, LR, WD, MAX_CELLS, SEEDS, PSEUDO_FIELDS = 40, 2e-4, 1e-4, 512, (0, 1, 2), 10


# ------------------------------------------------------------------------------ data
def load(root: Path, feat_dir: Path, keep_border: bool):
    folds = pd.read_csv(root / "inventory_out" / "shareable" / "folds.csv")
    folds = folds[folds.include.astype(str).str.lower() == "true"].copy()
    folds["patient_code"] = folds.patient_code.fillna("")
    fields, missing = {}, 0
    for r in folds.itertuples():
        p = feat_dir / f"{r.image_id}.npz"
        if not p.exists():
            missing += 1
            continue
        z = np.load(p)
        f = z["feats"].astype(np.float32)
        if len(f) and not keep_border:
            f = f[~z["touches_border"].astype(bool)]
        fields[r.image_id] = f
    folds = folds[folds.image_id.isin(fields)]
    if missing:
        print(f"WARNING: {missing} included images have no feature file (not yet extracted?)")
    return folds, fields


def make_bags(folds: pd.DataFrame, fields: dict, seed: int = 2026):
    """Bag = list of field arrays. Returns DataFrame of bags with id, label, role, fold, set, image_ids."""
    bags = []
    with_id = folds[folds.patient_code != ""]
    for pc, g in with_id.groupby("patient_code"):
        bags.append(dict(bag=pc, label5=g.label5.iloc[0], role=g.role.iloc[0], fold=int(g.fold.iloc[0]),
                         set=g.set.iloc[0], image_ids=list(g.image_id)))
    rng = np.random.default_rng(seed)
    no_id = folds[folds.patient_code == ""]
    for (cls7, lab), g in no_id.groupby(["class7", "label5"]):
        ids = list(g.image_id)
        rng.shuffle(ids)
        for k in range(0, len(ids), PSEUDO_FIELDS):
            bags.append(dict(bag=f"PSEUDO_{cls7}_{k // PSEUDO_FIELDS:03d}", label5=lab, role="train_only",
                             fold=-1, set=g.set.iloc[0], image_ids=ids[k:k + PSEUDO_FIELDS]))
    b = pd.DataFrame(bags)
    b["n_cells"] = [sum(len(fields[i]) for i in ids) for ids in b.image_ids]
    return b[b.n_cells > 0].reset_index(drop=True)


def bag_tensor(image_ids, fields, rng=None):
    if rng is not None:  # augmentation: random subset of fields, then of cells
        frac = rng.uniform(0.5, 1.0)
        keep = [i for i in image_ids if rng.random() < frac] or [image_ids[rng.integers(len(image_ids))]]
        x = np.concatenate([fields[i] for i in keep])
        if len(x) == 0:
            x = np.concatenate([fields[i] for i in image_ids])
        if len(x) > MAX_CELLS:
            x = x[rng.choice(len(x), MAX_CELLS, replace=False)]
    else:
        x = np.concatenate([fields[i] for i in image_ids])
    return torch.from_numpy(x)


# ------------------------------------------------------------------------------ model
def train_one(train_bags, fields, dim, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedAttentionMIL(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * len(train_bags)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    y = np.array([CLASSES.index(l) for l in train_bags.label5])
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0),
                     dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for i in rng.permutation(len(train_bags)):
            logits, _ = model(bag_tensor(train_bags.image_ids.iloc[i], fields, rng))
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([y[i]]), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.inference_mode()
def predict(models, image_ids, fields):
    x = bag_tensor(image_ids, fields)
    probs, att = [], []
    for m in models:
        logits, a = m(x)
        probs.append(torch.softmax(logits, 0).numpy())
        att.append(a.numpy())
    return np.mean(probs, 0), np.mean(att, 0)


# ------------------------------------------------------------------------------ metrics
def metrics(y_true, y_pred, labels):
    y_true, y_pred = list(y_true), list(y_pred)
    cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels),
                     rownames=["true"], colnames=["pred"], dropna=False)
    tp = np.diag(cm.values).astype(float)
    sens = tp / np.maximum(cm.values.sum(1), 1)
    prec = tp / np.maximum(cm.values.sum(0), 1)
    f1 = np.where(sens + prec > 0, 2 * sens * prec / np.maximum(sens + prec, 1e-9), 0.0)
    present = cm.values.sum(1) > 0
    return dict(n=len(y_true), acc=tp.sum() / max(len(y_true), 1), bal_acc=sens[present].mean(),
                macro_f1=f1[present].mean(), per_class_f1=dict(zip(labels, f1.round(3))), cm=cm)


def summarise(pat, title):
    L = [title, ""]
    for name, t, p, labels in (("5-class", pat.label5, pat.pred5, CLASSES), ("3-class", pat.label3, pat.pred3, C3)):
        m = metrics(t, p, labels)
        L.append(f"{name}: n={m['n']}  accuracy={m['acc']:.3f}  balanced accuracy={m['bal_acc']:.3f}  "
                 f"macro-F1={m['macro_f1']:.3f}")
        L.append("  F1 per class: " + ", ".join(f"{k}={v}" for k, v in m["per_class_f1"].items()))
        L.append(m["cm"].to_string())
        L.append("")
    if "fold" in pat and pat.fold.nunique() > 1:
        L.append("Per fold (5-class accuracy / macro-F1):")
        for f, g in pat.groupby("fold"):
            m = metrics(g.label5, g.pred5, CLASSES)
            L.append(f"  fold {f}: n={m['n']}  acc={m['acc']:.3f}  macro-F1={m['macro_f1']:.3f}")
    return "\n".join(L)


# ------------------------------------------------------------------------------ main
def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=home / "Downloads" / "Leukobloom")
    ap.add_argument("--features", choices=["b", "s"], default="b")
    ap.add_argument("--scheme", choices=["cv", "external"], default="cv")
    ap.add_argument("--keep-border", action="store_true", help="also use cells touching the image edge")
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    feat_dir = a.root / "features" / f"dinobloom_{a.features}"
    folds, fields = load(a.root, feat_dir, a.keep_border)
    dim = next(f.shape[1] for f in fields.values() if f.ndim == 2 and f.shape[1] > 0)
    bags = make_bags(folds, fields)
    tag = f"model_b_{a.features}_{a.scheme}" + ("_border" if a.keep_border else "")
    out = a.root / "results" / tag
    (out / "attention").mkdir(parents=True, exist_ok=True)
    print(f"{len(folds)} fields, {sum(len(f) for f in fields.values())} cells, {len(bags)} bags "
          f"({(bags.role == 'cv').sum()} CV patients); feature size {dim}")

    if a.scheme == "cv":
        splits = [(k, bags[(bags.role != "cv") | (bags.fold != k)], bags[(bags.role == "cv") & (bags.fold == k)])
                  for k in sorted(bags[bags.role == "cv"].fold.unique())]
    else:
        splits = [(-1, bags[bags.set == "training"], bags[(bags.set == "validation") & (bags.role == "cv")])]

    pat_rows, field_rows, t0 = [], [], time.time()
    for k, tr, te in splits:
        models = [train_one(tr.reset_index(drop=True), fields, dim, s) for s in SEEDS]
        for b in te.itertuples():
            p, att = predict(models, b.image_ids, fields)
            pat_rows.append(dict(patient_code=b.bag, fold=k, set=b.set, label5=b.label5, n_fields=len(b.image_ids),
                                 n_cells=sum(len(fields[i]) for i in b.image_ids),
                                 **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
            cell_img = np.concatenate([[i] * len(fields[i]) for i in b.image_ids])
            cell_idx = np.concatenate([np.arange(len(fields[i])) for i in b.image_ids])
            np.savez_compressed(out / "attention" / f"{b.bag}.npz", image_id=cell_img, cell=cell_idx, attention=att)
            for i in b.image_ids:
                fp = predict(models, [i], fields)[0] if len(fields[i]) else np.full(len(CLASSES), np.nan)
                field_rows.append(dict(image_id=i, patient_code=b.bag, fold=k, label5=b.label5, n_cells=len(fields[i]),
                                       **{f"p_{c}": v for c, v in zip(CLASSES, fp)}))
        print(f"  {'fold ' + str(k) if k >= 0 else 'external'}: trained on {len(tr)} bags, tested {len(te)} patients "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    pat = pd.DataFrame(pat_rows)
    P = pat[[f"p_{c}" for c in CLASSES]].values
    pat["pred5"] = [CLASSES[i] for i in P.argmax(1)]
    p3 = np.stack([P[:, 0] + P[:, 1], P[:, 2] + P[:, 3], P[:, 4]], 1)
    pat["pred3"] = [C3[i] for i in p3.argmax(1)]
    pat["label3"] = pat.label5.map(THREE)
    pat.to_csv(out / "patients.csv", index=False)
    pd.DataFrame(field_rows).to_csv(out / "fields.csv", index=False)

    title = (f"Model B (DinoBloom-{a.features.upper()} + attention-MIL), scheme={a.scheme}, "
             f"{'all cells' if a.keep_border else 'edge cells excluded'}, seeds={len(SEEDS)}, epochs={EPOCHS}")
    txt = summarise(pat, title)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    (a.root / "inventory_out" / "shareable" / f"{tag}_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\nResults: {out}\nSend to Claude: inventory_out\\shareable\\{tag}_summary.txt")


if __name__ == "__main__":
    main()
