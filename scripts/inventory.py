"""LeukoBloom data inventory.

Runs on the study laptop only. Images are read locally and never copied anywhere.

What it does
------------
For every image under  <data>/training/<CLASS>/  and  <data>/validation/<CASE>/ :
  * reads the label from the folder name (7 original classes and the 5 study classes)
  * works out the patient: hospital ID in the file name, teaching-slide number,
    or the case folder for validation cases
  * computes an exact fingerprint (SHA-256) and a visual fingerprint (256-bit
    difference hash, which also matches the same field saved at another size or format)
  * records image size, format and warning flags
Then it checks for exact and near-duplicate images, the same patient appearing in both
training and validation, patients filed under two diagnoses, and other problems.

Outputs (in --out, default: Leukobloom\\inventory_out, outside the code folder)
  shareable\\inventory.csv   one row per image, NO file names or hospital IDs
  shareable\\summary.txt     counts and all checks, safe to send to Claude
  shareable\\near_duplicates.csv  candidate near-duplicate pairs (coded IDs only)
  PRIVATE\\image_key.csv     image code -> real file path       (NEVER share or upload)
  PRIVATE\\patient_key.csv   patient code -> hospital ID         (NEVER share or upload)

Usage (Command Prompt):
  py -3.12 inventory.py
  py -3.12 inventory.py --data "C:\\path\\to\\data" --out "C:\\path\\to\\out"

Needs: Pillow, numpy   (pip install pillow numpy)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # large microscope fields are fine

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HASH_SIDE = 16                  # 16x16 = 256-bit difference hash
NEAR_DUP_MAX_BITS = 12          # <=12 of 256 bits different -> candidate near-duplicate

# Folder label -> (7-class label, 5-class study label)
LABELS = {
    "ALL": ("ALL", "ALL"),
    "AML": ("AML", "AML"),
    "APML": ("APML", "AML"),
    "CLL": ("CLL", "CLL"),
    "CML": ("CML", "CML"),
    "NORMAL": ("NORMAL", "NO_LEUKEMIA"),
    "REACTIVE": ("REACTIVE", "NO_LEUKEMIA"),
    "REACTIVE CELLS": ("REACTIVE", "NO_LEUKEMIA"),
}

RE_HOSPITAL_ID = re.compile(r"(?<![0-9])([0-9]{8,14}[A-Za-z]?)(?=[_\-\s.(]|$)")
RE_TEACHING = re.compile(r"TEACHING\s*SLIDE[_\s-]*([0-9]+)", re.I)
RE_COPY = re.compile(r"(\s-\s?copy|\scopy\b|\(\d+\)\s*$)", re.I)
RE_UNCERTAIN = re.compile(r"\b(possible|probable|suspected|query|\?|r/o)\b", re.I)


# ----------------------------------------------------------------------------- helpers
def label_from_folder(name: str):
    """Training: 'AML', 'REACTIVE CELLS'. Validation: 'AML_V001', 'Normal_N003'."""
    up = name.strip().upper()
    if up in LABELS:
        return LABELS[up]
    head = up.split("_")[0]
    return LABELS.get(head, ("UNKNOWN", "UNKNOWN"))


def patient_key_from_filename(stem: str):
    """Return (raw_key, kind). kind: hospital_id | teaching | none."""
    m = RE_HOSPITAL_ID.search(stem)
    if m:
        return m.group(1).upper(), "hospital_id"
    m = RE_TEACHING.search(stem)
    if m:
        return f"TEACHINGSLIDE_{m.group(1)}", "teaching"
    return None, "none"


def dhash_bits(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def process_one(path_str: str):
    """Runs in a worker process: fingerprints and basic properties of one image."""
    p = Path(path_str)
    out = {"path": path_str, "sha256": "", "dhash": "", "width": 0, "height": 0,
           "mode": "", "size_kb": 0, "error": ""}
    try:
        data = p.read_bytes()
        out["size_kb"] = round(len(data) / 1024)
        out["sha256"] = hashlib.sha256(data).hexdigest()
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            out["width"], out["height"] = im.size
            out["mode"] = im.mode
            if im.format == "JPEG":
                im.draft("L", (im.size[0] // 8, im.size[1] // 8))  # fast decode
            bits = dhash_bits(im)
        out["dhash"] = np.packbits(bits).tobytes().hex()
    except Exception as e:  # unreadable / corrupt
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ----------------------------------------------------------------------------- scan
def scan(data_root: Path):
    rows, other_files, empty_folders = [], [], []
    for set_name in ("training", "validation"):
        base = data_root / set_name
        if not base.is_dir():
            sys.exit(f"Folder not found: {base}")
        for folder in sorted(d for d in base.iterdir() if d.is_dir()):
            lab7, lab5 = label_from_folder(folder.name)
            n_img = 0
            for f in sorted(folder.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in IMAGE_EXT:
                    if f.name.lower() not in ("desktop.ini", "thumbs.db", ".ds_store"):
                        other_files.append(str(f.relative_to(data_root)))
                    continue
                n_img += 1
                stem = f.stem
                raw_key, kind = patient_key_from_filename(stem)
                flags = []
                if RE_COPY.search(stem):
                    flags.append("copy_in_name")
                if RE_UNCERTAIN.search(stem):
                    flags.append("uncertain_label_in_name")
                if f.parent != folder:
                    flags.append("in_subfolder")
                if kind == "teaching":
                    flags.append("teaching_slide")
                rows.append({
                    "set": set_name, "folder": folder.name, "class7": lab7, "class5": lab5,
                    "path": str(f), "rel": str(f.relative_to(data_root)), "ext": f.suffix.lower(),
                    "raw_key": raw_key, "key_kind": kind, "flags": flags,
                })
            if n_img == 0:
                empty_folders.append(f"{set_name}/{folder.name}")
    return rows, other_files, empty_folders


# ----------------------------------------------------------------------------- main
def main():
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    default_root = home / "Downloads" / "Leukobloom"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=default_root / "data")
    ap.add_argument("--out", type=Path, default=default_root / "inventory_out")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    t0 = time.time()
    print(f"Scanning {args.data} ...")
    rows, other_files, empty_folders = scan(args.data)
    print(f"Found {len(rows)} images. Fingerprinting with {args.workers} workers ...")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_one, [r["path"] for r in rows], chunksize=16), 1):
            results[r["path"]] = r
            if i % 250 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"  {i}/{len(rows)} images  ({el/60:.1f} min)", flush=True)
    for r in rows:
        r.update({k: v for k, v in results[r["path"]].items() if k != "path"})
        if r["error"]:
            r["flags"].append("unreadable")

    # --- patient grouping -------------------------------------------------------------
    # Unit for cross-validation = patient group. Hospital IDs link training and validation;
    # a validation case folder is one patient, linked to any hospital ID found inside it.
    uf = UnionFind()
    for r in rows:
        nodes = []
        if r["raw_key"]:
            nodes.append(("ID", r["raw_key"]))
        if r["set"] == "validation":
            nodes.append(("CASE", r["folder"]))
        for n in nodes:
            uf.find(n)
        for a, b in zip(nodes, nodes[1:]):
            uf.union(a, b)
        r["_node"] = nodes[0] if nodes else None

    groups = sorted({uf.find(r["_node"]) for r in rows if r["_node"]})
    gcode = {g: f"P{i:04d}" for i, g in enumerate(groups, 1)}
    for r in rows:
        r["patient_code"] = gcode[uf.find(r["_node"])] if r["_node"] else ""

    # image codes, stable order
    rows.sort(key=lambda r: (r["set"], r["folder"], r["patient_code"], r["rel"]))
    for i, r in enumerate(rows, 1):
        r["image_id"] = f"IMG{i:05d}"

    # --- exact duplicates -------------------------------------------------------------
    by_sha = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_sha[r["sha256"]].append(r)
    exact_groups = [g for g in by_sha.values() if len(g) > 1]
    for k, g in enumerate(exact_groups, 1):
        for r in g:
            r["exact_dup_group"] = f"E{k:04d}"

    # --- near duplicates (visual hash, catches resized / re-saved copies) --------------
    ok = [r for r in rows if r["dhash"]]
    near_pairs = []
    if ok:
        H = np.stack([np.frombuffer(bytes.fromhex(r["dhash"]), np.uint8) for r in ok])  # n x 32 bytes
        popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        n = len(ok)
        print("Comparing visual fingerprints ...")
        bi, bj = 256, 4096
        for s in range(0, n, bi):
            block = H[s:s + bi]
            for j0 in range(s, n, bj):
                other = H[j0:j0 + bj]
                dist = popcount[block[:, None, :] ^ other[None, :, :]].sum(axis=2, dtype=np.int16)
                ii, jj = np.nonzero(dist <= NEAR_DUP_MAX_BITS)
                for a, b in zip(ii, jj):
                    i, j = s + a, j0 + b
                    if j <= i:
                        continue
                    ri, rj = ok[i], ok[j]
                    if ri["sha256"] == rj["sha256"]:
                        continue  # already an exact duplicate
                    near_pairs.append((ri, rj, int(dist[a, b])))
    nuf = UnionFind()
    for ri, rj, _ in near_pairs:
        nuf.union(ri["image_id"], rj["image_id"])
    near_roots = sorted({nuf.find(r["image_id"]) for pair in near_pairs for r in pair[:2]})
    ncode = {g: f"N{k:04d}" for k, g in enumerate(near_roots, 1)}
    for r in rows:
        if r["image_id"] in nuf.parent:
            r["near_dup_group"] = ncode[nuf.find(r["image_id"])]

    # --- per-patient checks ------------------------------------------------------------
    pat_sets = defaultdict(set)
    pat_class5 = defaultdict(set)
    pat_class7 = defaultdict(set)
    case_ids = defaultdict(set)      # validation case folder -> hospital IDs inside
    pat_kind = {}
    for r in rows:
        pc = r["patient_code"]
        if pc:
            pat_sets[pc].add(r["set"])
            pat_class5[pc].add(r["class5"])
            pat_class7[pc].add(r["class7"])
            pat_kind.setdefault(pc, r["key_kind"] if r["set"] == "training" else "case_folder")
        if r["set"] == "validation" and r["raw_key"]:
            case_ids[r["folder"]].add(r["raw_key"])
    cross_set = sorted(p for p, s in pat_sets.items() if len(s) > 1)
    conflict5 = sorted(p for p, c in pat_class5.items() if len(c) > 1)
    conflict7 = sorted(p for p, c in pat_class7.items() if len(c) > 1 and p not in conflict5)
    multi_id_cases = sorted(c for c, ids in case_ids.items() if len(ids) > 1)
    case_folders_in_group = defaultdict(set)
    for r in rows:
        if r["set"] == "validation" and r["patient_code"]:
            case_folders_in_group[r["patient_code"]].add(r["folder"])
    merged_cases = {p: sorted(f) for p, f in case_folders_in_group.items() if len(f) > 1}

    for r in rows:
        if r["patient_code"] in cross_set:
            r["flags"].append("patient_in_both_sets")
        if r["patient_code"] in conflict5:
            r["flags"].append("patient_label_conflict")
        if not r["patient_code"]:
            r["flags"].append("no_patient_id")

    # --- write outputs -------------------------------------------------------------
    shr = args.out / "shareable"
    prv = args.out / "PRIVATE"
    shr.mkdir(parents=True, exist_ok=True)
    prv.mkdir(parents=True, exist_ok=True)

    cols = ["image_id", "set", "case_or_class_folder", "class7", "class5", "patient_code",
            "patient_source", "ext", "width", "height", "mode", "size_kb",
            "exact_dup_group", "near_dup_group", "flags", "sha256"]
    with open(shr / "inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            folder = r["folder"] if r["set"] == "validation" else r["class7"]
            w.writerow([r["image_id"], r["set"], folder, r["class7"], r["class5"], r["patient_code"],
                        r["key_kind"] if r["set"] == "training" else "case_folder",
                        r["ext"], r["width"], r["height"], r["mode"], r["size_kb"],
                        r.get("exact_dup_group", ""), r.get("near_dup_group", ""),
                        ";".join(sorted(set(r["flags"]))), r["sha256"]])

    with open(shr / "near_duplicates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_a", "image_b", "bits_different_of_256", "set_a", "set_b",
                    "patient_a", "patient_b", "class5_a", "class5_b", "size_a", "size_b"])
        for ri, rj, d in sorted(near_pairs, key=lambda x: x[2]):
            w.writerow([ri["image_id"], rj["image_id"], d, ri["set"], rj["set"],
                        ri["patient_code"] or "-", rj["patient_code"] or "-", ri["class5"], rj["class5"],
                        f'{ri["width"]}x{ri["height"]}', f'{rj["width"]}x{rj["height"]}'])

    with open(prv / "image_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "relative_path"])
        for r in rows:
            w.writerow([r["image_id"], r["rel"]])

    with open(prv / "patient_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_code", "hospital_ids_or_keys", "validation_case_folders"])
        members = defaultdict(set)
        for r in rows:
            if r["patient_code"] and r["raw_key"]:
                members[r["patient_code"]].add(r["raw_key"])
        for g in groups:
            pc = gcode[g]
            w.writerow([pc, ";".join(sorted(members.get(pc, []))),
                        ";".join(sorted(case_folders_in_group.get(pc, [])))])

    write_summary(shr / "summary.txt", args, rows, exact_groups, near_pairs, cross_set, conflict5,
                  conflict7, multi_id_cases, merged_cases, other_files, empty_folders, pat_class5,
                  pat_sets, pat_kind, time.time() - t0)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"  Send to Claude:  {shr}\\summary.txt  (and inventory.csv if asked)")
    print(f"  Keep private:    {prv}")


# ----------------------------------------------------------------------------- summary
def table(lines, header, data_rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *data_rows)] if data_rows else [len(h) for h in header]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines.append(fmt.format(*header))
    lines.append(fmt.format(*["-" * w for w in widths]))
    for d in data_rows:
        lines.append(fmt.format(*d))
    lines.append("")


def write_summary(path, args, rows, exact_groups, near_pairs, cross_set, conflict5, conflict7,
                  multi_id_cases, merged_cases, other_files, empty_folders, pat_class5, pat_sets,
                  pat_kind, elapsed):
    L = []
    L.append("LeukoBloom data inventory — summary (no file names or hospital IDs)")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')}; run time {elapsed/60:.1f} min")
    L.append(f"Near-duplicate threshold: <= {NEAR_DUP_MAX_BITS} of 256 visual-hash bits")
    L.append("")

    classes7 = ["ALL", "AML", "APML", "CLL", "CML", "NORMAL", "REACTIVE", "UNKNOWN"]
    classes5 = ["ALL", "AML", "CLL", "CML", "NO_LEUKEMIA", "UNKNOWN"]

    L.append("1. IMAGES BY SET AND CLASS (7 original classes)")
    data = []
    for s in ("training", "validation"):
        c = Counter(r["class7"] for r in rows if r["set"] == s)
        data.append([s] + [c.get(k, 0) for k in classes7] + [sum(c.values())])
    table(L, ["set"] + classes7 + ["total"], data)

    L.append("2. PATIENTS BY SET AND STUDY CLASS (5 classes; APML->AML, NORMAL+REACTIVE->NO_LEUKEMIA)")
    L.append("   'patients' = distinct patient codes; training images without an ID are counted separately")
    data = []
    for s in ("training", "validation"):
        pats = defaultdict(set)
        noid = Counter()
        for r in rows:
            if r["set"] != s:
                continue
            if r["patient_code"]:
                pats[r["class5"]].add(r["patient_code"])
            else:
                noid[r["class5"]] += 1
        data.append([f"{s}: patients"] + [len(pats.get(k, ())) for k in classes5]
                    + [len(set().union(*pats.values())) if pats else 0])
        data.append([f"{s}: images w/o patient ID"] + [noid.get(k, 0) for k in classes5] + [sum(noid.values())])
    allp = defaultdict(set)
    for r in rows:
        if r["patient_code"]:
            allp[r["class5"]].add(r["patient_code"])
    data.append(["ALL DATA: distinct patients"] + [len(allp.get(k, ())) for k in classes5]
                + [len({r['patient_code'] for r in rows if r['patient_code']})])
    table(L, ["", *classes5, "total"], data)

    L.append("   Training patient sources: " + ", ".join(
        f"{k}={v}" for k, v in Counter(pat_kind[p] for p in pat_kind if "training" in pat_sets[p]).items()))
    L.append("")

    L.append("3. IMAGES PER VALIDATION CASE")
    vc = Counter(r["folder"] for r in rows if r["set"] == "validation")
    if vc:
        vals = sorted(vc.values())
        L.append(f"   cases={len(vc)}  images={sum(vals)}  min={vals[0]}  median={vals[len(vals)//2]}  max={vals[-1]}")
        coh = Counter()
        for f in vc:
            m = re.match(r"^[A-Za-z]+_([A-Za-z]+)\d+", f)
            coh[(label_from_folder(f)[1], m.group(1).upper() if m else "?")] += 1
        table(L, ["class5", "cohort", "cases"], [[a, b, n] for (a, b), n in sorted(coh.items())])
        few = sorted((n, f) for f, n in vc.items() if n < 5)
        if few:
            L.append("   Cases with fewer than 5 images: " + ", ".join(f"{f} ({n})" for n, f in few))
            L.append("")

    L.append("4. IMAGE FORMAT AND SIZE BY CLASS (capture differences can act as a shortcut for the model)")
    res = Counter((r["class5"], f'{r["width"]}x{r["height"]}', r["ext"]) for r in rows if r["width"])
    top = Counter()
    for (c, wh, e), n in res.items():
        top[(wh, e)] += n
    common = [k for k, _ in top.most_common(8)]
    data = []
    for wh, e in common:
        data.append([f"{wh} {e}"] + [res.get((c, wh, e), 0) for c in classes5[:-1]])
    other = [sum(n for (c2, wh, e), n in res.items() if c2 == c and (wh, e) not in common) for c in classes5[:-1]]
    data.append(["other"] + other)
    table(L, ["size / format", *classes5[:-1]], data)

    L.append("5. EXACT DUPLICATES (identical files)")
    within_pat = cross_pat = cross_set_n = cross_cls = 0
    for g in exact_groups:
        sets = {r["set"] for r in g}
        pats = {r["patient_code"] or f"noid:{r['image_id']}" for r in g}
        cls = {r["class5"] for r in g}
        if len(sets) > 1:
            cross_set_n += 1
        if len(cls) > 1:
            cross_cls += 1
        if len(pats) > 1:
            cross_pat += 1
        else:
            within_pat += 1
    L.append(f"   groups={len(exact_groups)}  extra copies={sum(len(g)-1 for g in exact_groups)}")
    L.append(f"   same patient={within_pat}  different/unknown patients={cross_pat}  "
             f"across training/validation={cross_set_n}  across classes={cross_cls}")
    L.append("")

    L.append("6. NEAR DUPLICATES (same field re-saved, resized or slightly shifted) — candidates, check visually")
    nc = Counter()
    for ri, rj, d in near_pairs:
        kind = ("across sets" if ri["set"] != rj["set"] else
                "different class" if ri["class5"] != rj["class5"] else
                "same patient" if ri["patient_code"] and ri["patient_code"] == rj["patient_code"] else
                "different/unknown patient")
        nc[kind] += 1
    L.append(f"   pairs={len(near_pairs)}  " + "  ".join(f"{k}={v}" for k, v in nc.most_common()))
    L.append("   Details: near_duplicates.csv")
    L.append("")

    L.append("7. PATIENT-LEVEL CHECKS")
    L.append(f"   Patients in BOTH training and validation: {len(cross_set)}  {', '.join(cross_set)}")
    L.append(f"   Patients filed under two study classes:   {len(conflict5)}  " +
             "; ".join(f"{p} ({'/'.join(sorted(pat_class5[p]))})" for p in conflict5))
    L.append(f"   Patients in two original classes only (e.g. AML+APML): {len(conflict7)}  {', '.join(conflict7)}")
    L.append(f"   Validation folders containing >1 hospital ID: {len(multi_id_cases)}  {', '.join(multi_id_cases)}")
    L.append(f"   Different validation folders that are the same patient: {len(merged_cases)}  " +
             "; ".join(f"{p}: {'+'.join(f)}" for p, f in merged_cases.items()))
    L.append("")

    L.append("8. FLAGS")
    fc = Counter(f for r in rows for f in set(r["flags"]))
    data = []
    for f, n in fc.most_common():
        by = Counter(r["set"] + "/" + r["class7"] for r in rows if f in r["flags"])
        data.append([f, n, ", ".join(f"{k}:{v}" for k, v in by.most_common(6))])
    table(L, ["flag", "images", "where"], data)
    L.append(f"   Non-image files: {len(other_files)}   Empty folders: {', '.join(empty_folders) or 'none'}")
    errs = [r for r in rows if r["error"]]
    if errs:
        L.append(f"   Unreadable images: {len(errs)} (see 'unreadable' flag in inventory.csv)")
    L.append("")
    L.append("Keys linking codes to file names and hospital IDs are in the PRIVATE folder. Do not share them.")

    Path(path).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
