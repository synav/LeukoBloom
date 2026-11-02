"""White-cell detection and cropping for May-Grunwald-Giemsa high-power fields.

No training needed: white cells (nucleus and most cytoplasm) are purple-blue, red cells pink,
background pale. Steps:
  1. Estimate the image scale from red-cell size (median diameter of isolated red cells),
     so every camera is cropped at the same PHYSICAL size.
  2. Mask purple pixels (blue minus red above a threshold), clean up, and split touching
     cells with a distance-transform watershed.
  3. Keep components of white-cell size; drop platelets, debris and cells cut by the border.
  4. Return square crops centred on each cell, CROP_RBC red-cell diameters wide
     (default 3.0 ~ 22 um, enough for a large blast or monocyte).

Use detect_cells(image) -> list of Cell; crop_cells(image, cells, out_px) -> list of PIL crops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature

DETECTOR_VERSION = "v2"   # v2: per-field white balance before colour thresholds
WORK_RBC_PX = 40.0        # images are rescaled so a red cell is ~40 px wide for detection
PURPLE_THR = 35.0         # blue - red (0-255 scale); pink RBCs are negative, background ~0-15
CROP_RBC = 3.0            # crop side = 3 red-cell diameters
MIN_AREA_RBC = 0.30       # white cell must cover >= 0.30 red-cell areas (platelets ~0.05)
MAX_AREA_RBC = 12.0       # larger blobs are clumps / smears, kept but flagged
MERGE_RBC = 1.2           # small pieces closer than this (red-cell diameters) are one cell (nuclear lobes)
LOBE_MAX_RBC = 1.2        # a piece larger than this (red-cell areas) is a whole cell, never merged
MERGED_MAX_RBC = 2.5      # merged lobes cannot exceed this area
DEFAULT_RBC_PX = 90.0     # fallback red-cell diameter at full resolution (1600x1200 camera)


@dataclass
class Cell:
    x: float                  # centre, full-resolution pixels
    y: float
    area_rbc: float           # purple area in red-cell areas
    touches_border: bool
    clump: bool
    score: float              # mean purpleness, for ranking / QC


def rgb_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def white_balance(a: np.ndarray, target: float = 240.0) -> np.ndarray:
    """Per-field white balance: scale each channel so the slide background (brightest ~15% of
    pixels) is neutral grey-white. Removes camera / lamp colour casts before colour thresholds."""
    s = a.sum(axis=2)
    bright = s >= np.percentile(s, 85)
    bg = np.median(a[bright], axis=0)
    return np.clip(a * (target / np.maximum(bg, 1.0)), 0, 255)


def estimate_rbc_diameter(img: Image.Image) -> tuple[float, int]:
    """Median diameter (full-res px) of isolated, round red cells. Returns (diameter, n_cells)."""
    w, h = img.size
    f = 4.0 if max(w, h) > 2500 else 2.0
    small = img.convert("RGB").resize((int(w / f), int(h / f)), Image.Resampling.BILINEAR)
    a = white_balance(rgb_float(small))
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    # red cells: clearly darker than background in green, and pink (red > blue)
    bg = np.percentile(G, 95)
    rbc = (G < bg - 25) & (R - B > -5)
    rbc = ndi.binary_fill_holes(ndi.binary_opening(rbc, morphology.disk(2)))
    lab = measure.label(rbc)
    d = []
    for r in measure.regionprops(lab):
        if r.area < 50 or r.solidity < 0.93 or r.eccentricity > 0.6:
            continue
        minr, minc, maxr, maxc = r.bbox
        if minr == 0 or minc == 0 or maxr == lab.shape[0] or maxc == lab.shape[1]:
            continue
        d.append(r.equivalent_diameter_area)
    if len(d) < 8:
        return DEFAULT_RBC_PX * (max(w, h) / 1600.0), len(d)
    d = np.array(d)
    # isolated single RBCs form the main mode; drop doublets that slipped through
    med = np.median(d)
    d = d[(d > 0.6 * med) & (d < 1.4 * med)]
    return float(np.median(d) * f), int(len(d))


def detect_cells(img: Image.Image, rbc_px: float | None = None) -> tuple[list[Cell], float]:
    """Find white cells. Returns (cells, red-cell diameter in full-res px)."""
    if rbc_px is None:
        rbc_px, _ = estimate_rbc_diameter(img)
    scale = WORK_RBC_PX / rbc_px
    w, h = img.size
    small = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                      Image.Resampling.BILINEAR)
    a = white_balance(rgb_float(small))
    purple = a[..., 2] - a[..., 0]
    purple = ndi.gaussian_filter(purple, 1.0)
    mask = purple > PURPLE_THR
    mask = ndi.binary_opening(mask, morphology.disk(2))
    mask = ndi.binary_closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)

    rbc_area = np.pi * (WORK_RBC_PX / 2) ** 2
    lab0, n0 = ndi.label(mask)
    sizes = ndi.sum(mask, lab0, range(1, n0 + 1))
    mask = np.isin(lab0, 1 + np.flatnonzero(sizes >= MIN_AREA_RBC * rbc_area * 0.5))

    # split touching cells: peaks of the distance transform at least ~0.7 RBC apart
    dist = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(dist, min_distance=int(0.7 * WORK_RBC_PX),
                                    threshold_abs=0.15 * WORK_RBC_PX, labels=measure.label(mask))
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    lab = segmentation.watershed(-dist, markers, mask=mask) if len(coords) else measure.label(mask)

    H, W = mask.shape
    margin = 0.15 * WORK_RBC_PX
    parts = []
    for r in measure.regionprops(lab, intensity_image=purple):
        minr, minc, maxr, maxc = r.bbox
        border = minr < margin or minc < margin or maxr > H - margin or maxc > W - margin
        cy, cx = r.centroid
        parts.append([cx, cy, r.area, border, r.intensity_mean * r.area])

    # merge pieces of one cell (separate nuclear lobes) by centre distance
    n = len(parts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    lim = MERGE_RBC * WORK_RBC_PX
    for i in range(n):
        for j in range(i + 1, n):
            close = (parts[i][0] - parts[j][0]) ** 2 + (parts[i][1] - parts[j][1]) ** 2 < lim ** 2
            small = max(parts[i][2], parts[j][2]) < LOBE_MAX_RBC * rbc_area
            if close and small and parts[i][2] + parts[j][2] < MERGED_MAX_RBC * rbc_area:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(parts[i])

    cells = []
    for g in groups.values():
        area = sum(p[2] for p in g)
        area_rbc = area / rbc_area
        if area_rbc < MIN_AREA_RBC:
            continue
        cx = sum(p[0] * p[2] for p in g) / area
        cy = sum(p[1] * p[2] for p in g) / area
        cells.append(Cell(x=cx / scale, y=cy / scale, area_rbc=float(area_rbc),
                          touches_border=any(p[3] for p in g), clump=area_rbc > MAX_AREA_RBC,
                          score=float(sum(p[4] for p in g) / area)))
    return cells, rbc_px


def crop_cells(img: Image.Image, cells: list[Cell], rbc_px: float, out_px: int = 224,
               crop_rbc: float = CROP_RBC) -> list[Image.Image]:
    """Square crops of crop_rbc red-cell diameters, centred on each cell, resized to out_px.
    Regions outside the image are filled with the field's background colour."""
    side = crop_rbc * rbc_px
    rgb = img.convert("RGB")
    a = np.asarray(rgb)
    bgcol = tuple(int(v) for v in np.percentile(a.reshape(-1, 3), 90, axis=0))
    out = []
    for c in cells:
        x0, y0 = round(c.x - side / 2), round(c.y - side / 2)
        s = round(side)
        canvas = Image.new("RGB", (s, s), bgcol)
        box = (max(0, x0), max(0, y0), min(rgb.width, x0 + s), min(rgb.height, y0 + s))
        canvas.paste(rgb.crop(box), (box[0] - x0, box[1] - y0))
        out.append(canvas.resize((out_px, out_px), Image.Resampling.LANCZOS))
    return out
