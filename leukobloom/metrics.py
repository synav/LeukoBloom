"""Patient-level metrics with bootstrap confidence intervals (pre-specified in docs/STUDY_PLAN.md)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .rules import CLASSES, THREE

C3 = ["ACUTE", "CHRONIC", "NO_LEUKEMIA"]


def confusion(y, p, labels):
    cols = labels + sorted(set(p) - set(labels))
    m = np.zeros((len(labels), len(cols)), int)
    for a, b in zip(y, p):
        m[labels.index(a), cols.index(b)] += 1
    return m, cols


def label_metrics(y, p, labels):
    m, _ = confusion(y, p, labels)
    k = len(labels)
    tp = np.diag(m[:, :k]).astype(float)
    n_true = m.sum(1)
    sens = np.divide(tp, n_true, out=np.zeros(k), where=n_true > 0)
    n_pred = m[:, :k].sum(0)
    prec = np.divide(tp, n_pred, out=np.zeros(k), where=n_pred > 0)
    f1 = np.divide(2 * sens * prec, sens + prec, out=np.zeros(k), where=(sens + prec) > 0)
    tn = len(y) - n_true - n_pred + tp
    spec = np.divide(tn, len(y) - n_true, out=np.zeros(k), where=(len(y) - n_true) > 0)
    present = n_true > 0
    return dict(acc=tp.sum() / len(y), bal_acc=sens[present].mean(), macro_f1=f1[present].mean(),
                sens=sens, spec=spec, f1=f1, n_true=n_true)


def headline(y5, p5, leuk_pred):
    """The numbers compared between methods."""
    y5, p5 = list(y5), list(p5)
    y3, p3 = [THREE[v] for v in y5], [THREE[v] for v in p5]
    m5, m3 = label_metrics(y5, p5, CLASSES), label_metrics(y3, p3, C3)
    yl = np.array([v != "NO_LEUKEMIA" for v in y5])
    lp = np.asarray(leuk_pred, bool)
    sens = (lp & yl).sum() / max(yl.sum(), 1)
    spec = (~lp & ~yl).sum() / max((~yl).sum(), 1)
    return {"5c_macro_f1": m5["macro_f1"], "5c_bal_acc": m5["bal_acc"], "5c_acc": m5["acc"],
            "3c_bal_acc": m3["bal_acc"], "3c_macro_f1": m3["macro_f1"], "3c_acc": m3["acc"],
            "leuk_sens": sens, "leuk_spec": spec}


def bootstrap(fn, n, B=2000, seed=2026):
    """Percentile CI of fn(idx) over patient resamples; fn returns a dict of floats."""
    rng = np.random.default_rng(seed)
    point = fn(np.arange(n))
    draws = {k: [] for k in point}
    for _ in range(B):
        d = fn(rng.integers(0, n, n))
        for k, v in d.items():
            draws[k].append(v)
    return {k: (point[k], np.nanpercentile(draws[k], 2.5), np.nanpercentile(draws[k], 97.5)) for k in point}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def auroc(score, positive):
    """One-vs-rest AUROC (Mann-Whitney); NaN if one group is empty."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def brier(P, y5):
    Y = np.zeros_like(P)
    Y[np.arange(len(y5)), [CLASSES.index(v) for v in y5]] = 1
    return float(((P - Y) ** 2).sum(1).mean())


def calibration_bins(P, y5, bins=10):
    """Top-label reliability: per bin, mean confidence, accuracy, count; plus ECE."""
    conf = P.max(1)
    correct = P.argmax(1) == np.array([CLASSES.index(v) for v in y5])
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            rows.append((lo, hi, conf[m].mean(), correct[m].mean(), int(m.sum())))
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return rows, ece


def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def net_benefit(p_leuk, y_leuk, thresholds):
    y = np.asarray(y_leuk, bool)
    n = len(y)
    out = []
    for t in thresholds:
        pos = np.asarray(p_leuk) >= t
        tp, fp = (pos & y).sum(), (pos & ~y).sum()
        out.append((t, tp / n - fp / n * t / (1 - t), y.mean() - (1 - y.mean()) * t / (1 - t)))
    return out
