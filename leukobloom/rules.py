"""Patient-level aggregation rules shared by Model A and Model B (pre-specified).

Ported from LeukoAI-dev (`leukoai/rules.py::published_rule`, `leukoai/experimental.py`), where the published
LeukoAI used three networks. Here both models give ONE 5-class probability vector per field
(ALL, AML, CLL, CML, NO_LEUKEMIA), mapped onto the three original decisions:
  network 1 (leukemia vs not):  leukemic field if p_leuk = 1 - p_NO > 0.5        (tie -> not leukemic)
  network 2 (acute vs chronic): chronic field if p_CLL + p_CML > p_ALL + p_AML    (tie -> acute)
  network 3 (5-class):          the field's top class                            (tie -> first class)
and then the original rule, unchanged, with the published threshold T = 3:
  published     leukemia if > T leukemic fields; chronic if >= 2 fields chronic (network 2 over ALL fields);
                subtype = most field votes within ALL/AML or CLL/CML (tie -> ALL / CLL)
  exp_majority  as published, but chronic if chronic fields > acute fields (tie -> acute)
  exp_meanprob  as published, but chronic if the mean chronic share > mean acute share over leukemic fields
  meanprob      not a rule from LeukoAI: mean of field probabilities, then top class (probabilistic summary)
Fields without a prediction (Model B fields with no usable cell) are left out; a patient with none -> NO_CELLS.
"""

from __future__ import annotations

import numpy as np

CLASSES = ["ALL", "AML", "CLL", "CML", "NO_LEUKEMIA"]
THREE = {"ALL": "ACUTE", "AML": "ACUTE", "CLL": "CHRONIC", "CML": "CHRONIC", "NO_LEUKEMIA": "NO_LEUKEMIA",
         "NO_CELLS": "NO_CELLS"}
T = 3
RULES = ["published", "exp_majority", "exp_meanprob", "meanprob"]


def _subtype(P: np.ndarray, chronic: bool) -> str:
    votes = np.bincount(P.argmax(1), minlength=5)
    sub = votes[2:4] if chronic else votes[:2]
    return CLASSES[int(np.argmax(sub)) + (2 if chronic else 0)]   # argmax: first on ties, as in toolv9


def apply_rule(P: np.ndarray, rule: str) -> str:
    """P: n_fields x 5 probabilities (rows with NaN are ignored). Returns a 5-class label or NO_CELLS."""
    P = P[~np.isnan(P).any(1)]
    if len(P) == 0:
        return "NO_CELLS"
    if rule == "meanprob":
        return CLASSES[int(P.mean(0).argmax())]
    leuk = (1 - P[:, 4]) > 0.5
    if leuk.sum() <= T:
        return "NO_LEUKEMIA"
    acute, chronic = P[:, 0] + P[:, 1], P[:, 2] + P[:, 3]
    if rule == "published":
        is_chronic = (chronic > acute).sum() > 1
    elif rule == "exp_majority":
        is_chronic = (chronic > acute).sum() > (chronic <= acute).sum()
    elif rule == "exp_meanprob":
        s = acute[leuk] + chronic[leuk]
        is_chronic = (chronic[leuk] / s).mean() > (acute[leuk] / s).mean()
    else:
        raise ValueError(rule)
    return _subtype(P, bool(is_chronic))
