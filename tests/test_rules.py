"""Known-answer tests: the shared rules reproduce the published LeukoAI (toolv9) decisions.
Run:  py -3.12 -m pytest tests   (or: py -3.12 tests\\test_rules.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leukobloom.rules import apply_rule  # noqa: E402

AML = [.05, .85, .03, .02, .05]
CLL = [.02, .03, .85, .05, .05]
ALL = [.80, .10, .03, .02, .05]
NO = [.02, .02, .02, .02, .92]
NA = [np.nan] * 5


def f(*rows):
    return np.array(rows, float)


def test_threshold_needs_more_than_three_leukemic_fields():
    assert apply_rule(f(AML, AML, AML, NO, NO), "published") == "NO_LEUKEMIA"
    assert apply_rule(f(AML, AML, AML, AML, NO), "published") == "AML"


def test_published_two_chronic_fields_turn_case_chronic():
    assert apply_rule(f(AML, AML, AML, AML, CLL, CLL), "published") == "CLL"
    assert apply_rule(f(AML, AML, AML, AML, CLL, CLL), "exp_majority") == "AML"


def test_subtype_tie_goes_to_all():
    assert apply_rule(f(AML, AML, ALL, ALL, NO), "published") == "ALL"


def test_missing_fields():
    assert apply_rule(f(NA, NA), "published") == "NO_CELLS"
    assert apply_rule(f(AML, AML, AML, AML, NA), "meanprob") == "AML"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all rule tests pass")
