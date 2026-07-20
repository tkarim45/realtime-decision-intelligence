"""Tests for split-conformal sets — M2's calibrated-uncertainty claim.

Two things get pinned here, and the second one is a prediction that FAILED:

  1. The coverage guarantee holds on the stream (it does).
  2. Conformal was supposed to express the dependency_failure/bad_deploy ambiguity. It does
     not, because the model is confidently WRONG rather than uncertain. See
     `test_conformal_cannot_hedge_a_confidently_wrong_model`.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.conformal import coverage, naive_sets, prediction_sets, qhat, routing
from rdi.events import generate
from rdi.model import build_dataset, train


@pytest.fixture(scope="module")
def calibrated():
    """train -> calibrate -> test, in time order.

    Calibration is its own slice: calibrating on training points the model has already fit
    makes nonconformity optimistically small and the guarantee a fiction.
    """
    X, y, ts = build_dataset(generate(n_ticks=6000, seed=7, incidents_per_service=10))
    c1, c2 = np.quantile(ts, 0.55), np.quantile(ts, 0.75)
    tr, ca, te = ts < c1, (ts >= c1) & (ts < c2), ts >= c2
    model = train(X[tr], y[tr])
    classes = list(model.classes_)
    y_idx = np.array([classes.index(v) for v in y[ca]])
    return {
        "p_calib": model.predict_proba(X[ca]), "y_idx": y_idx,
        "p_test": model.predict_proba(X[te]), "y_test": y[te],
        "classes": classes, "model": model, "X_test": X[te],
    }


def _sets(c, alpha):
    return prediction_sets(c["p_test"], qhat(c["p_calib"], c["y_idx"], alpha), c["classes"])


# ---- the guarantee ----

@pytest.mark.parametrize("alpha", [0.10, 0.05])
def test_coverage_meets_its_target(calibrated, alpha):
    """P(true in set) >= 1 - alpha, distribution-free and finite-sample."""
    cov = coverage(_sets(calibrated, alpha), calibrated["y_test"])["coverage"]
    assert cov >= (1 - alpha) - 0.02, f"coverage {cov:.3f} short of {1 - alpha:.2f}"


def test_tighter_alpha_grows_the_sets(calibrated):
    """The only currency conformal has: more coverage is bought with bigger sets."""
    sizes = [coverage(_sets(calibrated, a), calibrated["y_test"])["avg_set_size"]
             for a in (0.10, 0.05, 0.01)]
    assert sizes[0] <= sizes[1] <= sizes[2]


def test_qhat_rises_as_alpha_falls(calibrated):
    q = [qhat(calibrated["p_calib"], calibrated["y_idx"], a) for a in (0.10, 0.05, 0.01)]
    assert q[0] <= q[1] <= q[2]


def test_naive_softmax_cannot_be_dialled(calibrated):
    """The counterexample. Thresholding the softmax barely moves with alpha, so its coverage
    plateaus near the model's accuracy — it cannot deliver a chosen guarantee."""
    covs = [coverage(naive_sets(calibrated["p_test"], a, calibrated["classes"]),
                     calibrated["y_test"])["coverage"] for a in (0.10, 0.05, 0.01)]
    assert max(covs) - min(covs) < 0.02, "naive coverage responds to alpha after all"
    assert covs[-1] < 0.99, "naive somehow hit a 99% target"


# ---- the prediction that failed ----

def test_conformal_cannot_hedge_a_confidently_wrong_model(calibrated):
    """The build plan predicted conformal would express the dependency_failure/bad_deploy
    ambiguity, so an ambiguous set could route the event to M3 instead of auto-remediating.
    It does not, and the reason matters.

    When the model misreads a dependency failure as a bad deploy it assigns the wrong class
    ~0.97 probability and the right one ~0.02. It is not uncertain, it is wrong. Conformal
    can only widen a set the model already hesitates on — it cannot manufacture doubt. At a
    95% target barely any dependency_failure set contains both classes.
    """
    c = calibrated
    dep = c["y_test"] == "dependency_failure"
    pred = c["model"].predict(c["X_test"])
    misread = dep & (pred == "bad_deploy")
    assert misread.sum() > 20, "the confusion this test is about did not occur"

    bad_i = c["classes"].index("bad_deploy")
    dep_i = c["classes"].index("dependency_failure")
    assert c["p_test"][misread, bad_i].mean() > 0.8, "not confidently wrong — story changed"
    assert c["p_test"][misread, dep_i].mean() < 0.2

    both = {"bad_deploy", "dependency_failure"}
    hedged = sum(1 for s, t in zip(_sets(c, 0.05), c["y_test"], strict=True)
                 if t == "dependency_failure" and both <= s)
    assert hedged / dep.sum() < 0.10, "conformal now hedges the confusion — re-verify"


def test_expressing_that_ambiguity_costs_the_whole_stream(calibrated):
    """Why you cannot simply demand a tighter alpha to recover the hedge.

    At 99% the sets do start containing both classes — and ~40% of ALL events become
    ambiguous, so 'escalate on an ambiguous set' would escalate a third of the stream to an
    LLM. The uncertainty layer cannot be the routing signal for this confusion at any usable
    operating point.
    """
    tight = coverage(_sets(calibrated, 0.01), calibrated["y_test"])
    loose = coverage(_sets(calibrated, 0.05), calibrated["y_test"])
    assert tight["ambiguous_rate"] > 0.25
    assert loose["ambiguous_rate"] < 0.10


# ---- operational read ----

def test_routing_splits_the_stream_into_act_and_escalate(calibrated):
    r = routing(_sets(calibrated, 0.05))
    assert abs(sum(r.values()) - 1.0) < 1e-9
    assert r["act_singleton"] > 0.8


def test_coverage_accounting_is_consistent(calibrated):
    cov = coverage(_sets(calibrated, 0.05), calibrated["y_test"])
    assert abs(cov["singleton_rate"] + cov["ambiguous_rate"] + cov["empty_rate"] - 1.0) < 1e-9
