"""Tests for the unsupervised detector — M2's unknown-unknowns claim.

The load-bearing test is `test_detector_flags_an_incident_type_the_classifier_never_saw`:
without it, the detector is redundant with the classifier and should be deleted rather than
shipped. The rest pin the honest limits, including the one place the story does NOT hold
(`test_detector_is_not_a_safety_net_under_classifier_misses`).
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.anomaly import ANOMALY_FEATURES, AnomalyDetector, detection_rate, fit_on_normal
from rdi.events import generate
from rdi.features import FEATURE_NAMES
from rdi.model import build_dataset, evaluate, temporal_split, train


@pytest.fixture(scope="module")
def split():
    X, y, ts = build_dataset(generate(n_ticks=6000, seed=7, incidents_per_service=10))
    return temporal_split(X, y, ts)


@pytest.fixture(scope="module")
def detector(split):
    Xtr, _, ytr, _ = split
    return fit_on_normal(Xtr, ytr)


# ---- construction ----

def test_anomaly_features_are_a_real_subset():
    """Excluding since_deploy_s and latency_ms is a decision, not an oversight."""
    assert set(ANOMALY_FEATURES) < set(FEATURE_NAMES)
    assert "since_deploy_s" not in ANOMALY_FEATURES  # a deploy is routine, not an anomaly
    assert "latency_ms" not in ANOMALY_FEATURES      # absolute; a slow-but-healthy service


def test_fit_refuses_a_training_set_with_no_normal_ticks():
    X = np.random.default_rng(0).random((20, len(FEATURE_NAMES)))
    with pytest.raises(ValueError, match="no normal ticks"):
        fit_on_normal(X, np.array(["memory_leak"] * 20))


def test_detector_never_trains_on_an_incident(split):
    """The whole claim rests on this: it has seen only healthy behaviour, so flagging an
    incident is a departure from normal rather than recognition of a learned class."""
    Xtr, _, ytr, _ = split
    normal = Xtr[ytr == "normal"]
    det = AnomalyDetector().fit(normal)
    assert len(normal) < len(Xtr)
    assert det.flag(normal).mean() < 0.10


# ---- it works ----

def test_false_positive_rate_on_normal_is_low(detector, split):
    _, Xte, _, yte = split
    assert detection_rate(detector, Xte, yte)["normal"]["flagged"] < 0.05


def test_every_incident_class_is_flagged_above_the_normal_rate(detector, split):
    _, Xte, _, yte = split
    rates = detection_rate(detector, Xte, yte)
    for cls, s in rates.items():
        if cls != "normal":
            assert s["flagged"] > rates["normal"]["flagged"] * 3, f"{cls} barely flagged"


def test_scores_are_higher_on_incidents_than_on_normal(detector, split):
    _, Xte, _, yte = split
    scores = detector.score(Xte)
    assert scores[yte != "normal"].mean() > scores[yte == "normal"].mean()


# ---- the reason it exists ----

def test_detector_flags_an_incident_type_the_classifier_never_saw(split):
    """Unknown-unknowns, demonstrated rather than asserted.

    Hide memory_leak from both models. The classifier cannot say "I don't know" — it has no
    such class — so it confidently spreads the leak across the labels it does have, calling
    ~39% of it `normal`. The detector, which was never told what a leak is either, still finds
    ~77% of it far from healthy behaviour.
    """
    Xtr, Xte, ytr, yte = split
    seen = ytr != "memory_leak"
    clf = train(Xtr[seen], ytr[seen])
    det = fit_on_normal(Xtr[seen], ytr[seen])

    held_out = yte == "memory_leak"
    assert held_out.sum() > 100, "not enough held-out examples to conclude anything"
    assert "memory_leak" not in set(clf.classes_)

    called_normal = float((clf.predict(Xte[held_out]) == "normal").mean())
    flagged = float(det.flag(Xte[held_out]).mean())
    assert called_normal > 0.20, "classifier did not silently absorb the unknown class"
    assert flagged > 0.60, f"detector only flagged {flagged:.0%} of an unseen incident"
    assert flagged > called_normal


def test_classifier_and_detector_are_complementary_across_classes(split):
    """Measured, not assumed: their per-class strengths are anti-correlated (~-0.66).

    The classifier's worst class (dependency_failure, recall ~0.50) is the detector's best
    (~99% flagged), and its best (bad_deploy, recall 1.00) is the detector's worst (~30%).
    Two models that failed on the same classes would not be worth the second one.
    """
    Xtr, Xte, ytr, yte = split
    per = evaluate(train(Xtr, ytr), Xte, yte)["per_class"]
    rates = detection_rate(fit_on_normal(Xtr, ytr), Xte, yte)
    incidents = [c for c in per if c != "normal"]
    corr = float(np.corrcoef([per[c]["recall"] for c in incidents],
                             [rates[c]["flagged"] for c in incidents])[0, 1])
    assert corr < 0.0, f"strengths correlate ({corr:+.2f}) — the detector adds little"


def test_detector_is_not_a_safety_net_under_classifier_misses(split):
    """The honest limit, pinned so nobody oversells the layer.

    Class-level complementarity does NOT imply event-level complementarity. The events the
    classifier misses (calls `normal`) are the mild ones — an early leak ramp, a pre-breach
    tick — and those look normal to the detector too. Measured: it catches under a quarter of
    them, and "classifier says normal but detector flags" carries no signal at all (~15% of
    such events are incidents against a ~20% base rate — worse than guessing).
    """
    Xtr, Xte, ytr, yte = split
    clf, det = train(Xtr, ytr), fit_on_normal(Xtr, ytr)
    pred, flag = clf.predict(Xte), det.flag(Xte)

    missed = (pred == "normal") & (yte != "normal")
    assert missed.sum() > 50
    assert float(flag[missed].mean()) < 0.50, \
        "detector now rescues most misses — re-check whether the failure modes really differ"

    disagree = (pred == "normal") & (flag == 1)
    precision = float((yte[disagree] != "normal").mean())
    base_rate = float((yte != "normal").mean())
    assert precision < base_rate * 1.5, \
        "disagreement now beats the base rate — it would be a usable escalation signal"
