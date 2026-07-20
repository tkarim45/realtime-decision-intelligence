"""Tests for the hot-path classifier.

The most valuable tests here are the ones guarding against fake tasks:
`test_label_is_not_a_learnable_target` (predicting the SLO breach is an if-statement) and
`test_random_split_inflates_the_score` (an episode's own neighbours leaking into test).
Both are traps this code actually fell into before they were pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.events import SLO_ERROR_RATE, SLO_LATENCY_MS, generate
from rdi.features import FEATURE_NAMES
from rdi.model import (
    INCIDENT_CLASSES,
    WARMUP_TICKS,
    build_dataset,
    evaluate,
    importances,
    random_split,
    temporal_split,
    train,
)


@pytest.fixture(scope="module")
def data():
    return build_dataset(generate(n_ticks=6000, seed=7, incidents_per_service=10))


@pytest.fixture(scope="module")
def fitted(data):
    X, y, ts = data
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    return train(Xtr, ytr), Xte, yte


# ---- the task is real ----

def test_label_is_not_a_learnable_target():
    """`label` is a deterministic function of two features, so a model on it is an
    if-statement in disguise. This is why the target is incident_type."""
    events = generate(n_ticks=400, seed=7)
    for e in events:
        assert e["label"] == int(e["latency_ms"] > SLO_LATENCY_MS
                                 or e["error_rate"] > SLO_ERROR_RATE)


def test_dataset_covers_every_class(data):
    X, y, _ = data
    assert set(y) == set(INCIDENT_CLASSES)
    assert X.shape[1] == len(FEATURE_NAMES)


def test_warmup_ticks_are_dropped(data):
    """Cold-start ratios are all exactly 1.0, an artifact, not data."""
    _, _, ts = data
    assert ts.min() >= WARMUP_TICKS


# ---- the split is honest ----

def test_temporal_split_trains_on_the_past_only(data):
    X, y, ts = data
    cut = np.quantile(ts, 0.7)
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    assert len(Xtr) + len(Xte) == len(X)
    assert len(ytr) == len(Xtr) and len(yte) == len(Xte)
    assert (ts < cut).sum() == len(Xtr)


def test_temporal_split_raises_when_a_class_lands_on_one_side_only():
    """A plain time cut can leave a whole class out of a side, and then macro-F1 is silently a
    subset average. Measured at the sparse default: incidents are rare episodes, so this is
    the normal case rather than an edge one, hence a raise, not a warning."""
    X, y, ts = build_dataset(generate(n_ticks=3000, seed=7))
    with pytest.raises(ValueError, match="absent from (train|test)"):
        temporal_split(X, y, ts)


def test_evaluate_refuses_to_score_a_missing_class(fitted):
    model, Xte, yte = fitted
    keep = yte != "bad_deploy"
    with pytest.raises(ValueError, match="no examples to score"):
        evaluate(model, Xte[keep], yte[keep])


def test_random_split_inflates_the_score(data):
    """The leak, measured. An episode spans 40-300 near-identical ticks; shuffling scores the
    model on ticks whose neighbours it memorized.

    Worth noting *why* this test is possible: on the earlier over-clean generator the honest
    score was already 0.97, and the leak was worth only +0.02, invisible. A leak only shows
    up where there is headroom for it.
    """
    X, y, ts = data
    honest = evaluate(train(*temporal_split(X, y, ts)[0::2]), *temporal_split(X, y, ts)[1::2])
    rXtr, rXte, rytr, ryte = random_split(X, y, ts)
    leaky = evaluate(train(rXtr, rytr), rXte, ryte)
    assert leaky["macro_f1"] > honest["macro_f1"] + 0.05


# ---- the model works, and its weakness is where we said it is ----

def test_beats_the_majority_baseline(fitted):
    """`normal` is ~85% of ticks. Predicting it always scores 0.85 accuracy and 0.18 macro-F1
    while missing every incident, which is why macro-F1 is the metric."""
    model, Xte, yte = fitted
    assert evaluate(model, Xte, yte)["macro_f1"] > 0.6


def test_every_class_is_actually_detected(fitted):
    """A macro average can hide a class scoring zero."""
    model, Xte, yte = fitted
    for cls, s in evaluate(model, Xte, yte)["per_class"].items():
        assert s["recall"] > 0.2, f"{cls} recall {s['recall']:.2f}, effectively undetected"


def test_dependency_failure_is_the_hard_class(fitted):
    """Pins the honest weakness. A retry storm drives CPU, latency and errors up together ,
    a bad deploy's exact shape, and benign deploys ship often enough that one recently
    landed. If this ever becomes easy, check the generator got less realistic rather than
    the model getting smarter."""
    per = evaluate(*fitted)["per_class"]
    assert per["dependency_failure"]["recall"] < per["traffic_spike"]["recall"]


def test_dependency_failures_are_mistaken_for_bad_deploys(fitted):
    """The specific confusion a log-reading step would need to resolve."""
    model, Xte, yte = fitted
    res = evaluate(model, Xte, yte)
    i = res["labels"].index("dependency_failure")
    row = res["confusion"][i]
    wrong = [(res["labels"][j], v) for j, v in enumerate(row) if j != i and v > 0]
    assert wrong, "dependency_failure is perfectly classified, generator too clean?"
    assert max(wrong, key=lambda kv: kv[1])[0] == "bad_deploy"


def test_predictions_are_valid_classes(fitted):
    model, Xte, _ = fitted
    assert set(model.predict(Xte)) <= set(INCIDENT_CLASSES)


def test_training_is_deterministic(data):
    X, y, ts = data
    Xtr, Xte, ytr, _ = temporal_split(X, y, ts)
    assert (train(Xtr, ytr, seed=0).predict(Xte) == train(Xtr, ytr, seed=0).predict(Xte)).all()


def test_importances_cover_every_feature(fitted):
    names = [n for n, _ in importances(fitted[0])]
    assert sorted(names) == sorted(FEATURE_NAMES)
