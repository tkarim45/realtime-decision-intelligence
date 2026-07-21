"""Tests for drift detection and safe rollout.

The one that matters is `test_drift_monitor_tells_drift_apart_from_incidents`. A monitor that
fires on outages is worse than none, because it triggers retraining on exactly the data you
don't want to learn from. `test_naive_psi_cannot_tell_them_apart` keeps the counterexample
measured rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.events import generate
from rdi.model import build_dataset, temporal_split, train
from rdi.ops import (
    Deployment,
    DriftMonitor,
    canary,
    healthy_reference,
    macro_f1,
    psi,
    shadow,
)


@pytest.fixture(scope="module")
def reference():
    return healthy_reference(generate(n_ticks=3000, seed=1))


@pytest.fixture(scope="module")
def streams():
    return (generate(n_ticks=1500, seed=7),
            generate(n_ticks=1500, seed=7, drifted=True))


def _run(ref, events, **kw):
    m = DriftMonitor(ref, window=kw.pop("window", 600), **kw)
    for e in events:
        m.observe(e["latency_ms"])
    return m


# ---- psi mechanics ----

def test_psi_is_zero_against_itself(reference):
    assert psi(reference, reference) < 0.01


def test_psi_grows_with_the_size_of_the_shift(reference):
    small = psi(reference, reference * 1.1)
    large = psi(reference, reference * 2.0)
    assert 0 < small < large


def test_psi_handles_empty_input(reference):
    assert psi(reference, np.array([])) == 0.0
    assert psi(np.array([]), reference) == 0.0


# ---- the claim ----

def test_drift_monitor_tells_drift_apart_from_incidents(reference, streams):
    """Silent on a stream full of outages, loud on one where only the baseline moved.

    The clean stream contains real incidents. The drifted stream contains none, just a
    fleet-wide shift. Retraining is the right response to the second and the wrong response to
    the first, so the monitor has to separate them.
    """
    clean, drifted = streams
    assert not _run(reference, clean).fired, "fired on incidents, that's a false retrain"
    assert _run(reference, drifted).fired, "missed a genuine baseline shift"


def test_naive_psi_cannot_tell_them_apart(streams):
    """Why the reference is healthy-only and why the median gate exists.

    Reference built from every tick, PSI alone as the rule: it fires on about as many clean
    windows as drifted ones. A few large incident spikes move the distribution more than a
    1.8x shift spread across every tick does.
    """
    clean, drifted = streams
    naive_ref = np.array([e["latency_ms"] for e in generate(n_ticks=3000, seed=1)])

    def windows_firing(events):
        vals = [e["latency_ms"] for e in events]
        return sum(psi(naive_ref, np.array(vals[i:i + 600])) > 0.2
                   for i in range(0, len(vals) - 600, 600))

    assert windows_firing(clean) > 0, "the naive monitor should misfire, that is the point"
    assert abs(windows_firing(clean) - windows_firing(drifted)) <= 3, \
        "naive PSI now separates them, re-check whether the fix is still needed"


def test_alert_records_what_moved(reference, streams):
    alert = _run(reference, streams[1]).alerts[0]
    assert alert.psi > 0.2
    assert alert.median_ratio > 1.2
    assert alert.at_event > 0


def test_healthy_reference_excludes_incidents():
    ev = generate(n_ticks=800, seed=4)
    ref = healthy_reference(ev)
    assert len(ref) < len(ev)
    assert ref.max() < max(e["latency_ms"] for e in ev)


def test_reference_needs_healthy_ticks():
    with pytest.raises(ValueError, match="no healthy ticks"):
        healthy_reference([{"latency_ms": 1.0, "incident_type": "memory_leak"}])
    with pytest.raises(ValueError, match="empty reference"):
        DriftMonitor(np.array([]))


def test_monitor_waits_for_a_full_window(reference):
    m = DriftMonitor(reference, window=500)
    for _ in range(499):
        assert m.observe(9999.0) is None
    assert m.observe(9999.0) is not None


# ---- rollout ----

@pytest.fixture(scope="module")
def models():
    X, y, ts = build_dataset(generate(n_ticks=4000, seed=7, incidents_per_service=10))
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    good = train(Xtr, ytr, seed=0)
    weak = train(Xtr[:400], ytr[:400], seed=0)   # starved on purpose
    return good, weak, Xte, yte


def test_canary_promotes_a_model_that_holds_up(models):
    good, _, Xte, yte = models
    r = canary(good, good, Xte, yte, macro_f1, fraction=0.5)
    assert r.promoted and r.events_seen >= 200


def test_canary_blocks_a_regression(models):
    good, weak, Xte, yte = models
    r = canary(good, weak, Xte, yte, macro_f1, fraction=0.5)
    assert not r.promoted
    assert "regression" in r.reason
    assert r.candidate_score < r.baseline_score


def test_canary_refuses_to_judge_on_too_little_traffic(models):
    """A candidate that looks better over 20 events hasn't been measured."""
    good, weak, Xte, yte = models
    r = canary(good, weak, Xte, yte, macro_f1, fraction=0.002)
    assert not r.promoted and "need" in r.reason


def test_shadow_scores_without_acting(models):
    good, weak, Xte, _ = models
    s = shadow(good, weak, Xte)
    assert s["n"] == len(Xte)
    assert 0.0 < s["disagreement"] < 1.0
    assert (s["baseline"] == good.predict(Xte)).all()


def test_promote_then_rollback_restores_the_previous_model(models):
    good, weak, _, _ = models
    d = Deployment(live=good)
    d.promote(weak, "candidate v2")
    assert d.live is weak and d.previous is good
    d.rollback("regression")
    assert d.live is good and d.previous is weak
    assert [h.split()[0] for h in d.history] == ["promote", "rollback"]


def test_rollback_needs_something_to_roll_back_to(models):
    with pytest.raises(ValueError, match="nothing to roll back"):
        Deployment(live=models[0]).rollback()


def test_full_loop_drift_then_blocked_promotion(reference, streams, models):
    """Drift fires, a weak retrain is proposed, the canary refuses it, live model is untouched."""
    good, weak, Xte, yte = models
    assert _run(reference, streams[1]).fired

    d = Deployment(live=good)
    result = canary(d.live, weak, Xte, yte, macro_f1, fraction=0.5)
    if result.promoted:
        d.promote(weak)
    assert not result.promoted
    assert d.live is good and d.previous is None
