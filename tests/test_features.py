"""Tests for online features and the train=serve guarantee.

Three groups, in order of what they're worth:

  1. parity, the offline path and the streaming path produce identical vectors
  2. no-lookahead, every feature at event i depends only on events <= i
  3. fingerprints, each feature actually discriminates the incident it was added for

(1) alone is nearly a tautology, since compute_offline replays the online code on purpose.
It is here to fail loudly if someone later "optimizes" the training path into a second
implementation. (2) is the test with teeth: it fails for any implementation that reaches into
the future, which is what a vectorized batch rewrite naturally does.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from rdi.broker import Broker
from rdi.consumer import Consumer
from rdi.events import generate
from rdi.features import (
    FEATURE_NAMES,
    NO_DEPLOY_S,
    OnlineFeatures,
    compute_offline,
    compute_offline_leaky,
)

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


@pytest.fixture(scope="module")
def events() -> list[dict]:
    return generate(n_ticks=400, seed=7)


def _tick(service="checkout-api", ts=0.0, latency_ms=100.0, error_rate=0.001,
          cpu_pct=40.0, mem_pct=50.0, rps=200.0, version="v1.0.0") -> dict:
    return {"service": service, "ts": ts, "latency_ms": latency_ms, "error_rate": error_rate,
            "cpu_pct": cpu_pct, "mem_pct": mem_pct, "rps": rps, "version": version}


def _rows_for(events, matrix, kind):
    ordered = sorted(events, key=lambda e: e["ts"])
    idx = [i for i, e in enumerate(ordered) if e["incident_type"] == kind]
    return matrix[idx]


def _normal_rows(events, matrix):
    ordered = sorted(events, key=lambda e: e["ts"])
    idx = [i for i, e in enumerate(ordered) if e["incident_type"] is None]
    return matrix[idx]


# ---- shape & sanity ----

def test_vector_matches_declared_feature_names(events):
    assert len(OnlineFeatures().update_and_extract(events[0])) == len(FEATURE_NAMES)


def test_offline_matrix_has_a_row_per_event(events):
    assert compute_offline(events).shape == (len(events), len(FEATURE_NAMES))


def test_features_are_finite(events):
    assert np.isfinite(compute_offline(events)).all()


def test_first_tick_of_a_service_has_unit_ratios():
    """No history yet: the only honest baseline is the current value, so ratios are 1.0."""
    v = OnlineFeatures().update_and_extract(_tick())
    assert v[F["latency_over_baseline"]] == pytest.approx(1.0)
    assert v[F["cpu_over_baseline"]] == pytest.approx(1.0)
    assert v[F["rps_over_baseline"]] == pytest.approx(1.0)
    assert v[F["mem_slope_per_min"]] == 0.0


# ---- 1. train = serve ----

def test_offline_and_streamed_features_are_identical(tmp_path, events):
    """The parity claim: the training matrix and what the live consumer computes are the
    same numbers, because they are the same code."""
    offline = compute_offline(events)

    with Broker(str(tmp_path / "s.jsonl")) as b:
        for e in events:
            b.append(e)
        streamed = Consumer("c", b, "g").run(batch=32).features

    assert len(streamed) == len(offline)
    online = np.array([f for _, f in sorted(streamed, key=lambda x: x[0])], dtype=float)
    np.testing.assert_allclose(online, offline, rtol=0, atol=0)


def test_batch_size_does_not_change_features(tmp_path, events):
    """Features must be a function of the stream, not of how it was chunked."""
    out = []
    for batch in (1, 7, 256):
        with Broker(str(tmp_path / f"s{batch}.jsonl")) as b:
            for e in events:
                b.append(e)
            res = Consumer(f"c{batch}", b, "g").run(batch=batch)
        out.append(np.array([f for _, f in sorted(res.features)], dtype=float))
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=0)
    np.testing.assert_allclose(out[0], out[2], rtol=0, atol=0)


# ---- 2. no lookahead (the test with teeth) ----

def test_truncating_the_future_does_not_change_the_past(events):
    """Causality, stated as a property: features for the first k events must be bit-identical
    whether or not the rest of the stream exists. Any implementation that normalizes against
    a full-series statistic fails this, see the leaky counterexample below."""
    ordered = sorted(events, key=lambda e: e["ts"])
    k = len(ordered) // 3
    np.testing.assert_allclose(
        compute_offline(ordered[:k]), compute_offline(ordered)[:k], rtol=0, atol=0)


def test_a_late_incident_cannot_affect_earlier_features():
    """Sharpened version: bolt an outage onto the end and the earlier rows must not move."""
    calm = [_tick(ts=float(i), latency_ms=100.0) for i in range(60)]
    outage = [_tick(ts=float(60 + i), latency_ms=5000.0) for i in range(20)]
    np.testing.assert_allclose(
        compute_offline(calm), compute_offline(calm + outage)[:len(calm)], rtol=0, atol=0)


def test_baseline_excludes_the_current_event():
    """A spike must not contribute to the baseline it is measured against, or a big enough
    spike normalizes itself away and the model never sees it."""
    warmup = [_tick(ts=float(i), latency_ms=100.0) for i in range(30)]
    spike = _tick(ts=30.0, latency_ms=1000.0)
    f = OnlineFeatures()
    for e in warmup:
        f.update_and_extract(e)
    assert f.update_and_extract(spike)[F["latency_over_baseline"]] == pytest.approx(10.0, rel=0.01)


def test_services_do_not_contaminate_each_other():
    """State is per service. A fleet-wide mean would let a noisy neighbor mask an outage."""
    f = OnlineFeatures()
    for i in range(30):
        f.update_and_extract(_tick(service="a", ts=float(i), latency_ms=100.0))
        f.update_and_extract(_tick(service="b", ts=float(i), latency_ms=1000.0))
    v = f.update_and_extract(_tick(service="a", ts=30.0, latency_ms=100.0))
    assert v[F["latency_over_baseline"]] == pytest.approx(1.0, rel=0.01)


def test_a_sustained_outage_does_not_normalize_itself_away():
    """The failure a rolling MEAN baseline has: a long incident poisons the very baseline it
    is measured against, so `latency_over_baseline` decays toward 1.0 and the feature goes
    blind exactly when the outage is worst.

    Caught by test_leaky_baseline_understates_incidents_worst_where_it_matters, which found
    the online path decaying 2.88 -> 1.16 across a single bad_deploy episode. Incidents run
    40-300 ticks (docs/04-domain.md), so this is the common case, not a corner.
    """
    warmup = [_tick(ts=float(i), latency_ms=100.0) for i in range(300)]
    outage = [_tick(ts=float(300 + i), latency_ms=300.0) for i in range(250)]
    m = compute_offline(warmup + outage)
    col = F["latency_over_baseline"]
    first, last = m[300, col], m[-1, col]
    assert first == pytest.approx(3.0, rel=0.05), "outage not visible at onset"
    assert last > 2.5, f"outage decayed to {last:.2f}x, baseline ate the incident"


def test_baseline_survives_a_long_memory_leak(events):
    """memory_leak runs up to 300 ticks, the longest episode, so the worst poisoning case."""
    m = compute_offline(events)
    leak = _rows_for(events, m, "memory_leak")
    late = leak[len(leak) // 2:][:, F["latency_over_baseline"]]
    assert late.mean() > 1.2, "late-leak latency ratio collapsed toward 1.0"


def test_feature_extraction_stays_inside_the_hot_path_budget():
    """The hot path has a sub-ms SLO and the model still has to fit inside it.

    The median baseline is O(window) per metric per event, which is a real cost worth guarding:
    measured p99 is ~0.3ms at a steady-state 901-deep window. The threshold here is deliberately
    loose, this is a guard against a catastrophic regression (an accidental sort, a window that
    stops evicting), not a benchmark. A tight bound would just flake on a busy laptop.
    """
    events = generate(n_ticks=350, seed=5)
    f = OnlineFeatures()
    warm = len(events) // 2
    for e in events[:warm]:
        f.update_and_extract(e)

    lat = []
    for e in events[warm:]:
        t0 = time.perf_counter()
        f.update_and_extract(e)
        lat.append((time.perf_counter() - t0) * 1000)

    assert np.percentile(lat, 99) < 2.0, f"p99 {np.percentile(lat, 99):.2f}ms, hot path at risk"


def test_baseline_window_stops_growing_at_steady_state():
    """The eviction that makes the above affordable. Unbounded windows would turn the median
    into a slow leak that only shows up hours into a run."""
    f = OnlineFeatures(baseline_s=50.0)
    for i in range(400):
        f.update_and_extract(_tick(ts=float(i)))
    assert len(f._hist["checkout-api"]["latency_ms"]) <= 52


def test_baseline_window_ages_out_old_events():
    f = OnlineFeatures(baseline_s=10.0)
    for i in range(5):
        f.update_and_extract(_tick(ts=float(i), latency_ms=1000.0))
    # Jump far past the window: the old 1000ms ticks are gone, so 100ms is the new normal.
    f.update_and_extract(_tick(ts=500.0, latency_ms=100.0))
    v = f.update_and_extract(_tick(ts=501.0, latency_ms=100.0))
    assert v[F["latency_over_baseline"]] == pytest.approx(1.0, rel=0.01)


# ---- the leaky counterexample, quantified ----

def test_leaky_offline_features_diverge_from_online(events):
    """The classic feature-store skew, in feature space.

    A per-service mean over the whole series is what you get by building training features in
    pandas and serving features in a stream worker. It is unavailable at serving time and it
    disagrees with the online path, so a model trained on it is trained on inputs that will
    never exist in production.
    """
    online = compute_offline(events)
    leaky = compute_offline_leaky(events)
    col = F["latency_over_baseline"]
    assert not np.allclose(online[:, col], leaky[:, col])


def test_leaky_baseline_understates_incidents_worst_where_it_matters(events):
    """The damage is not uniform, it is concentrated on the events that matter.

    A service's whole-series mean latency includes its own outages, so an outage is scored
    against an inflated 'normal' and reads MILDER than it truly is. The feature is least
    trustworthy exactly on the incident ticks it exists to flag.
    """
    online = compute_offline(events)
    leaky = compute_offline_leaky(events)
    col = F["latency_over_baseline"]
    for kind in ("dependency_failure", "bad_deploy"):
        assert _rows_for(events, leaky, kind)[:, col].mean() < \
               _rows_for(events, online, kind)[:, col].mean(), \
            f"{kind}: leaky baseline did not understate the incident"


def test_leaky_features_look_into_the_future(events):
    """The property test that the honest implementation passes and this one cannot."""
    ordered = sorted(events, key=lambda e: e["ts"])
    k = len(ordered) // 3
    assert not np.allclose(compute_offline_leaky(ordered[:k]),
                           compute_offline_leaky(ordered)[:k])


# ---- 3. each feature discriminates what it was added for ----

def test_memory_leak_shows_a_positive_memory_slope(events):
    m = compute_offline(events)
    leak = _rows_for(events, m, "memory_leak")[:, F["mem_slope_per_min"]]
    normal = _normal_rows(events, m)[:, F["mem_slope_per_min"]]
    assert leak.mean() > 0.5
    assert leak.mean() > normal.mean() + 0.4


def test_dependency_failure_cpu_is_bimodal_not_uniformly_low(events):
    """CPU is NOT the clean discriminator the domain doc first claimed.

    Blocked threads push CPU down, but clients retry and retries burn CPU, so ~a third of
    dependency-failure ticks run CPU *above* baseline. The distribution is bimodal, and its
    mean (~0.9) therefore describes no actual tick. The earlier version of this test asserted
    `mean < 0.9` and passed on an average that misrepresented a third of the data.
    """
    m = compute_offline(events)
    dep = _rows_for(events, m, "dependency_failure")[:, F["cpu_over_baseline"]]
    below, above = float((dep < 1.0).mean()), float((dep > 1.0).mean())
    assert below > 0.4, f"only {below:.0%} of dep ticks run CPU below baseline"
    assert above > 0.15, f"only {above:.0%} run CPU above baseline, retry storms missing"


def test_dependency_failure_still_blows_latency(events):
    m = compute_offline(events)
    assert _rows_for(events, m, "dependency_failure")[:, F["latency_over_baseline"]].mean() > 1.5


def test_traffic_spike_shows_cpu_and_rps_above_baseline(events):
    m = compute_offline(events)
    spike = _rows_for(events, m, "traffic_spike")
    assert spike[:, F["rps_over_baseline"]].mean() > 1.5
    assert spike[:, F["cpu_over_baseline"]].mean() > 1.2


def test_rps_not_cpu_is_what_separates_the_two_high_latency_incidents(events):
    """Corrects the original claim. Both incidents blow latency and demand opposite
    remediations (failover vs scale_out); the question is which feature actually tells them
    apart once retry storms exist.

    Not CPU: the ranges overlap, so no threshold on it separates the classes.
    It is `rps_over_baseline`, a traffic spike genuinely has traffic and a failing
    dependency does not. Boring, and true.
    """
    m = compute_offline(events)
    dep_cpu = _rows_for(events, m, "dependency_failure")[:, F["cpu_over_baseline"]]
    spike_cpu = _rows_for(events, m, "traffic_spike")[:, F["cpu_over_baseline"]]
    assert max(dep_cpu.min(), spike_cpu.min()) <= min(dep_cpu.max(), spike_cpu.max()), \
        "CPU ranges no longer overlap, the retry storms that make this realistic are gone"

    dep_rps = _rows_for(events, m, "dependency_failure")[:, F["rps_over_baseline"]]
    spike_rps = _rows_for(events, m, "traffic_spike")[:, F["rps_over_baseline"]]
    assert dep_rps.max() < spike_rps.max()
    assert dep_rps.mean() < 1.5 < spike_rps.mean()


def test_bad_deploy_is_recent_after_a_deploy(events):
    m = compute_offline(events)
    deploys = _rows_for(events, m, "bad_deploy")[:, F["since_deploy_s"]]
    assert deploys.mean() < NO_DEPLOY_S
    assert (deploys < 300.0).mean() > 0.5


def test_no_deploy_seen_yields_the_sentinel_not_a_false_signal():
    """First sighting is not a deploy, we started watching, we didn't witness one. Stamping
    it would fire a bad_deploy signal for every service at t=0."""
    f = OnlineFeatures()
    for i in range(10):
        assert f.update_and_extract(_tick(ts=float(i)))[F["since_deploy_s"]] == NO_DEPLOY_S


def test_a_version_change_stamps_a_deploy():
    f = OnlineFeatures()
    for i in range(5):
        f.update_and_extract(_tick(ts=float(i), version="v1.0.0"))
    assert f.update_and_extract(_tick(ts=5.0, version="v1.0.1"))[F["since_deploy_s"]] == 0.0
    assert f.update_and_extract(_tick(ts=9.0, version="v1.0.1"))[F["since_deploy_s"]] == 4.0
