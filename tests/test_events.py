"""Tests for the M0 stream generator.

These assert the properties every downstream milestone depends on: that the labels mean what
docs/04-domain.md says they mean, that each incident type actually carries its documented
fingerprint (a generator that emits indistinguishable incidents would make the whole scoring
layer a fiction), and that the drifted stream shifts without breaking.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.events import (
    INCIDENT_TYPES,
    METRIC_NAMES,
    REMEDIATIONS,
    SERVICES,
    SLO_ERROR_RATE,
    SLO_LATENCY_MS,
    generate,
)

SCHEMA = {
    "ts": float, "service": str, "version": str, "latency_ms": float, "error_rate": float,
    "cpu_pct": float, "mem_pct": float, "rps": float, "log": str, "label": int,
}


@pytest.fixture(scope="module")
def stream() -> list[dict]:
    return generate(n_ticks=900, seed=7)


def _of_type(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e["incident_type"] == kind]


def _mean(events: list[dict], key: str) -> float:
    return float(np.mean([e[key] for e in events]))


# ---- schema & determinism ----

def test_schema_complete_on_every_event(stream):
    for e in stream:
        for field, typ in SCHEMA.items():
            assert isinstance(e[field], typ), f"{field} is {type(e[field])}, expected {typ}"
        assert e["incident_type"] is None or e["incident_type"] in INCIDENT_TYPES


def test_one_event_per_service_per_tick(stream):
    assert len(stream) == 900 * len(SERVICES)
    assert {e["service"] for e in stream} == set(SERVICES)


def test_same_seed_same_stream():
    assert generate(n_ticks=100, seed=3) == generate(n_ticks=100, seed=3)


def test_different_seed_different_stream():
    assert generate(n_ticks=100, seed=3) != generate(n_ticks=100, seed=4)


def test_metrics_stay_in_physical_bounds(stream):
    for e in stream:
        assert 0.0 <= e["cpu_pct"] <= 100.0
        assert 0.0 <= e["mem_pct"] <= 100.0
        assert 0.0 <= e["error_rate"] <= 1.0
        assert e["latency_ms"] > 0.0
        assert e["rps"] > 0.0


# ---- the label is derived from the SLO, not painted on ----

def test_label_is_exactly_the_slo_predicate(stream):
    for e in stream:
        expected = int(e["latency_ms"] > SLO_LATENCY_MS or e["error_rate"] > SLO_ERROR_RATE)
        assert e["label"] == expected


def test_normal_ticks_almost_never_breach(stream):
    normal = [e for e in stream if e["incident_type"] is None]
    breach_rate = sum(e["label"] for e in normal) / len(normal)
    # Blips are allowed and intended; a normal stream that breaches often is a broken baseline.
    assert breach_rate < 0.01, f"normal breach rate {breach_rate:.2%} — baseline too hot"


def test_incidents_are_a_minority_of_the_stream(stream):
    incident_rate = sum(e["incident_type"] is not None for e in stream) / len(stream)
    assert 0.02 < incident_rate < 0.40


# ---- anomaly != incident (the schema's load-bearing idea) ----

def test_every_incident_type_appears(stream):
    present = {e["incident_type"] for e in stream} - {None}
    assert present == set(INCIDENT_TYPES)


def test_absorbed_traffic_spikes_exist_and_do_not_breach(stream):
    """The uplift engine's reason to exist: a real anomaly that must NOT be acted on."""
    spikes = _of_type(stream, "traffic_spike")
    unbreaching = [e for e in spikes if e["label"] == 0]
    assert unbreaching, "no absorbed spike ticks — uplift has no negative-value case to learn"
    assert len(unbreaching) / len(spikes) > 0.20


def test_memory_leak_has_non_breaching_early_ticks(stream):
    """Early detection is only worth something if the leak is real before it breaches."""
    leaks = _of_type(stream, "memory_leak")
    assert any(e["label"] == 0 for e in leaks)
    assert any(e["label"] == 1 for e in leaks)


def test_incident_type_and_label_are_not_redundant(stream):
    """If label were just (incident_type is not None), every downstream claim collapses."""
    anomalous = [e for e in stream if e["incident_type"] is not None]
    assert any(e["label"] == 0 for e in anomalous), "every anomaly breaches — fields redundant"


# ---- each incident type carries its documented fingerprint ----

def test_memory_leak_ramps_memory_upward(stream):
    leaks = _of_type(stream, "memory_leak")
    normal_mem = _mean([e for e in stream if e["incident_type"] is None], "mem_pct")
    assert _mean(leaks, "mem_pct") > normal_mem + 10.0


def test_memory_leak_memory_climbs_within_an_episode():
    """Monotonic ramp is the discriminator vs every step-change incident."""
    events = [e for e in generate(n_ticks=900, seed=11) if e["incident_type"] == "memory_leak"]
    by_service: dict[str, list[dict]] = {}
    for e in events:
        by_service.setdefault(e["service"], []).append(e)
    checked = 0
    for series in by_service.values():
        if len(series) < 60:
            continue
        mem = [e["mem_pct"] for e in series]
        # Compare episode halves rather than adjacent ticks — noise is real, the trend is the claim.
        assert np.mean(mem[len(mem) // 2:]) > np.mean(mem[:len(mem) // 2])
        checked += 1
    assert checked, "no memory_leak episode long enough to test the ramp"


def test_dependency_failure_shows_high_errors_with_flat_cpu(stream):
    """The fingerprint that separates it from traffic_spike: latency up, CPU *down*.

    Both incidents blow the latency SLO. Only one has the service actually doing work — a
    scorer that can't tell them apart will pick the wrong remediation every time.
    """
    deps = _of_type(stream, "dependency_failure")
    normal = [e for e in stream if e["incident_type"] is None]
    assert _mean(deps, "error_rate") > 10 * _mean(normal, "error_rate")
    assert _mean(deps, "latency_ms") > _mean(normal, "latency_ms")
    assert _mean(deps, "cpu_pct") < _mean(normal, "cpu_pct")


def test_traffic_spike_shows_high_rps_and_high_cpu(stream):
    spikes = _of_type(stream, "traffic_spike")
    normal = [e for e in stream if e["incident_type"] is None]
    assert _mean(spikes, "rps") > 2.0 * _mean(normal, "rps")
    assert _mean(spikes, "cpu_pct") > _mean(normal, "cpu_pct")


def test_traffic_spike_and_dependency_failure_are_separable_by_cpu(stream):
    """Stated as its own test because M2's whole job rests on it."""
    assert _mean(_of_type(stream, "traffic_spike"), "cpu_pct") > \
           _mean(_of_type(stream, "dependency_failure"), "cpu_pct")


def test_bad_deploy_bumps_the_version(stream):
    """The deploy marker is the causal evidence the LLM agent cites to justify rollback."""
    by_service: dict[str, list[dict]] = {}
    for e in stream:
        by_service.setdefault(e["service"], []).append(e)
    bumped = False
    for series in by_service.values():
        for prev, cur in zip(series, series[1:], strict=False):
            if cur["version"] != prev["version"]:
                assert cur["incident_type"] == "bad_deploy"
                bumped = True
    assert bumped, "no deploy occurred — bad_deploy has no marker to explain"


def test_bad_deploy_raises_latency_and_errors_together(stream):
    deploys = _of_type(stream, "bad_deploy")
    normal = [e for e in stream if e["incident_type"] is None]
    assert _mean(deploys, "latency_ms") > _mean(normal, "latency_ms")
    assert _mean(deploys, "error_rate") > _mean(normal, "error_rate")


# ---- logs carry signal for the LLM ----

def test_log_lines_are_incident_flavored(stream):
    for kind, needle in [("memory_leak", "heap"), ("dependency_failure", "timeout"),
                         ("traffic_spike", "queue depth"), ("bad_deploy", "Exception")]:
        sample = _of_type(stream, kind)
        assert sample, f"no {kind} events"
        assert all(needle in e["log"] for e in sample)


def test_bad_deploy_log_cites_the_deployed_version(stream):
    for e in _of_type(stream, "bad_deploy"):
        assert e["version"] in e["log"]


# ---- drift: the M5 specificity fixture ----

def test_drifted_stream_injects_no_incidents():
    """Drift means the world changed, not that something broke. If this stream carried
    incidents, a drift monitor firing on it would prove nothing."""
    drifted = generate(n_ticks=600, seed=7, drifted=True)
    assert all(e["incident_type"] is None for e in drifted)


def test_drifted_stream_shifts_latency_distribution():
    """Drift moves the BASELINE, so it must be measured against normal ticks.

    Comparing against the whole clean stream would compare a shifted baseline to a mean
    inflated by incident spikes — see test_incidents_inflate_clean_mean_above_drifted_mean.
    """
    clean_normal = [e for e in generate(n_ticks=600, seed=7) if e["incident_type"] is None]
    drifted = generate(n_ticks=600, seed=7, drifted=True)
    assert _mean(drifted, "latency_ms") > 1.4 * _mean(clean_normal, "latency_ms")


def test_incidents_inflate_clean_mean_above_drifted_mean():
    """A confound M5 has to survive, pinned here so it can't silently disappear.

    The clean stream's MEAN latency (incidents included) exceeds the drifted stream's, even
    though the drifted baseline is 1.8x higher — a few big incident spikes outweigh a
    fleet-wide shift. So a PSI monitor comparing live traffic against a training reference
    will read incidents as drift unless incident ticks are excluded from the reference or the
    window is chosen to outlast an episode. Naive PSI here fires on the wrong stream.
    """
    clean_all = generate(n_ticks=600, seed=7)
    drifted = generate(n_ticks=600, seed=7, drifted=True)
    assert _mean(clean_all, "latency_ms") < _mean(drifted, "latency_ms")
    assert _mean(clean_all, "latency_ms") > 1.4 * _mean(
        [e for e in clean_all if e["incident_type"] is None], "latency_ms")


def test_drifted_stream_stays_mostly_within_slo():
    """Inflated but not broken — otherwise 'drift' and 'incident' are indistinguishable
    and S7's specificity claim is untestable."""
    drifted = generate(n_ticks=600, seed=7, drifted=True)
    assert sum(e["label"] for e in drifted) / len(drifted) < 0.10


# ---- domain constants ----

def test_every_incident_type_has_a_remediation():
    assert set(REMEDIATIONS) == set(INCIDENT_TYPES)


def test_remediation_is_not_leaked_onto_events(stream):
    """M4 has to recover the mapping from metrics. Shipping it inline would be the answer key."""
    for e in stream[:100]:
        assert "remediation" not in e
        assert "action" not in e


def test_metric_names_match_the_event_schema(stream):
    for name in METRIC_NAMES:
        assert name in stream[0]
