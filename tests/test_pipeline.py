"""End-to-end tests.

Every layer is tested on its own elsewhere. These exist for the failures that only appear when
the layers are joined: a feature vector that doesn't survive the trip from scorer to policy, a
reader that gets called on the hot path, an escalation that silently drops an event.
"""
from __future__ import annotations

import pytest

from rdi.decision import ACTIONS
from rdi.events import generate
from rdi.pipeline import PipelineResult, demo, fit_all, render, run
from rdi.reasoning import MockReader


@pytest.fixture(scope="module")
def parts():
    events = sorted(generate(n_ticks=3000, seed=7, incidents_per_service=10),
                    key=lambda e: e["ts"])
    cut = int(0.6 * len(events))
    return fit_all(events[:cut], seed=7), events[cut:]


def test_every_event_makes_it_through(parts):
    fitted, live = parts
    res = run(live, *fitted)
    assert res.processed == len(live)
    assert sum(res.incidents.values()) == len(live)


def test_actions_are_decided_for_every_event(parts):
    fitted, live = parts
    res = run(live, *fitted)
    assert sum(res.actions.values()) == len(live)
    assert set(res.actions) <= set(ACTIONS)
    assert len(res.decisions) == len(live)


def test_the_loop_acts_on_a_minority_and_leaves_the_rest(parts):
    """Acting on everything would be the failure mode this whole design avoids."""
    fitted, live = parts
    res = run(live, *fitted)
    assert 0 < res.acted < len(live) * 0.5
    assert res.actions["none"] > res.acted


def test_policy_value_is_positive(parts):
    """The system as a whole has to be worth running."""
    fitted, live = parts
    assert run(live, *fitted).policy_value > 0


def test_log_reading_changes_some_verdicts(parts):
    """If reading logs never overturned the scorer, the layer would be pure cost."""
    fitted, live = parts
    res = run(live, *fitted)
    assert res.escalated > 0
    assert res.resolved_by_log > 0
    assert res.resolved_by_log <= res.escalated


def test_reader_is_never_called_on_the_whole_stream(parts):
    """Escalation is meant to be selective."""
    fitted, live = parts
    reader = MockReader()
    res = run(live, *fitted, reader=reader)
    assert reader.stats.calls == res.escalated
    assert reader.stats.calls < len(live) * 0.5


def test_hot_path_stays_fast_with_every_layer_attached(parts):
    """Loose bound: the point is that reading logs and deciding actions stay off the hot path.

    A regression that put a model call inline would cost milliseconds per event and blow this
    by an order of magnitude. Real numbers come from `make pipeline` in a quiet process.
    """
    fitted, live = parts
    res = run(live, *fitted)
    assert res.p50_ms < 5.0
    assert len(res.hot_latencies_ms) == len(live)


def test_runs_through_a_durable_log(tmp_path, parts):
    """The same results whether events arrive from memory or from the broker."""
    fitted, live = parts
    direct = run(live, *fitted)
    through_log = run(live, *fitted, log_path=str(tmp_path / "s.jsonl"))
    assert through_log.processed == direct.processed
    assert through_log.incidents == direct.incidents


def test_anomaly_detection_still_runs_batched(parts):
    fitted, live = parts
    res = run(live, *fitted)
    assert res.anomalies > 0


def test_fit_all_refuses_a_stream_it_cannot_calibrate_on():
    """A class seen once can only land on one side of any split, so q would be unsound."""
    ev = sorted(generate(n_ticks=400, seed=7, incidents_per_service=1), key=lambda e: e["ts"])
    with pytest.raises(ValueError):
        fit_all(ev[:120])


def test_render_reports_the_numbers_it_measured(parts):
    fitted, live = parts
    res = run(live, *fitted)
    out = render(res)
    for header in ("DETECT", "EXPLAIN", "ACT", "MONITOR"):
        assert header in out
    assert f"{res.processed:,}" in out
    assert "policy value" in out


def test_render_handles_an_empty_run():
    assert "events processed" in render(PipelineResult())


def test_demo_is_reproducible():
    a, b = demo(n_ticks=1200, seed=3), demo(n_ticks=1200, seed=3)
    assert (a.processed, a.escalated, a.acted) == (b.processed, b.escalated, b.acted)
    assert a.policy_value == pytest.approx(b.policy_value)


def test_drift_separates_a_shifted_baseline_from_a_stream_full_of_incidents(parts):
    """The claim is separation, not perfection.

    The live slice has real incidents and no baseline shift, so retraining would be the wrong
    response. The drifted slice is the opposite. Measured here: roughly 8% of clean windows
    alert against 100% of drifted ones. A monitor that never false-alarms on a noisy stream
    would be suspicious, so this asserts the gap rather than a zero.
    """
    fitted, live = parts
    clean = run(live, *fitted)
    drifted = run(sorted(generate(n_ticks=1500, seed=7, drifted=True), key=lambda e: e["ts"]),
                  *fitted)

    clean_rate = clean.drift_alerts / max(clean.processed // 600, 1)
    drift_rate = drifted.drift_alerts / max(drifted.processed // 600, 1)
    assert clean_rate < 0.25, f"{clean_rate:.0%} of clean windows alerted, that is a retrain loop"
    assert drift_rate > 0.75, f"only {drift_rate:.0%} of drifted windows alerted"
    assert drift_rate > clean_rate * 3
