"""Tests for the log-reading layer.

The one that matters is `test_reading_logs_fixes_the_confusion_metrics_cannot`: without that
lift the layer is cost with no return and should be deleted.

Everything here runs on `MockReader`, which needs no network. That keeps the suite free, and
it also means these tests say nothing about how a real model performs. The mock matches the
phrases the generator writes, so it is near-perfect by construction. `test_mock_is_a_fixture_
not_a_result` pins that caveat where someone will see it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.metrics import f1_score

from rdi.events import REMEDIATIONS, generate
from rdi.model import build_dataset, evaluate, temporal_split, train
from rdi.reasoning import CLASSES, ClaudeReader, MockReader, Verdict, is_grounded, resolve


@pytest.fixture(scope="module")
def stream():
    return sorted(generate(n_ticks=6000, seed=7, incidents_per_service=10),
                  key=lambda e: e["ts"])


@pytest.fixture(scope="module")
def scored(stream):
    X, y, ts = build_dataset(stream)
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    clf = train(Xtr, ytr)
    cut = np.quantile(ts, 0.7)
    warm = [e for e in stream if e["ts"] >= 60.0]
    te_events = [e for e, keep in zip(warm, ts >= cut, strict=True) if keep]
    return clf, Xte, yte, te_events, evaluate(clf, Xte, yte)


# ---- the reader reads ----

@pytest.mark.parametrize("kind", ["memory_leak", "dependency_failure", "traffic_spike",
                                  "bad_deploy"])
def test_each_incident_is_recognisable_from_its_log_line(stream, kind):
    reader = MockReader()
    events = [e for e in stream if e["incident_type"] == kind][:40]
    assert events, f"no {kind} in the stream"
    hits = sum(reader.read(e).incident == kind for e in events)
    assert hits / len(events) > 0.9


def test_normal_traffic_is_not_flagged(stream):
    reader = MockReader()
    normal = [e for e in stream if e["incident_type"] is None][:60]
    assert all(reader.read(e).incident == "normal" for e in normal)


def test_verdict_carries_the_matching_remediation(stream):
    reader = MockReader()
    for e in (e for e in stream if e["incident_type"]):
        v = reader.read(e)
        if v.incident != "normal":
            assert v.remediation == REMEDIATIONS[v.incident]


def test_verdict_quotes_the_phrase_it_used(stream):
    reader = MockReader()
    e = next(e for e in stream if e["incident_type"] == "dependency_failure")
    v = reader.read(e)
    assert v.evidence and v.evidence in e["log"]


def test_explanations_cite_a_real_value(stream):
    reader = MockReader()
    sample = [e for e in stream if e["incident_type"]][:50]
    assert all(reader.read(e).grounded for e in sample)


def test_groundedness_rejects_invented_numbers(stream):
    e = stream[0]
    assert is_grounded(f"latency was {e['latency_ms']:.0f}ms", e)
    assert not is_grounded("latency was 999999ms and cpu hit 42.7%", e)


# ---- the reason the layer exists ----

def test_reading_logs_fixes_the_confusion_metrics_cannot(scored):
    """The whole justification, measured.

    The hot path calls roughly half of all dependency failures a bad deploy, because a retry
    storm has the same metric shape as a bad release. The log names the failing upstream, so
    reading it should push that recall up sharply.
    """
    clf, Xte, yte, te_events, base = scored
    pred = list(clf.predict(Xte))
    after = resolve(te_events, pred, MockReader())

    dep = [(t, p) for t, p in zip(yte, after, strict=True) if t == "dependency_failure"]
    recall_after = sum(p == t for t, p in dep) / len(dep)
    recall_before = base["per_class"]["dependency_failure"]["recall"]

    assert recall_before < 0.7, "the confusion this layer targets is gone, re-check the scorer"
    assert recall_after > recall_before + 0.3
    assert f1_score(yte, after, average="macro", zero_division=0) > base["macro_f1"]


def test_only_the_ambiguous_classes_get_escalated(scored):
    """Reading every event would cost more than it returns, so most keep their fast label."""
    clf, Xte, yte, te_events, _ = scored
    pred = list(clf.predict(Xte))
    reader = MockReader()
    resolve(te_events, pred, reader)

    escalated = sum(p in {"bad_deploy", "dependency_failure"} for p in pred)
    assert reader.stats.calls == escalated
    assert escalated / len(pred) < 0.25, "escalating this much defeats the point"


def test_unescalated_predictions_are_left_alone(scored):
    clf, Xte, yte, te_events, _ = scored
    pred = list(clf.predict(Xte))
    after = resolve(te_events, pred, MockReader(), only={"dependency_failure"})
    for before, now in zip(pred, after, strict=True):
        if before != "dependency_failure":
            assert now == before


def test_escalation_set_is_configurable(scored):
    clf, Xte, _, te_events, _ = scored
    pred = list(clf.predict(Xte))
    reader = MockReader()
    resolve(te_events, pred, reader, only={"memory_leak"})
    assert reader.stats.calls == sum(p == "memory_leak" for p in pred)


# ---- honesty guards ----

def test_mock_is_a_fixture_not_a_result(stream):
    """Pinned so the mock's score is never quoted as a model result.

    It matches the exact phrases `events._log_line` writes. Scoring above 0.95 here means the
    two files agree with each other, which is not evidence about any real model.
    """
    reader = MockReader()
    labelled = [e for e in stream if e["incident_type"]][:200]
    acc = sum(reader.read(e).incident == e["incident_type"] for e in labelled) / len(labelled)
    assert acc > 0.95, "mock no longer tracks the generator's vocabulary"


def test_reader_returns_a_known_class_for_junk_text(stream):
    e = dict(stream[0], log="\x00 nothing parseable here \x00")
    assert MockReader().read(e).incident in CLASSES


def test_stats_count_every_call(stream):
    reader = MockReader()
    for e in stream[:25]:
        reader.read(e)
    assert reader.stats.calls == 25


# ---- the real provider ----

def test_claude_reader_needs_credentials():
    """Constructing without any provider configured should fail loudly, not silently mock."""
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("credentials present in the environment")
    with pytest.raises(RuntimeError, match="no credentials"):
        ClaudeReader()


@pytest.mark.skipif(not os.getenv("RDI_LIVE"), reason="set RDI_LIVE=1 to spend real tokens")
def test_claude_reader_live(stream):
    """Opt-in, because it costs money. Never runs by default."""
    reader = ClaudeReader()
    e = next(e for e in stream if e["incident_type"] == "dependency_failure")
    v = reader.read(e)
    assert v.incident in CLASSES
    assert reader.stats.failures == 0, v.explanation


def test_verdict_is_a_plain_record():
    v = Verdict("bad_deploy", "rollback", "x", "y")
    assert (v.incident, v.remediation, v.grounded, v.source) == ("bad_deploy", "rollback",
                                                                 False, "mock")
