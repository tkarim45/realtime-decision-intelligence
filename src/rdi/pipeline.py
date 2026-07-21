"""The whole loop, wired together.

Every other module does one job and is tested on its own. This runs them as one system, which
is the only way to catch the failures that live between components rather than inside them.

Per event:

    durable log -> online features -> classifier -> conformal set -> act or escalate
                                                          |
                        escalated events -> read the log line -> resolved class
                                                          |
                                          uplift effects -> choose an action, or none
                                                          |
                             latency -> drift monitor -> retrain signal

Nothing here calls a model over the network on the hot path. Log reading happens after an
event has already been scored and routed, and the anomaly detector runs batched on its own
consumer group, for the reason the scoring module goes into: a single-row IsolationForest call
costs 5.9ms against a sub-millisecond budget.

The console view is deliberately plain text. A browser dashboard would mean a second toolchain
and a node_modules tree, and this project's whole argument is that it installs with three
dependencies and runs offline.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rdi.anomaly import fit_on_normal
from rdi.broker import Broker
from rdi.conformal import qhat
from rdi.decision import (
    ACTION_COST,
    TLearner,
    policy_from_effects,
    policy_value,
    simulate_experiment,
)
from rdi.events import generate
from rdi.features import FEATURE_NAMES
from rdi.model import WARMUP_TICKS, build_dataset, train
from rdi.ops import DriftMonitor, healthy_reference
from rdi.reasoning import MockReader
from rdi.scoring import HotPathScorer, NearLineDetector


@dataclass
class PipelineResult:
    processed: int = 0
    acted: int = 0
    escalated: int = 0
    resolved_by_log: int = 0
    anomalies: int = 0
    drift_alerts: int = 0
    actions: Counter = field(default_factory=Counter)
    incidents: Counter = field(default_factory=Counter)
    hot_latencies_ms: list[float] = field(default_factory=list)
    policy_value: float = 0.0
    decisions: list = field(default_factory=list)

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.hot_latencies_ms, 99)) if self.hot_latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.hot_latencies_ms, 50)) if self.hot_latencies_ms else 0.0

    @property
    def throughput_eps(self) -> float:
        if not self.hot_latencies_ms:
            return 0.0
        return 1000.0 / max(float(np.mean(self.hot_latencies_ms)), 1e-9)


def fit_all(train_events: list[dict], alpha: float = 0.05, seed: int = 0):
    """Train every component on a past slice, in time order."""
    X, y, ts = build_dataset(train_events)

    # Calibration is a random holdout from the training pool, not a later time slice.
    #
    # A time cut fails here for a structural reason rather than a tuning one: incidents are
    # sparse episodes, and a class can sit entirely inside one side of any cut. Measured on the
    # default stream, every bad_deploy in the training window lands after the 0.8 quantile, so
    # no cut puts that class on both sides, and conformal then fits q on a subset of the classes
    # and quietly stops guaranteeing coverage.
    #
    # Splitting at random inside the training pool costs nothing that matters. Both halves are
    # already in the past, so the model still never sees the future, and split-conformal only
    # asks that calibration points weren't fitted on. It also keeps every class represented,
    # which the guarantee needs.
    rng = np.random.default_rng(seed)
    holdout = rng.random(len(X)) < 0.3
    tr, ca = ~holdout, holdout

    missing = set(y[ca]) - set(y[tr])
    if missing or not ca.sum():
        raise ValueError(
            f"{sorted(missing)} landed only in calibration. The stream is too short or has too "
            f"few incidents to calibrate on. Raise n_ticks or incidents_per_service."
        )

    model = train(X[tr], y[tr], seed=seed)
    detector = fit_on_normal(X[tr], y[tr])
    classes = list(model.classes_)
    q = qhat(model.predict_proba(X[ca]), np.array([classes.index(v) for v in y[ca]]), alpha)

    warm = [e for e in sorted(train_events, key=lambda e: e["ts"]) if e["ts"] >= WARMUP_TICKS]
    exp = simulate_experiment(warm, seed=seed + 1)
    learner = TLearner(seed=seed).fit(X, exp["action"], exp["outcome"])

    reference = healthy_reference(train_events)
    return model, detector, q, learner, reference


def run(events: list[dict], model, detector, q, learner, reference, *,
        reader=None, log_path: str | None = None, drift_window: int = 600) -> PipelineResult:
    """Stream events through the full system and record what it did."""
    reader = reader or MockReader()
    res = PipelineResult()
    scorer = HotPathScorer(model, q)
    near = NearLineDetector(detector, batch=256)
    monitor = DriftMonitor(reference, window=drift_window)

    ordered = sorted(events, key=lambda e: e["ts"])
    broker = Broker(log_path) if log_path else None
    if broker:
        for e in ordered:
            broker.append(e)
        ordered = broker.claim("pipeline", "worker", len(ordered))

    feats_for_policy, pending = [], []
    for e in ordered:
        t0 = time.perf_counter()
        d = scorer.score(e)
        res.hot_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        res.processed += 1

        # Off the measured hot path from here down.
        if broker:
            broker.ack("pipeline", e["offset"])
        res.anomalies += sum(1 for _, flag in near.submit(d) if flag)

        incident = d.incident
        if d.action == "escalate" or incident in ("dependency_failure", "bad_deploy"):
            res.escalated += 1
            verdict = reader.read(e, dict(zip(FEATURE_NAMES, d.features, strict=True)))
            if verdict.incident != incident:
                res.resolved_by_log += 1
            incident = verdict.incident

        res.incidents[incident] += 1
        feats_for_policy.append(d.features)
        pending.append(e)

        if monitor.observe(e["latency_ms"]):
            res.drift_alerts += 1

    if broker:
        broker.close()
    res.anomalies += sum(1 for _, flag in near.flush() if flag)

    if feats_for_policy:
        effects = learner.effects(np.array(feats_for_policy))
        actions = policy_from_effects(effects)
        res.actions.update(actions)
        res.acted = sum(a != "none" for a in actions)
        res.policy_value = policy_value(pending, actions)
        res.decisions = list(zip([e["service"] for e in pending], actions, strict=True))
    return res


def render(res: PipelineResult) -> str:
    """Plain-text view of one run."""
    lines = [
        "REAL-TIME DECISION PIPELINE",
        "",
        f"  events processed      {res.processed:,}",
        f"  hot path p50 / p99    {res.p50_ms:.3f} / {res.p99_ms:.3f} ms",
        f"  throughput            {res.throughput_eps:,.0f} events/s",
        "",
        "  DETECT",
    ]
    for cls, n in res.incidents.most_common():
        lines.append(f"    {cls:<22}{n:>7,}")
    lines += [
        f"    anomalies flagged     {res.anomalies:>7,}  (near-line, batched)",
        "",
        "  EXPLAIN",
        f"    escalated to log      {res.escalated:>7,}  "
        f"({res.escalated / max(res.processed, 1):.1%} of stream)",
        f"    reclassified          {res.resolved_by_log:>7,}  the metrics had these wrong",
        "",
        "  ACT",
    ]
    for a, n in res.actions.most_common():
        cost = ACTION_COST[a] * n
        lines.append(f"    {a:<22}{n:>7,}   cost {cost:>8.1f}")
    lines += [
        f"    policy value          {res.policy_value:>7.1f}  breaches avoided net of cost",
        "",
        "  MONITOR",
        f"    drift alerts          {res.drift_alerts:>7}",
    ]
    return "\n".join(lines)


def demo(n_ticks: int = 3000, seed: int = 7) -> PipelineResult:
    """Train on the first 60% of a stream, then run the rest through the full system."""
    events = sorted(generate(n_ticks=n_ticks, seed=seed, incidents_per_service=10),
                    key=lambda e: e["ts"])
    cut = int(0.6 * len(events))
    parts = fit_all(events[:cut], seed=seed)
    return run(events[cut:], *parts)
