"""The hot path — one event in, one decision out, under a sub-millisecond budget.

Online features -> classifier -> conformal set -> a routing decision. This is what the stream
consumer calls per event, so every line in `score` is on the latency budget.

WHAT COMES OUT, and why it is not just a label:

    incident      the argmax class — what to remediate, if we act
    conformal_set the coverage-guaranteed set of plausible classes
    action        act | escalate — the only field the rest of the system reads

`action` is `escalate` when the conformal set is not a singleton, i.e. the model itself
hesitates. Everything else is `act`.

THE ANOMALY DETECTOR IS NOT HERE, and that is a measured decision rather than an oversight.
It was on this path first. Measured per-component cost:

    features              p50 0.043ms
    model.predict_proba   p50 0.182ms
    detector.flag         p50 5.033ms      <- 99% of the budget, 28x the classifier

sklearn's per-call dispatch dominates: the same IsolationForest costs 0.0284ms/event at batch
256, so single-row scoring is ~176x more expensive per event than batched. It scales with tree
count (25 trees 0.73ms, 200 trees 5.1ms), which is the signature of per-tree Python overhead
rather than real compute. (The sibling `model-serving` repo measured the same effect from the
other side: micro-batching bought it 8.0x throughput.)

Paying 5ms per event for it would blow a sub-ms SLO by 7x. And what it buys per event is weak
anyway — M2 measured that it rescues under a quarter of the classifier's misses and that
"classifier says normal, detector disagrees" carries no signal above the base rate. So the
detector moved to `NearLineDetector`: same broker, its own consumer group, batched inference
at ~0.03ms/event. It still catches the unknown-unknowns it is there for (~77% of an incident
class the classifier was never trained on); it just does so a batch later instead of inline.

This is the architecture's own thesis applied one layer down. The docs always said the LLM
must stay off the hot path; measurement said the anomaly detector must too, for exactly the
same reason.

AND MOVING IT OFF THE PATH IS NOT ENOUGH — IT HAS TO LEAVE THE THREAD. Running the batched
detector in the same loop, with its cost explicitly excluded from the timing, still degrades
the hot path:

    hot path alone                p50 0.358ms  p99 0.633ms  max  2.89ms   2,679 eps
    + near-line detector in-loop  p50 0.377ms  p99 1.525ms  max 20.52ms   2,097 eps

p99 inflates 2.4x and the tail 7x from work that is not being counted, because a 7ms batched
tree walk every 256 events evicts cache and churns the allocator under the next few events.
(It is not GC — disabling the collector does not help.) The near-line consumer is a separate
consumer group by design, so in production it belongs in a separate process. "Don't count it"
is not latency isolation; only separation is.

An honest caveat, pinned in tests: escalation does NOT catch the confusion that matters most.
When the model misreads a dependency failure as a bad deploy it does so at ~0.97 probability,
so the set stays a confident singleton and the event routes to `act`. Escalation covers
hesitation, not confident error. Resolving that needs the log line — M3's job, not the hot
path's.

NO LLM HERE, EVER. Anything that blocks on a network call belongs on the slow path or the
throughput number below is fiction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rdi.anomaly import AnomalyDetector
from rdi.conformal import prediction_sets, qhat
from rdi.features import OnlineFeatures


@dataclass
class Decision:
    service: str
    ts: float
    incident: str
    conformal_set: set[str]
    action: str            # "act" | "escalate"
    reason: str
    features: np.ndarray   # handed to the near-line detector; not recomputed there


class HotPathScorer:
    """Stateful per-event scorer. One instance per consumer — it owns the feature windows."""

    def __init__(self, model, q: float, features: OnlineFeatures | None = None) -> None:
        self.model = model
        self.q = q
        self.features = features or OnlineFeatures()
        self.classes = list(model.classes_)

    def score(self, event: dict) -> Decision:
        feats = np.asarray(self.features.update_and_extract(event), dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        incident = self.classes[int(proba.argmax())]
        cset = prediction_sets(proba.reshape(1, -1), self.q, self.classes)[0]

        if len(cset) != 1:
            action, reason = "escalate", f"ambiguous set {sorted(cset)}"
        else:
            action, reason = "act", f"confident {incident}"

        return Decision(event["service"], event["ts"], incident, cset, action, reason, feats[0])


class NearLineDetector:
    """The anomaly detector, off the hot path and batched.

    Accumulates feature vectors and scores them in one call when the buffer fills. At batch
    256 this costs ~0.03ms/event against ~5ms/event single-row — the entire reason it is not
    inline. `flush` drains a partial buffer so a quiet stream still gets scored.
    """

    def __init__(self, detector: AnomalyDetector, batch: int = 256) -> None:
        self.detector = detector
        self.batch = batch
        self._buf: list[np.ndarray] = []
        self._meta: list[Decision] = []

    def submit(self, decision: Decision) -> list[tuple[Decision, bool]]:
        self._buf.append(decision.features)
        self._meta.append(decision)
        return self.flush() if len(self._buf) >= self.batch else []

    def flush(self) -> list[tuple[Decision, bool]]:
        if not self._buf:
            return []
        flags = self.detector.flag(np.vstack(self._buf))
        out = list(zip(self._meta, [bool(f) for f in flags], strict=True))
        self._buf, self._meta = [], []
        return out


def fit_hot_path(X, y, ts, alpha: float = 0.05, seed: int = 0):
    """Train every hot-path component on a train/calibrate split, in time order.

    Raises if calibration holds a class training never saw. Incidents are sparse episodes, so
    a three-way time cut drops a class fairly easily — and the failure is silent-by-default:
    the class has no column in `predict_proba`, so its nonconformity cannot be computed and
    q̂ would quietly be fitted on a subset, breaking the very guarantee conformal exists to
    provide. Same reasoning as `model.temporal_split`: raise rather than mis-score.
    """
    from rdi.anomaly import fit_on_normal
    from rdi.model import train

    c1, c2 = np.quantile(ts, 0.55), np.quantile(ts, 0.75)
    tr, ca = ts < c1, (ts >= c1) & (ts < c2)
    model = train(X[tr], y[tr], seed=seed)
    detector = fit_on_normal(X[tr], y[tr])
    classes = list(model.classes_)

    unseen = set(y[ca]) - set(classes)
    if unseen:
        raise ValueError(
            f"{sorted(unseen)} appear in calibration but not in training, so q̂ would be "
            f"fitted on a subset of the classes and the coverage guarantee would not hold. "
            f"Lengthen the stream or raise incidents_per_service."
        )

    y_idx = np.array([classes.index(v) for v in y[ca]])
    return model, detector, qhat(model.predict_proba(X[ca]), y_idx, alpha), (ts >= c2)


def latency_profile(scorer: HotPathScorer, events: list[dict], warmup: int = 500,
                    detector: NearLineDetector | None = None) -> dict:
    """Per-event wall-clock, warm windows only. Cold-start ticks would flatter the numbers
    (empty deques are cheap to median), so they are excluded rather than averaged in."""
    for e in events[:warmup]:
        scorer.score(e)

    lat, actions, flagged = [], {"act": 0, "escalate": 0}, 0
    t_start = time.perf_counter()
    for e in events[warmup:]:
        t0 = time.perf_counter()
        d = scorer.score(e)
        lat.append((time.perf_counter() - t0) * 1000.0)  # hot path only — see below
        actions[d.action] += 1
        if detector is not None:
            # Off the measured hot path on purpose: this is a different consumer group in the
            # real system, so its cost must not be attributed to per-event scoring latency.
            flagged += sum(1 for _, f in detector.submit(d) if f)
    wall = time.perf_counter() - t_start
    if detector is not None:
        flagged += sum(1 for _, f in detector.flush() if f)

    lat_arr = np.array(lat)
    return {
        "n": len(lat),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "max_ms": float(lat_arr.max()),
        "throughput_eps": len(lat) / wall,
        "actions": actions,
        "anomalies_flagged": flagged,
    }
