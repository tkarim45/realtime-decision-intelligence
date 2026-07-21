"""Keeping the model alive: drift detection, canary releases, rollback.

Two jobs. Notice when the world has moved far enough that the model's inputs no longer look
like what it trained on, and ship a replacement without betting the fleet on it.

THE TRAP THIS MODULE IS BUILT AROUND. Population Stability Index compares a live window against
a training reference and fires when they diverge. Run it naively on this stream and it fires on
incidents, not on drift, which is exactly backwards. An outage moves mean latency further than a
fleet-wide baseline shift does: measured here, a clean stream's mean latency sits *above* a
drifted stream's, because a handful of large incident spikes outweigh a 1.8x shift spread over
every tick.

Two fixes, both needed. Build the reference from healthy ticks only, so incidents in the
training window don't get baked into "normal". Compare on a robust statistic, since a rolling
mean is dragged around by the same spikes. `psi` takes quantile bins, which helps, and
`DriftMonitor` adds a median-shift check that a mean-based one fails.

The distinction that matters operationally: drift means retrain, an incident means remediate.
A monitor that confuses them either retrains on an outage or pages someone about a Tuesday.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Above ~0.2 is the usual "investigate" line.

    Bin edges come from the reference quantiles rather than a fixed grid, so the measure
    doesn't depend on the units of whatever is being compared.
    """
    reference, live = np.asarray(reference, float), np.asarray(live, float)
    if len(reference) == 0 or len(live) == 0:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(live, edges)[0] / len(live), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@dataclass
class DriftAlert:
    at_event: int
    psi: float
    median_ratio: float
    reason: str


class DriftMonitor:
    """Windowed PSI plus a median-shift check, against a healthy-only reference.

    Requiring both to trip is what separates drift from incidents. A burst of outages moves PSI
    (the tail changes) but barely moves the median, since most ticks in the window are still
    normal. A genuine baseline shift moves both, because every tick moved.
    """

    def __init__(self, reference: np.ndarray, window: int = 500, threshold: float = 0.2,
                 median_tolerance: float = 0.25) -> None:
        if len(reference) == 0:
            raise ValueError("empty reference, the monitor would have nothing to compare against")
        self.reference = np.asarray(reference, float)
        self.ref_median = float(np.median(self.reference))
        self.window = window
        self.threshold = threshold
        self.median_tolerance = median_tolerance
        self.alerts: list[DriftAlert] = []
        self._buf: list[float] = []
        self._seen = 0

    def observe(self, value: float) -> DriftAlert | None:
        self._buf.append(float(value))
        self._seen += 1
        if len(self._buf) < self.window:
            return None
        live = np.asarray(self._buf)
        self._buf = []
        score = psi(self.reference, live)
        ratio = float(np.median(live) / max(self.ref_median, 1e-9))
        shifted = abs(ratio - 1.0) > self.median_tolerance
        if score > self.threshold and shifted:
            alert = DriftAlert(self._seen, round(score, 3), round(ratio, 3),
                               "distribution and median both moved")
            self.alerts.append(alert)
            return alert
        return None

    @property
    def fired(self) -> bool:
        return bool(self.alerts)


def healthy_reference(events: list[dict], metric: str = "latency_ms") -> np.ndarray:
    """Reference built from non-incident ticks only.

    Including incidents teaches the monitor that outages are normal, which is the quiet way a
    drift detector stops working.
    """
    vals = [e[metric] for e in events if e.get("incident_type") is None]
    if not vals:
        raise ValueError("no healthy ticks to build a reference from")
    return np.asarray(vals, float)


# ---- shipping a new model ----

@dataclass
class CanaryResult:
    promoted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    traffic_fraction: float
    events_seen: int


@dataclass
class Deployment:
    """Which model is live, with the previous one kept for instant rollback."""
    live: object
    previous: object | None = None
    history: list[str] = field(default_factory=list)

    def promote(self, candidate: object, note: str = "") -> None:
        self.previous = self.live
        self.live = candidate
        self.history.append(f"promote {note}".strip())

    def rollback(self, note: str = "") -> None:
        if self.previous is None:
            raise ValueError("nothing to roll back to")
        self.live, self.previous = self.previous, self.live
        self.history.append(f"rollback {note}".strip())


def shadow(baseline, candidate, X: np.ndarray) -> dict:
    """Score both models on live traffic while only the baseline is allowed to act.

    Free to run and the safest way to find out whether a candidate disagrees before it can do
    anything about it.
    """
    a, b = baseline.predict(X), candidate.predict(X)
    return {"n": len(X), "disagreement": float((a != b).mean()),
            "baseline": a, "candidate": b}


def canary(baseline, candidate, X: np.ndarray, y: np.ndarray, score,
           fraction: float = 0.1, min_events: int = 200, tolerance: float = 0.02,
           seed: int = 0) -> CanaryResult:
    """Route a slice of traffic to the candidate and promote only if it doesn't regress.

    `tolerance` allows a small dip so ordinary noise doesn't block every release. Refusing to
    promote on too little traffic matters more than the threshold: a candidate that looks
    better over 20 events hasn't been measured, it's been sampled.
    """
    rng = np.random.default_rng(seed)
    pick = rng.random(len(X)) < fraction
    n = int(pick.sum())
    base = float(score(baseline, X[pick], y[pick])) if n else 0.0
    cand = float(score(candidate, X[pick], y[pick])) if n else 0.0

    if n < min_events:
        return CanaryResult(False, f"only {n} events, need {min_events}", base, cand, fraction, n)
    if cand < base - tolerance:
        return CanaryResult(False, f"regression: {cand:.3f} vs {base:.3f}", base, cand,
                            fraction, n)
    return CanaryResult(True, f"no regression: {cand:.3f} vs {base:.3f}", base, cand, fraction, n)


def macro_f1(model, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, model.predict(X), average="macro", zero_division=0))
