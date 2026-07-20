"""Unsupervised anomaly detection — the unknown-unknowns layer.

WHY THIS EXISTS ALONGSIDE A CLASSIFIER. The classifier can only ever name an incident it was
trained on. Given something genuinely new — a failure mode nobody has labelled yet — it does
not say "I don't know", it says `normal`, confidently, because that is the nearest class it
has. That is the failure mode this layer covers: not "what kind of incident is this" but "is
this service behaving unlike itself at all".

So it is trained on NORMAL TICKS ONLY. An IsolationForest fitted on the full stream would
learn the incidents as part of the background and stop finding them surprising; fitted on
healthy behaviour alone, anything that departs from healthy is far from the training manifold.
That is what makes it able to flag a class the classifier has never seen — proven in
`tests/test_anomaly.py::test_detector_flags_an_incident_type_the_classifier_never_saw`.

FEATURE SUBSET. Deliberately not all eight. `since_deploy_s` is excluded because a deploy is
not an anomaly — deploys are routine, and including it would make every one of the ~93% benign
deploys look like a deviation. `latency_ms` is excluded because it is an absolute that varies
by an order of magnitude across services, so a healthy slow service would score as anomalous
purely for being slow; `latency_over_baseline` already carries the per-service question ("is
this service unlike ITSELF"), which is the one worth asking.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from rdi.features import FEATURE_NAMES

# The "is this service unlike itself" features. See FEATURE SUBSET above for the exclusions.
ANOMALY_FEATURES = [
    "latency_over_baseline",
    "error_rate",
    "cpu_over_baseline",
    "mem_pct",
    "mem_slope_per_min",
    "rps_over_baseline",
]
_IDX = [FEATURE_NAMES.index(f) for f in ANOMALY_FEATURES]


def anomaly_view(X: np.ndarray) -> np.ndarray:
    return X[:, _IDX]


class AnomalyDetector:
    """IsolationForest over healthy behaviour. `score` is higher = more anomalous."""

    def __init__(self, contamination: float = 0.02, seed: int = 7) -> None:
        # contamination is what fraction of the TRAINING data is expected to be anomalous.
        # Training data is normal-only, so this should be near zero — it exists to absorb the
        # genuine blips a healthy stream contains (a normal tick can breach the SLO on noise).
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed, n_jobs=1,
        )

    def fit(self, X_normal: np.ndarray) -> AnomalyDetector:
        self.model.fit(anomaly_view(X_normal))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. sklearn's score_samples is the opposite sign."""
        return -self.model.score_samples(anomaly_view(X))

    def flag(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(anomaly_view(X)) == -1).astype(int)


def fit_on_normal(X: np.ndarray, y: np.ndarray, **kw) -> AnomalyDetector:
    """Fit on the healthy slice of a labelled training set.

    Note this uses labels — in production you would use a known-good window instead. The
    guarantee it buys is the one that matters: the detector has never seen ANY incident, so
    flagging one is genuinely a departure-from-normal rather than recognition.
    """
    normal = X[y == "normal"]
    if len(normal) == 0:
        raise ValueError("no normal ticks to fit on — the detector would learn nothing")
    return AnomalyDetector(**kw).fit(normal)


def detection_rate(detector: AnomalyDetector, X: np.ndarray, y: np.ndarray) -> dict:
    """Per-class flag rate. On `normal` this is the false-positive rate; on every incident
    class it is recall — and recall here means "noticed", not "identified"."""
    flags = detector.flag(X)
    out = {}
    for cls in sorted(set(y)):
        mask = y == cls
        out[cls] = {"flagged": float(flags[mask].mean()), "n": int(mask.sum())}
    return out
