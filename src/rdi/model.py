"""The hot-path classifier — what kind of incident is this?

WHAT THE MODEL DOES NOT PREDICT: `label`. The SLO breach is a deterministic function of two
features (`latency_ms > 500 or error_rate > 0.05`), so a classifier on it would score ~1.0 and
have learned nothing — it would be an if-statement wearing a model's clothes. `label` is an
*observation*: it tells M4 whether an intervention was warranted. It is not a target.

WHAT THE MODEL DOES PREDICT: `incident_type`, one of five classes. This is the question the
rest of the system needs answered, because the four incident types demand four different
remediations (docs/04-domain.md) and picking the wrong one is worse than picking none. The
hard pair is dependency_failure vs traffic_spike: both blow the latency SLO, and they demand
opposite actions (failover vs scale_out). Nothing in the raw metrics separates them except
`cpu_over_baseline`.

THE SPLIT IS TEMPORAL, NOT RANDOM. An incident episode spans 40-300 near-identical ticks. A
random split scatters ticks from the *same episode* across train and test, so the model is
scored on ticks whose neighbours it memorized — the classic streaming-ML leak. `random_split`
is kept as a measured counterexample; see `make score` for the gap it opens.
"""
from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMClassifier

from rdi.features import FEATURE_NAMES, compute_offline

# LightGBM stamps feature_names_in_ = ['Column_0', ...] on every fit, including a fit on an
# unnamed ndarray, and it is a read-only property (deleting it silently no-ops). sklearn then
# warns on each ndarray predict about a mismatch against placeholders that never carried
# meaning. The fix sklearn wants — DataFrames in and out — would put per-event DataFrame
# construction on a sub-ms hot path. Filtered by exact message so a real name mismatch
# elsewhere would still surface. pytest re-applies this in pyproject's filterwarnings.
warnings.filterwarnings("ignore", message="X does not have valid feature names",
                        category=UserWarning)

INCIDENT_CLASSES = ["normal", "memory_leak", "dependency_failure", "traffic_spike", "bad_deploy"]

# The first ticks of a stream have no history, so every *_over_baseline ratio is exactly 1.0
# (see OnlineFeatures._baseline). Training on those teaches the model that a unit ratio is
# normal-looking noise; they are a cold-start artifact, not data.
WARMUP_TICKS = 60.0


def target_of(event: dict) -> str:
    return event["incident_type"] or "normal"


def build_dataset(events: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream ALL events through the online features once, then return (X, y, ts).

    The whole stream is featurized in one pass on purpose: splitting first and featurizing
    each side separately would let the test set start with cold windows, which never happens
    in production — the consumer's windows are always warm by the time it sees a new event.
    """
    ordered = sorted(events, key=lambda e: e["ts"])
    X = compute_offline(ordered)
    y = np.array([target_of(e) for e in ordered])
    ts = np.array([e["ts"] for e in ordered], dtype=float)

    keep = ts >= WARMUP_TICKS
    return X[keep], y[keep], ts[keep]


def temporal_split(X, y, ts, frac: float = 0.7, require_all_classes: bool = True):
    """Split at a point in TIME — train on the past, test on the future. The honest one.

    Raises if a class lands entirely on one side. Incidents are sparse episodes, so a plain
    time cut can easily leave a whole class out of the test set — and then the score is a
    silent average over the classes that happened to survive. Measured: at 3000 ticks,
    `bad_deploy` had zero test examples and the macro-F1 was quietly a 4-class number.
    Better to fail here than to report an average over a class that wasn't scored.
    """
    cut = np.quantile(ts, frac)
    tr, te = ts < cut, ts >= cut
    if require_all_classes:
        for side, mask in (("train", tr), ("test", te)):
            missing = set(np.unique(y)) - set(np.unique(y[mask]))
            if missing:
                raise ValueError(
                    f"{sorted(missing)} absent from {side} — the split would score a subset of "
                    f"the classes and call it macro. Lengthen the stream or raise "
                    f"incidents_per_service."
                )
    return X[tr], X[te], y[tr], y[te]


def random_split(X, y, ts, frac: float = 0.7, seed: int = 0):
    """The leaky counterexample: shuffle ticks, then split.

    Kept to be measured, not used. Consecutive ticks of one episode are near-duplicates, so
    this puts a memory leak's tick 41 in train and its tick 42 in test and calls that
    generalization. It answers "can the model recognize an episode it has already seen?" —
    which is not a question anyone has in production.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(frac * len(X))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def train(X: np.ndarray, y: np.ndarray, seed: int = 0) -> LGBMClassifier:
    """Small on purpose. This runs inline on the hot path against a sub-ms SLO, and the
    signal is a handful of engineered ratios rather than something needing depth."""
    model = LGBMClassifier(
        n_estimators=120, num_leaves=15, learning_rate=0.1, min_child_samples=10,
        class_weight="balanced",  # normal is ~85% of ticks; without this the rare types lose
        random_state=seed, verbose=-1, n_jobs=1,
    )
    # Fit on the raw array, not a named frame: the hot path scores a plain list of floats, and
    # building a DataFrame per event to satisfy sklearn would cost more than the model does.
    #
    # LightGBM then stamps feature_names_in_ = ['Column_0', ...] anyway — a read-only property
    # derived from the booster — so sklearn warns on every ndarray predict about a mismatch
    # against placeholder names that never meant anything. It cannot be unset (deleting it
    # silently does nothing), so it is ignored by name in pyproject's filterwarnings. The real
    # names live in FEATURE_NAMES; see `importances`.
    model.fit(X, y)
    return model


def importances(model: LGBMClassifier) -> list[tuple[str, int]]:
    return sorted(zip(FEATURE_NAMES, model.feature_importances_, strict=True),
                  key=lambda kv: -kv[1])


def evaluate(model: LGBMClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Per-class recall/precision plus macro-F1.

    Macro, not accuracy: `normal` is ~72% of ticks, so a model that predicted `normal` for
    everything would score 0.72 accuracy and miss every incident. Macro-F1 weights the rare
    classes — the only ones anyone cares about — equally.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    pred = model.predict(X)
    missing = set(INCIDENT_CLASSES) - set(y)
    if missing:
        raise ValueError(f"{sorted(missing)} has no examples to score — macro-F1 over the "
                         f"survivors would silently be a different metric")
    labels = list(INCIDENT_CLASSES)
    p, r, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "labels": labels,
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                "support": int(support[i])}
            for i, c in enumerate(labels)
        },
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }
