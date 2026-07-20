"""Split-conformal prediction sets — per-decision uncertainty with a coverage guarantee.

Adapted from the sibling `conformal-prediction` repo (LAC / Least Ambiguous set-valued
Classifier). On a calibration set the nonconformity of the true label is `s = 1 - p[true]`;
take the finite-sample-corrected (1-alpha) quantile q̂, and the set for a new point is every
class with `p_y >= 1 - q̂`. The guarantee is distribution-free and finite-sample:
P(true in set) >= 1 - alpha, no matter how miscalibrated the model's softmax is.

WHY THIS EARNS ITS PLACE HERE, specifically. M2 measured a real confusion: dependency_failure
is misread as bad_deploy 84/184 times, because a retry storm and a bad deploy have the same
metric shape. A bare argmax reports `bad_deploy` with a confident-looking probability and the
system rolls back a deploy that was fine while the upstream stays broken. A conformal set
reports `{bad_deploy, dependency_failure}` — which is the honest answer, and an actionable
one: an ambiguous set is precisely the signal to send the event to M3's log-reading agent
rather than auto-remediating on it.

So set SIZE is not a nuisance metric here; it is the routing decision. Singleton => act.
Ambiguous => escalate to reasoning. Empty => nothing looks plausible, escalate too.

CALIBRATION MUST BE ITS OWN TEMPORAL SLICE. Calibrating on training data would use points the
model has already fit, making nonconformity scores optimistically small and the guarantee a
fiction. The split is train -> calibrate -> test, all in time order.
"""
from __future__ import annotations

import numpy as np


def qhat(p_calib: np.ndarray, y_idx: np.ndarray, alpha: float) -> float:
    """The (1-alpha) quantile of calibration nonconformity, finite-sample corrected."""
    n = len(y_idx)
    scores = 1.0 - p_calib[np.arange(n), y_idx]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(p: np.ndarray, q: float, classes: list[str]) -> list[set[str]]:
    return [{classes[j] for j in np.where(row >= 1.0 - q)[0]} for row in p]


def naive_sets(p: np.ndarray, alpha: float, classes: list[str]) -> list[set[str]]:
    """What people actually do: take the argmax, add anything above the nominal confidence.

    Kept to be measured. It never grows the set much, so its coverage plateaus near the
    model's accuracy and it simply cannot deliver a 95% guarantee on an 86%-accurate model.
    It trusts the softmax to mean what it says.
    """
    out = []
    for row in p:
        s = {classes[int(row.argmax())]}
        s.update(classes[int(j)] for j in np.where(row >= 1.0 - alpha)[0])
        out.append(s)
    return out


def coverage(sets: list[set[str]], y_true: np.ndarray) -> dict:
    sizes = np.array([len(s) for s in sets])
    return {
        "coverage": float(np.mean([t in s for s, t in zip(sets, y_true, strict=True)])),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "ambiguous_rate": float((sizes > 1).mean()),
        "empty_rate": float((sizes == 0).mean()),
    }


def routing(sets: list[set[str]]) -> dict:
    """How the stream splits into act / escalate — the operational read of set size."""
    sizes = np.array([len(s) for s in sets])
    return {
        "act_singleton": float((sizes == 1).mean()),
        "escalate_ambiguous": float((sizes > 1).mean()),
        "escalate_empty": float((sizes == 0).mean()),
    }
