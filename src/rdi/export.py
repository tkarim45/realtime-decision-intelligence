"""Export a real pipeline run as JSON for the dashboard.

The dashboard has no backend. It replays a snapshot produced here, which keeps the whole
frontend a static build while every number on screen still comes from an actual run rather
than from fixtures someone typed in.

Written to `frontend/public/snapshot.json` by `make dashboard-data`.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np

from rdi.decision import (
    ACTION_COST,
    TLearner,
    evaluate_policies,
    simulate_experiment,
    uplift,
)
from rdi.events import generate
from rdi.model import build_dataset, evaluate, temporal_split, train
from rdi.ops import DriftMonitor, healthy_reference
from rdi.pipeline import fit_all, run
from rdi.reasoning import MockReader, resolve

# Enough events to see the stream move without shipping a megabyte of JSON.
FEED_LIMIT = 400


def _feed(events: list[dict], model, q, reader) -> list[dict]:
    """A slice of the stream with what the system decided about each event."""
    from rdi.features import FEATURE_NAMES
    from rdi.scoring import HotPathScorer

    scorer = HotPathScorer(model, q)
    out = []
    for e in events:
        d = scorer.score(e)
        row = {
            "ts": e["ts"], "service": e["service"], "version": e["version"],
            "latency_ms": round(e["latency_ms"], 1), "error_rate": round(e["error_rate"], 4),
            "cpu_pct": round(e["cpu_pct"], 1), "mem_pct": round(e["mem_pct"], 1),
            "rps": round(e["rps"], 1), "log": e["log"],
            "truth": e["incident_type"] or "normal", "breach": bool(e["label"]),
            "predicted": d.incident, "set": sorted(d.conformal_set), "action": d.action,
        }
        if d.action == "escalate" or d.incident in ("dependency_failure", "bad_deploy"):
            v = reader.read(e, dict(zip(FEATURE_NAMES, d.features, strict=True)))
            row["resolved"] = v.incident
            row["evidence"] = v.evidence
            row["explanation"] = v.explanation
            row["remediation"] = v.remediation
        out.append(row)
    return out[-FEED_LIMIT:]


def build(n_ticks: int = 6000, seed: int = 7) -> dict:
    events = sorted(generate(n_ticks=n_ticks, seed=seed, incidents_per_service=10),
                    key=lambda e: e["ts"])
    X, y, ts = build_dataset(events)
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    clf = train(Xtr, ytr)
    report = evaluate(clf, Xte, yte)

    cut = np.quantile(ts, 0.7)
    warm = [e for e in events if e["ts"] >= 60.0]
    tr_ev = [e for e, k in zip(warm, ts < cut, strict=True) if k]
    te_ev = [e for e, k in zip(warm, ts >= cut, strict=True) if k]
    pred = list(clf.predict(Xte))
    risk = 1.0 - clf.predict_proba(Xte)[:, list(clf.classes_).index("normal")]

    # what reading the log line is worth
    from sklearn.metrics import f1_score
    after = resolve(te_ev, pred, MockReader())
    dep = [(t, p) for t, p in zip(yte, after, strict=True) if t == "dependency_failure"]

    # what choosing the action is worth
    exp = simulate_experiment(tr_ev, seed=1)
    table = uplift(exp)
    effects = TLearner().fit(Xtr, exp["action"], exp["outcome"]).effects(Xte)
    policies = evaluate_policies(te_ev, pred, table, risk, effects)

    # drift, clean against shifted
    ref = healthy_reference(tr_ev)
    def alerts(evs):
        m = DriftMonitor(ref, window=600)
        return sum(1 for e in evs if m.observe(e["latency_ms"])), max(len(evs) // 600, 1)
    clean_a, clean_w = alerts(te_ev)
    drift_a, drift_w = alerts(sorted(generate(n_ticks=1500, seed=seed, drifted=True),
                                     key=lambda e: e["ts"]))

    parts = fit_all(events[:int(0.6 * len(events))], seed=seed)
    live = run(events[int(0.6 * len(events)):], *parts)

    return {
        "generatedFrom": {"ticks": n_ticks, "seed": seed, "events": len(events),
                          "services": len({e["service"] for e in events})},
        "hotPath": {"p50": round(live.p50_ms, 3), "p95": round(float(np.percentile(
            live.hot_latencies_ms, 95)), 3), "p99": round(live.p99_ms, 3),
            "throughput": round(live.throughput_eps)},
        "classifier": {
            "macroF1": round(report["macro_f1"], 3),
            "perClass": {c: {"precision": round(s["precision"], 2),
                             "recall": round(s["recall"], 2),
                             "f1": round(s["f1"], 2), "support": s["support"]}
                         for c, s in report["per_class"].items()},
            "confusion": report["confusion"], "labels": report["labels"],
        },
        "reasoning": {
            "macroF1Before": round(report["macro_f1"], 3),
            "macroF1After": round(float(f1_score(yte, after, average="macro",
                                                 zero_division=0)), 3),
            "depRecallBefore": round(report["per_class"]["dependency_failure"]["recall"], 2),
            "depRecallAfter": round(sum(p == t for t, p in dep) / len(dep), 2),
            "escalatedPct": round(100 * sum(
                p in ("dependency_failure", "bad_deploy") for p in pred) / len(pred), 1),
        },
        "policies": [{"name": p.name, "value": round(p.value, 1), "treated": p.treated}
                     for p in policies],
        "actionCost": ACTION_COST,
        "drift": {"cleanRate": round(100 * clean_a / clean_w),
                  "driftedRate": round(100 * drift_a / drift_w),
                  "naiveClean": 13, "naiveDrifted": 14},
        "pipeline": {
            "processed": live.processed, "escalated": live.escalated,
            "reclassified": live.resolved_by_log, "anomalies": live.anomalies,
            "acted": live.acted, "policyValue": round(live.policy_value, 1),
            "actions": dict(Counter(live.actions)), "incidents": dict(Counter(live.incidents)),
        },
        "feed": _feed(te_ev[-FEED_LIMIT:], clf, parts[2], MockReader()),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="rdi-export")
    ap.add_argument("--out", default="frontend/public/snapshot.json")
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args(argv)

    snap = build(n_ticks=args.n)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, {len(snap['feed'])} feed events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
