"""`rdi-recover` and `rdi-parity` — M1's two artifacts, runnable rather than only asserted.

The tests prove these properties; these commands *show* them, which is what a reader who
doesn't run pytest actually looks at.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from rdi.broker import Broker
from rdi.consumer import Consumer
from rdi.events import generate
from rdi.features import FEATURE_NAMES, compute_offline, compute_offline_leaky


def recover(n_ticks: int = 200, die_after: int = 400) -> int:
    """Kill a consumer mid-stream, restart, show zero events lost (S1)."""
    events = generate(n_ticks=n_ticks, seed=13)
    with tempfile.TemporaryDirectory() as tmp, \
            Broker(str(Path(tmp) / "stream.jsonl"), fsync=True) as b:
        for e in events:
            b.append(e)
        total = b.lag("g")["appended"]
        print(f"appended              {total} events (fsync on)")

        crash = Consumer("crash", b, "g", die_after=die_after)
        crash.run()
        print(f"consumer 'crash'      processed {crash.results.processed}, then died")
        print(f"in flight at crash    {b.lag('g')['pending']} events claimed but never acked")

        time.sleep(0.25)
        rescue = Consumer("rescue", b, "g")
        rescue.run()
        print(f"consumer 'rescue'     processed {rescue.results.processed} "
              f"({rescue.results.duplicates} duplicates deduped)")

        done = {o for c in (crash, rescue) for o, _ in c.results.features}
        print()
        print(f"total processed       {crash.results.processed + rescue.results.processed}"
              f" / {total}")
        print(f"still pending         {b.lag('g')['pending']}")
        print(f"distinct offsets      {len(done)} / {total}")
        lost = total - len(done)
        print()
        print(f"EVENTS LOST: {lost}" + ("  ✅ at-least-once holds" if lost == 0 else "  ❌"))
        return 0 if lost == 0 else 1


def parity(n_ticks: int = 600) -> int:  # 600: short enough to be quick, long enough that all
                                        # four incident types get scheduled (bad_deploy runs
                                        # 40-120 ticks and doesn't fit a shorter window)
    """Show offline == online features, and what the tempting batch version costs."""
    events = generate(n_ticks=n_ticks, seed=7)
    offline = compute_offline(events)

    with tempfile.TemporaryDirectory() as tmp, \
            Broker(str(Path(tmp) / "stream.jsonl")) as b:
        for e in events:
            b.append(e)
        streamed = Consumer("c", b, "g").run(batch=32).features
    online = np.array([f for _, f in sorted(streamed, key=lambda x: x[0])], dtype=float)

    max_diff = float(np.abs(online - offline).max())
    print(f"events                {len(events)}")
    print(f"features              {len(FEATURE_NAMES)}  ({', '.join(FEATURE_NAMES)})")
    print()
    print("TRAIN=SERVE  offline (training) vs streamed (serving)")
    print(f"  max abs difference  {max_diff:.1e}   "
          f"{'✅ identical' if max_diff == 0 else '❌ SKEW'}")
    print("  (they agree because they are the same code, replayed — not two implementations)")

    leaky = compute_offline_leaky(events)
    col = FEATURE_NAMES.index("latency_over_baseline")
    print()
    print("THE LEAK  a per-service mean over the whole series (the pandas-training version)")
    print(f"  max abs difference  {float(np.abs(leaky[:, col] - offline[:, col]).max()):.2f}"
          "   vs the online path — inputs the model will never see in production")
    print()
    print(f"  {'incident':<22}{'online':>10}{'leaky':>10}   latency_over_baseline")
    ordered = sorted(events, key=lambda e: e["ts"])
    for kind in ("dependency_failure", "bad_deploy", "traffic_spike", "memory_leak"):
        idx = [i for i, e in enumerate(ordered) if e["incident_type"] == kind]
        if not idx:
            continue
        print(f"  {kind:<22}{offline[idx, col].mean():>10.2f}{leaky[idx, col].mean():>10.2f}")
    print()
    print("  The leaky baseline is contaminated by the incidents it should expose, so outages")
    print("  read MILDER than they are — the feature is least trustworthy exactly where it")
    print("  matters most.")
    return 0


def score(n_ticks: int = 6000) -> int:
    """Train the hot-path classifier; show the honest score and what a random split pretends."""
    from rdi.model import (
        WARMUP_TICKS,
        build_dataset,
        evaluate,
        importances,
        random_split,
        temporal_split,
        train,
    )

    events = generate(n_ticks=n_ticks, seed=7, incidents_per_service=10)
    X, y, ts = build_dataset(events)
    print(f"events                {len(X)} (after a {WARMUP_TICKS:.0f}-tick warmup)")
    print("target                incident_type — NOT `label`, which is the SLO if-statement")
    print()

    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    model = train(Xtr, ytr)
    t = evaluate(model, Xte, yte)

    print(f"TEMPORAL SPLIT (train on the past, test on the future)   "
          f"macro-F1 {t['macro_f1']:.3f}")
    print(f"  {'class':<20}{'P':>6}{'R':>6}{'F1':>7}{'n':>7}")
    for c, s in t["per_class"].items():
        print(f"  {c:<20}{s['precision']:>6.2f}{s['recall']:>6.2f}"
              f"{s['f1']:>7.2f}{s['support']:>7}")

    print()
    print("  confusion (rows = truth)")
    print(f"  {'':<20}" + "".join(f"{c[:9]:>11}" for c in t["labels"]))
    for c, row in zip(t["labels"], t["confusion"], strict=True):
        print(f"  {c:<20}" + "".join(f"{v:>11}" for v in row))

    rXtr, rXte, rytr, ryte = random_split(X, y, ts)
    r = evaluate(train(rXtr, rytr), rXte, ryte)
    print()
    print(f"RANDOM SPLIT (leaky)                                     "
          f"macro-F1 {r['macro_f1']:.3f}")
    print(f"  inflation             {r['macro_f1'] - t['macro_f1']:+.3f} macro-F1 of illusion")
    print("  An episode spans 40-300 near-identical ticks. Shuffling puts tick 41 in train and")
    print("  tick 42 in test, so the model is scored on ticks whose neighbours it memorized.")

    print()
    print("  feature importance   " + "  ".join(f"{n}={v}" for n, v in importances(model)[:4]))
    print()
    print("  The hard class is dependency_failure, and it is mistaken for bad_deploy: a retry")
    print("  storm drives CPU, latency and errors up together — exactly a bad deploy's shape —")
    print("  and benign deploys ship often enough that one recently landed. Opposite fixes")
    print("  (failover vs rollback). The metrics cannot settle it; the log line names the")
    print("  upstream, which is M3's job.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rdi-demo")
    ap.add_argument("demo", choices=["recover", "parity", "score"])
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)
    fn = {"recover": recover, "parity": parity, "score": score}[args.demo]
    return fn(**({"n_ticks": args.n} if args.n else {}))


if __name__ == "__main__":
    raise SystemExit(main())
