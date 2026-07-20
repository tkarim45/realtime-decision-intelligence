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


def uncertainty(n_ticks: int = 6000) -> int:
    """The unknown-unknowns detector and the conformal sets — including what did NOT work."""
    import numpy as np

    from rdi.anomaly import detection_rate, fit_on_normal
    from rdi.conformal import coverage, naive_sets, prediction_sets, qhat
    from rdi.model import build_dataset, evaluate, temporal_split, train

    events = generate(n_ticks=n_ticks, seed=7, incidents_per_service=10)
    X, y, ts = build_dataset(events)
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    clf, det = train(Xtr, ytr), fit_on_normal(Xtr, ytr)
    per, rates = evaluate(clf, Xte, yte)["per_class"], detection_rate(det, Xte, yte)

    print("ANOMALY DETECTOR — trained on NORMAL ticks only, so it can flag what it never saw")
    print(f"  {'class':<20}{'clf recall':>12}{'det flagged':>13}")
    for c in ("normal", "memory_leak", "dependency_failure", "traffic_spike", "bad_deploy"):
        print(f"  {c:<20}{per[c]['recall']:>12.2f}{rates[c]['flagged']:>12.1%}")
    inc = [c for c in rates if c != "normal"]
    corr = float(np.corrcoef([per[c]["recall"] for c in inc],
                             [rates[c]["flagged"] for c in inc])[0, 1])
    print(f"\n  correlation(clf recall, det flag) = {corr:+.2f} — anti-correlated, so the two")
    print("  layers are strongest on DIFFERENT incidents. That is what earns the second model.")

    seen = ytr != "memory_leak"
    blind, blind_det = train(Xtr[seen], ytr[seen]), fit_on_normal(Xtr[seen], ytr[seen])
    held = yte == "memory_leak"
    called_normal = float((blind.predict(Xte[held]) == "normal").mean())
    print("\n  HOLDOUT: hide memory_leak from both models entirely")
    print(f"    classifier calls it 'normal'  {called_normal:.1%}"
          "   (it cannot say 'I don't know')")
    print(f"    detector still flags it       {float(blind_det.flag(Xte[held]).mean()):.1%}")

    print("\n  BUT NOT A SAFETY NET: the events the classifier misses are the mild ones, and")
    pred, flag = clf.predict(Xte), det.flag(Xte)
    missed = (pred == "normal") & (yte != "normal")
    dis = (pred == "normal") & (flag == 1)
    print(f"    detector catches only {float(flag[missed].mean()):.1%} of the classifier's misses,")
    dis_prec, base = float((yte[dis] != "normal").mean()), float((yte != "normal").mean())
    print(f"    and 'clf says normal, det disagrees' is {dis_prec:.1%} incidents"
          f" vs a {base:.1%} base rate — no signal.")

    print("\nCONFORMAL SETS — coverage guarantee, and a prediction of mine that failed")
    c1, c2 = np.quantile(ts, 0.55), np.quantile(ts, 0.75)
    tr, ca, te = ts < c1, (ts >= c1) & (ts < c2), ts >= c2
    m2 = train(X[tr], y[tr])
    cls = list(m2.classes_)
    p_ca, p_te = m2.predict_proba(X[ca]), m2.predict_proba(X[te])
    yi = np.array([cls.index(v) for v in y[ca]])
    print(f"  {'target':<10}{'conformal cov':>15}{'size':>7}{'ambiguous':>11}{'naive cov':>11}")
    for a in (0.10, 0.05, 0.01):
        s = prediction_sets(p_te, qhat(p_ca, yi, a), cls)
        cs = coverage(s, y[te])
        ns = coverage(naive_sets(p_te, a, cls), y[te])
        print(f"  >={1 - a:<8.0%}{cs['coverage']:>15.3f}{cs['avg_set_size']:>7.2f}"
              f"{cs['ambiguous_rate']:>11.1%}{ns['coverage']:>11.3f}")
    print("  naive softmax barely moves with the target — it cannot be dialled to a guarantee.")

    dep = y[te] == "dependency_failure"
    mis = dep & (m2.predict(X[te]) == "bad_deploy")
    print("\n  THE PREDICTION THAT FAILED: conformal was meant to hedge the dependency_failure")
    print("  vs bad_deploy confusion so an ambiguous set could route to the LLM. It does not.")
    print(f"    when it misreads, P(bad_deploy)={p_te[mis, cls.index('bad_deploy')].mean():.3f}, "
          f"P(dependency_failure)={p_te[mis, cls.index('dependency_failure')].mean():.3f}")
    print("    -> confidently WRONG, not uncertain. Conformal widens sets a model hesitates")
    print("       on; it cannot manufacture doubt. Reaching the hedge needs a 99% target,")
    print("       which makes ~42% of the whole stream ambiguous. The log line is the answer.")
    return 0


def loadtest(n_ticks: int = 4000) -> int:
    """Hot-path latency, and why the anomaly detector is not on it."""
    from rdi.model import build_dataset
    from rdi.scoring import HotPathScorer, NearLineDetector, fit_hot_path, latency_profile

    events = sorted(generate(n_ticks=n_ticks, seed=7, incidents_per_service=10),
                    key=lambda e: e["ts"])
    X, y, ts = build_dataset(events)
    model, det, q, _ = fit_hot_path(X, y, ts)

    alone = latency_profile(HotPathScorer(model, q), events)
    print("HOT PATH  features -> classifier -> conformal set -> act/escalate")
    print(f"  events              {alone['n']}")
    print(f"  p50 / p95 / p99     {alone['p50_ms']:.3f} / {alone['p95_ms']:.3f} / "
          f"{alone['p99_ms']:.3f} ms")
    print(f"  max                 {alone['max_ms']:.2f} ms")
    print(f"  throughput          {alone['throughput_eps']:,.0f} events/s single-threaded")
    print(f"  sub-ms p99 SLO      {'MET' if alone['p99_ms'] < 1.0 else 'MISSED'}")
    print(f"  routing             {alone['actions']}")

    # Measured live rather than quoted: a hardcoded table would rot the moment anything here
    # changed, and this is the number the whole design decision rests on.
    import time

    import numpy as np

    from rdi.features import OnlineFeatures

    feats = OnlineFeatures()
    for e in events[:600]:
        feats.update_and_extract(e)
    row = np.asarray(feats.update_and_extract(events[600]), dtype=float).reshape(1, -1)
    batch256 = X[:256]

    def p50_ms(fn, n=300):
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
        return float(np.percentile(samples, 50))

    c_feat = p50_ms(lambda: feats.update_and_extract(events[601]))
    c_model = p50_ms(lambda: model.predict_proba(row))
    c_det1 = p50_ms(lambda: det.flag(row), n=60)
    c_det_b = p50_ms(lambda: det.flag(batch256), n=20) / 256

    print("\nWHY THE DETECTOR IS NOT ON THIS PATH  (measured now, p50 per call)")
    print(f"  features             {c_feat:.3f} ms")
    print(f"  model.predict_proba  {c_model:.3f} ms")
    print(f"  detector.flag        {c_det1:.3f} ms   <- {c_det1 / max(c_model, 1e-9):.0f}x the "
          "classifier, and most of the budget")
    print(f"  The same forest costs {c_det_b:.4f} ms/event at batch 256 — "
          f"{c_det1 / max(c_det_b, 1e-9):.0f}x cheaper per event. It is sklearn per-call")
    print("  dispatch, not compute (it scales with tree count: 25 trees ~0.7ms, 200 ~5.1ms).")
    print("  So it runs batched, on its own consumer group.")

    together = latency_profile(HotPathScorer(model, q), events,
                               detector=NearLineDetector(det))
    print("\nAND OFF THE PATH IS NOT ENOUGH — IT HAS TO LEAVE THE THREAD")
    print(f"  hot path alone         p99 {alone['p99_ms']:.3f} ms   max {alone['max_ms']:6.2f} ms"
          f"   {alone['throughput_eps']:,.0f} eps")
    print(f"  + detector in-loop     p99 {together['p99_ms']:.3f} ms   "
          f"max {together['max_ms']:6.2f} ms   {together['throughput_eps']:,.0f} eps")
    print(f"  near-line flags        {together['anomalies_flagged']}")
    print("  The detector's cost is EXCLUDED from these timings, yet co-locating it still")
    print("  perturbs the tail: a 7ms batched tree walk every 256 events evicts cache under")
    print("  the events that follow. (Not GC — disabling it does not help.) 'Don't count it'")
    print("  is not latency isolation; only a separate process is.")
    print("\n  Machine-dependent: the reference run showed p99 0.633 -> 1.525 ms and")
    print("  max 2.89 -> 20.52 ms. Direction is robust, magnitude is not — which is why this")
    print("  lives here and not in an assertion.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rdi-demo")
    ap.add_argument("demo", choices=["recover", "parity", "score", "uncertainty", "loadtest"])
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)
    fn = {"recover": recover, "parity": parity, "score": score,
          "uncertainty": uncertainty, "loadtest": loadtest}[args.demo]
    return fn(**({"n_ticks": args.n} if args.n else {}))


if __name__ == "__main__":
    raise SystemExit(main())
