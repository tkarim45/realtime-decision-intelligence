"""`rdi-stream`, emit the labeled synthetic AIOps stream as JSONL on stdout.

Downstream consumers pipe this into the broker. It also exists so
the stream is inspectable by eye and by `jq` before anything consumes it.

    rdi-stream --n 600 --summary        # ground-truth breakdown, no JSONL
    rdi-stream --n 600 | jq -c 'select(.label == 1)'
    rdi-stream --drifted --summary      # should show zero incidents, inflated latency
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from rdi.events import SLO_ERROR_RATE, SLO_LATENCY_MS, generate


def _summary(events: list[dict]) -> str:
    n = len(events)
    by_type = Counter(e["incident_type"] or "normal" for e in events)
    breaching = sum(e["label"] for e in events)
    lat = sorted(e["latency_ms"] for e in events)
    p50, p99 = lat[int(0.50 * n)], lat[int(0.99 * n)]

    lines = [
        f"events                {n}",
        f"SLO                   latency_ms > {SLO_LATENCY_MS:.0f}  or  "
        f"error_rate > {SLO_ERROR_RATE}",
        f"breaching (label=1)   {breaching}  ({breaching / n:.1%})",
        f"latency p50 / p99     {p50:.0f}ms / {p99:.0f}ms",
        "",
        "incident_type          count      of which breaching (label=1)",
    ]
    for kind, count in by_type.most_common():
        hit = sum(e["label"] for e in events if (e["incident_type"] or "normal") == kind)
        lines.append(f"  {kind:<20} {count:>6}      {hit:>6}  ({hit / count:.0%})")
    lines += [
        "",
        "anomaly != incident: a type with count > 0 but few breaching is one the detector",
        "should see and the decision layer should mostly leave alone.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rdi-stream", description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000, help="ticks (events = ticks x 6 services)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drifted", action="store_true",
                    help="fleet-wide distribution shift, no incidents injected (M5 test)")
    ap.add_argument("--incidents-per-service", type=int, default=3)
    ap.add_argument("--summary", action="store_true",
                    help="print the ground-truth breakdown instead of JSONL")
    args = ap.parse_args(argv)

    events = generate(n_ticks=args.n, seed=args.seed, drifted=args.drifted,
                      incidents_per_service=args.incidents_per_service)

    if args.summary:
        print(_summary(events))
        return 0

    try:
        for e in events:
            sys.stdout.write(json.dumps(e, separators=(",", ":")) + "\n")
    except BrokenPipeError:
        # `rdi-stream | head` closes the pipe early, that's the documented usage, not a
        # failure. Retarget stdout to devnull so the interpreter's flush-on-exit can't
        # re-raise into a nonzero exit and a stack trace.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
