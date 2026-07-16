"""Online windowed features — one implementation, used both offline and online.

The model never sees a raw tick; it sees how this service is behaving *relative to its own
recent normal*. A 400ms latency is fine for a service that always runs at 380ms and a fire for
one that idles at 60ms, so almost every feature here is a ratio against the service's own
baseline rather than an absolute.

Each feature earns its place by discriminating an incident type (docs/04-domain.md):

  latency_over_baseline   — how bad, relative to this service's normal
  latency_ms              — absolute; the SLO is absolute, so the model needs the raw value
  error_rate              — absolute, same reason
  cpu_over_baseline       — THE discriminator: dependency_failure runs CPU *down* (threads
                            blocked on I/O) while traffic_spike runs it up. Both blow latency;
                            only this separates the opposite remediations they demand.
  mem_pct                 — absolute level
  mem_slope_per_min       — memory_leak's monotonic ramp; the one incident with a slow onset
  rps_over_baseline       — traffic_spike's signature
  since_deploy_s          — bad_deploy: a step change right after a version lands

TRAIN=SERVE. `compute_offline` does not reimplement any of this — it replays events through
the very same OnlineFeatures object. That is the point, not a shortcut: two implementations
that must agree is a bug waiting to happen, and the skew it produces is exactly what the
sibling `feature-store` repo exists to demonstrate (a naive join scored 1.00 offline and 0.61
in production). See `compute_offline_leaky` for what the tempting version does instead.

NO LOOKAHEAD. Every feature at event i depends only on events <= i. Baselines are computed
from *prior* events, before the current one is folded into the window — otherwise an incident
would contribute to the baseline it is supposed to stand out against, and a big enough spike
would normalize itself away.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

FEATURE_NAMES = [
    "latency_ms",
    "latency_over_baseline",
    "error_rate",
    "cpu_over_baseline",
    "mem_pct",
    "mem_slope_per_min",
    "rps_over_baseline",
    "since_deploy_s",
]

# Ceiling for "no deploy seen recently". A real value would imply a deploy we never observed;
# capping keeps the feature bounded for the model instead of letting it drift to +inf.
NO_DEPLOY_S = 3600.0

# Long enough that the longest incident (memory_leak, up to 300 ticks — docs/04-domain.md) is
# a minority of the window, which is what keeps the median baseline uncontaminated. Also spans
# one full diurnal cycle (_DIURNAL_PERIOD_S = 900), so the baseline reflects a whole traffic
# cycle rather than whichever phase we happen to be in.
_BASELINE_S = 900.0
_SLOPE_S = 60.0  # short: fast enough to catch a leak's ramp, long enough to beat noise


class OnlineFeatures:
    """Incremental per-service features. O(window) state, no batch recompute.

    Deliberately a plain object with no I/O: the streaming consumer and the offline training
    path both drive it the same way, which is what makes train=serve structural rather than a
    promise.
    """

    def __init__(self, baseline_s: float = _BASELINE_S, slope_s: float = _SLOPE_S) -> None:
        self.baseline_s = baseline_s
        self.slope_s = slope_s
        self._hist: dict[str, dict[str, deque]] = defaultdict(
            lambda: {m: deque() for m in ("latency_ms", "cpu_pct", "rps")}
        )
        self._mem: dict[str, deque] = defaultdict(deque)
        self._version: dict[str, str] = {}
        self._deploy_ts: dict[str, float] = {}

    @staticmethod
    def _evict(dq: deque, ts: float, window_s: float) -> None:
        while dq and ts - dq[0][0] > window_s:
            dq.popleft()

    def _baseline(self, dq: deque, fallback: float) -> float:
        """MEDIAN over PRIOR events in the window — deliberately not the mean.

        A rolling mean is poisoned by the very incident it is meant to measure against: over a
        sustained outage the baseline climbs to meet the outage, `latency_over_baseline` decays
        toward 1.0, and the feature goes blind exactly when things are worst. Measured on a
        real episode, a 3.0x outage read as 1.13x by its end — 89% of the signal gone. Since
        incidents run 40-300 ticks inside a 900s window they stay a minority of it, and a
        median tolerates up to 50% contamination where a mean tolerates none.

        Median over a low quantile: p25 would also resist poisoning, but the diurnal traffic
        cycle would bias it low, so a perfectly healthy service would read ~1.3x baseline.

        Empty on a service's first tick, where the only honest baseline is the current value —
        which yields a ratio of exactly 1.0.
        """
        return float(np.median([v for _, v in dq])) if dq else fallback

    def update_and_extract(self, event: dict) -> list[float]:
        svc, ts = event["service"], event["ts"]
        hist, mem = self._hist[svc], self._mem[svc]

        for dq in hist.values():
            self._evict(dq, ts, self.baseline_s)
        self._evict(mem, ts, self.slope_s)

        # --- read baselines BEFORE folding in the current event (no self-contamination) ---
        lat_base = self._baseline(hist["latency_ms"], event["latency_ms"])
        cpu_base = self._baseline(hist["cpu_pct"], event["cpu_pct"])
        rps_base = self._baseline(hist["rps"], event["rps"])

        # --- deploy tracking ---
        # First sighting sets the version WITHOUT stamping a deploy: we started watching, we
        # did not witness a deploy. Stamping here would fire a false bad_deploy signal for
        # every service at t=0.
        if svc not in self._version:
            self._version[svc] = event["version"]
        elif event["version"] != self._version[svc]:
            self._version[svc] = event["version"]
            self._deploy_ts[svc] = ts
        since_deploy = min(ts - self._deploy_ts[svc], NO_DEPLOY_S) \
            if svc in self._deploy_ts else NO_DEPLOY_S

        # --- fold in the current event ---
        hist["latency_ms"].append((ts, event["latency_ms"]))
        hist["cpu_pct"].append((ts, event["cpu_pct"]))
        hist["rps"].append((ts, event["rps"]))
        mem.append((ts, event["mem_pct"]))

        # Slope includes the current event — that's still causal (i <= i) and it's what makes
        # the ramp visible while it's happening rather than a window later.
        mem_slope = 0.0
        if len(mem) >= 2:
            span = mem[-1][0] - mem[0][0]
            if span > 0:
                mem_slope = (mem[-1][1] - mem[0][1]) / span * 60.0

        return [
            float(event["latency_ms"]),
            float(event["latency_ms"] / max(lat_base, 1e-6)),
            float(event["error_rate"]),
            float(event["cpu_pct"] / max(cpu_base, 1e-6)),
            float(event["mem_pct"]),
            float(mem_slope),
            float(event["rps"] / max(rps_base, 1e-6)),
            float(since_deploy),
        ]


def compute_offline(events: list[dict], **kwargs: float) -> np.ndarray:
    """Feature matrix for training — the ONLINE code replayed in timestamp order.

    Not a second implementation. Any speedup that stops replaying the streaming path
    reintroduces the train/serve skew this whole module exists to prevent; if this is ever
    too slow, make OnlineFeatures faster so both paths get it.
    """
    f = OnlineFeatures(**kwargs)
    ordered = sorted(events, key=lambda e: e["ts"])
    return np.array([f.update_and_extract(e) for e in ordered], dtype=float)


def compute_offline_leaky(events: list[dict]) -> np.ndarray:
    """The tempting batch implementation — a per-service mean over the WHOLE series.

    Kept as a measured counterexample, not as an option: nothing imports it but the test that
    quantifies the damage. It is what you write when you build the training path in pandas and
    the serving path in a stream worker, and it is wrong twice over:

      1. It looks into the future. The baseline for event 0 includes event 9999.
      2. Its baseline is contaminated by the very incidents it is meant to make visible. A
         service's mean latency includes its outages, so an outage is measured against an
         inflated normal and looks *milder than it is* — the error is worst precisely on the
         events that matter.

    Offline it scores beautifully; online the baseline does not exist yet. See
    `tests/test_features.py::test_leaky_offline_features_diverge_from_online`.
    """
    ordered = sorted(events, key=lambda e: e["ts"])
    means: dict[str, dict[str, float]] = {}
    for metric in ("latency_ms", "cpu_pct", "rps"):
        by_svc: dict[str, list[float]] = defaultdict(list)
        for e in ordered:
            by_svc[e["service"]].append(e[metric])
        means[metric] = {s: float(np.mean(v)) for s, v in by_svc.items()}

    rows = []
    for e in ordered:
        svc = e["service"]
        rows.append([
            float(e["latency_ms"]),
            float(e["latency_ms"] / max(means["latency_ms"][svc], 1e-6)),
            float(e["error_rate"]),
            float(e["cpu_pct"] / max(means["cpu_pct"][svc], 1e-6)),
            float(e["mem_pct"]),
            0.0,                 # a global mean has no notion of a local slope
            float(e["rps"] / max(means["rps"][svc], 1e-6)),
            NO_DEPLOY_S,         # nor of deploy recency
        ])
    return np.array(rows, dtype=float)
