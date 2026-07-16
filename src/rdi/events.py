"""Synthetic AIOps telemetry stream with labeled injected incidents.

A fleet of six services emits one metric tick each per second: latency, error rate, CPU,
memory, request rate, and a log line. Normal behavior is each service humming around its own
baseline under a diurnal traffic curve. Into that we inject four incident types, each with a
distinct *multi-metric* fingerprint — no single metric identifies any of them, which is the
whole reason the scoring layer has work to do (see docs/04-domain.md).

Two ground-truth fields, and they are not the same thing:

  incident_type — WHAT IS HAPPENING (None when normal). The anomaly detector's target.
  label         — SHOULD WE ACT: 1 iff this tick breaches the SLO. The decision layer's target.

An absorbed traffic spike is incident_type="traffic_spike", label=0 — a real anomaly that
costs money to remediate and doesn't need it. A memory leak's first minutes are label=0 too:
real, not yet breaching. Collapsing the two fields would let a risk threshold masquerade as a
causal policy and would erase the uplift engine's job.

The label is DERIVED from the emitted metrics against the SLO, never painted on:

    label = 1  iff  latency_ms > SLO_LATENCY_MS  or  error_rate > SLO_ERROR_RATE

so the SLO constant *is* the ground-truth definition. A noisy normal tick can breach on its
own — that's a blip, and it's meant to be there.
"""
from __future__ import annotations

import numpy as np

SERVICES = [
    "checkout-api",
    "payments-worker",
    "search-api",
    "auth-api",
    "cart-api",
    "recommend-api",
]

INCIDENT_TYPES = ["memory_leak", "dependency_failure", "traffic_spike", "bad_deploy"]

# Ground truth for Milestone 4. Deliberately NOT a field on the event — the uplift model has
# to recover this from the metrics, and the policy has to weigh it against the cost of acting.
REMEDIATIONS = {
    "memory_leak": "restart",
    "dependency_failure": "failover",
    "traffic_spike": "scale_out",
    "bad_deploy": "rollback",
}

ACTIONS = ["none", "restart", "failover", "scale_out", "rollback"]

# The SLO *is* the ground-truth definition of "incident". Changing these re-labels the stream.
SLO_LATENCY_MS = 500.0
SLO_ERROR_RATE = 0.05

METRIC_NAMES = ["latency_ms", "error_rate", "cpu_pct", "mem_pct", "rps"]

# Upstream each service depends on — used to make dependency_failure logs concrete.
_UPSTREAM = {
    "checkout-api": "payments-db",
    "payments-worker": "ledger-db",
    "search-api": "search-index",
    "auth-api": "session-store",
    "cart-api": "cart-cache",
    "recommend-api": "feature-store",
}

_DIURNAL_PERIOD_S = 900.0  # compressed "day" — a full traffic cycle every 15 min of stream


def _baseline_profile(rng: np.random.Generator) -> dict:
    return {
        "latency_ms": float(rng.uniform(40, 120)),
        "error_rate": float(rng.uniform(0.001, 0.010)),
        "cpu_pct": float(rng.uniform(20, 45)),
        "mem_pct": float(rng.uniform(30, 55)),
        "rps": float(rng.uniform(50, 400)),
    }


def _schedule(rng: np.random.Generator, n_ticks: int, per_service: int) -> list[dict]:
    """Non-overlapping incident episodes for one service.

    Durations differ by type on purpose: a memory leak that resolved in 20 ticks would be
    indistinguishable from a spike, and the slow-vs-fast onset split is half the signal.
    """
    spans = {
        "memory_leak": (120, 300),
        "dependency_failure": (20, 60),
        "traffic_spike": (15, 50),
        "bad_deploy": (40, 120),
    }
    episodes: list[dict] = []
    for _ in range(per_service):
        kind = INCIDENT_TYPES[int(rng.integers(0, len(INCIDENT_TYPES)))]
        lo, hi = spans[kind]
        duration = int(rng.integers(lo, hi + 1))
        if duration >= n_ticks:
            continue
        start = int(rng.integers(0, n_ticks - duration))
        end = start + duration
        if any(start < e["end"] and e["start"] < end for e in episodes):
            continue  # overlapping episodes would blend fingerprints into mush
        ep = {"kind": kind, "start": start, "end": end, "duration": duration}
        if kind == "traffic_spike":
            # ~40% of spikes are ABSORBED: the service handles the load, latency stays under
            # SLO, label=0. Scaling out for these is a pure loss — that's the uplift signal.
            ep["absorbed"] = bool(rng.random() < 0.40)
            ep["magnitude"] = float(rng.uniform(3.0, 6.0))
        elif kind == "dependency_failure":
            ep["magnitude"] = float(rng.uniform(3.0, 8.0))
            ep["err"] = float(rng.uniform(0.30, 0.60))
        elif kind == "bad_deploy":
            ep["magnitude"] = float(rng.uniform(2.0, 4.0))
            ep["err"] = float(rng.uniform(0.08, 0.25))
        episodes.append(ep)
    return sorted(episodes, key=lambda e: e["start"])


def _apply(kind: str, ep: dict, p: float, m: dict, rng: np.random.Generator) -> dict:
    """Apply an incident fingerprint at progress p in [0, 1]. Mutates a copy of the metrics."""
    m = dict(m)
    if kind == "memory_leak":
        # Memory climbs monotonically; latency follows quadratically (GC pressure); errors
        # arrive LATE (cubic) — the service degrades long before it fails.
        m["mem_pct"] += 45.0 * p
        m["latency_ms"] *= 1.0 + 6.0 * p**2
        m["error_rate"] += 0.09 * p**3
        m["cpu_pct"] += 12.0 * p  # GC burns some CPU
    elif kind == "dependency_failure":
        # Threads blocked on I/O: latency and errors explode while CPU goes DOWN — the
        # service isn't working, it's waiting. That's what separates this from a spike.
        m["latency_ms"] *= ep["magnitude"]
        m["error_rate"] = ep["err"]
        m["cpu_pct"] *= 0.6
        m["rps"] *= 0.9
    elif kind == "traffic_spike":
        mag = ep["magnitude"]
        m["rps"] *= mag
        m["cpu_pct"] = min(97.0, m["cpu_pct"] * (1.0 + 0.35 * mag))
        # Absorbed: latency rises but stays comfortably under SLO. Breaking: blows through it.
        m["latency_ms"] *= 1.6 if ep["absorbed"] else (1.0 + 1.2 * mag)
        m["error_rate"] += 0.004 if ep["absorbed"] else 0.02
    elif kind == "bad_deploy":
        # Step change, not a ramp: latency and errors move together the instant the version
        # lands, and stay there until someone rolls back.
        m["latency_ms"] *= ep["magnitude"]
        m["error_rate"] += ep["err"]
    return m


def _log_line(service: str, kind: str | None, m: dict, version: str,
              rng: np.random.Generator) -> str:
    """A log line per tick — the LLM agent's raw material, flavored by what's actually wrong."""
    if kind == "memory_leak":
        return (f"GC pause {int(rng.uniform(180, 900))}ms; heap at {m['mem_pct']:.0f}% of limit; "
                f"old-gen collection did not reclaim")
    if kind == "dependency_failure":
        return (f"upstream {_UPSTREAM[service]} timeout after 5000ms; "
                f"connection pool exhausted ({int(rng.uniform(20, 60))} waiting)")
    if kind == "traffic_spike":
        return (f"request queue depth {int(m['rps'] / 4)}; worker pool saturated; "
                f"{int(rng.uniform(0, 40))} requests shed")
    if kind == "bad_deploy":
        return (f"NullPointerException in {service.split('-')[0].title()}Handler.validate "
                f"({version}); returning 500")
    path = ["/health", "/api/v1/query", "/api/v1/submit", "/metrics"][int(rng.integers(0, 4))]
    return f"GET {path} 200 in {m['latency_ms']:.0f}ms"


def generate(n_ticks: int = 3000, seed: int = 7, drifted: bool = False,
             incidents_per_service: int = 3) -> list[dict]:
    """Emit a labeled AIOps telemetry stream: one event per service per tick.

    drifted=True inflates fleet-wide baseline latency and shifts the request-rate
    distribution and injects NO incidents. The world changed; nothing broke. PSI must fire
    on this stream and the incident alert must stay silent — that's the M5 specificity test.
    """
    rng = np.random.default_rng(seed)
    profiles = {s: _baseline_profile(rng) for s in SERVICES}
    versions = {s: [1, int(rng.integers(0, 6)), int(rng.integers(0, 10))] for s in SERVICES}

    if drifted:
        for s in SERVICES:
            profiles[s]["latency_ms"] *= 1.8   # new hardware / noisier neighbors
            profiles[s]["rps"] *= 1.5          # traffic pattern moved
        schedules = {s: [] for s in SERVICES}
    else:
        schedules = {s: _schedule(rng, n_ticks, incidents_per_service) for s in SERVICES}

    events: list[dict] = []
    for tick in range(n_ticks):
        # One diurnal curve across the whole fleet — shared traffic seasonality is exactly the
        # confound that makes "high rps" a useless standalone incident signal.
        diurnal = 1.0 + 0.45 * np.sin(2 * np.pi * tick / _DIURNAL_PERIOD_S)
        for service in SERVICES:
            base = profiles[service]
            m = {
                "latency_ms": base["latency_ms"] * float(rng.normal(1.0, 0.12)),
                "error_rate": max(0.0, base["error_rate"] * float(rng.normal(1.0, 0.30))),
                "cpu_pct": base["cpu_pct"] * diurnal * float(rng.normal(1.0, 0.08)),
                "mem_pct": base["mem_pct"] * float(rng.normal(1.0, 0.04)),
                "rps": base["rps"] * diurnal * float(rng.normal(1.0, 0.10)),
            }

            active = next((e for e in schedules[service] if e["start"] <= tick < e["end"]), None)
            kind = active["kind"] if active else None
            if active is not None:
                if active["kind"] == "bad_deploy" and tick == active["start"]:
                    versions[service][2] += 1  # the deploy that broke it
                p = (tick - active["start"]) / max(active["duration"] - 1, 1)
                m = _apply(active["kind"], active, p, m, rng)

            version = "v{}.{}.{}".format(*versions[service])
            m["cpu_pct"] = float(np.clip(m["cpu_pct"], 0.0, 100.0))
            m["mem_pct"] = float(np.clip(m["mem_pct"], 0.0, 100.0))
            m["error_rate"] = float(np.clip(m["error_rate"], 0.0, 1.0))

            events.append({
                "ts": float(tick),
                "service": service,
                "version": version,
                "latency_ms": round(m["latency_ms"], 2),
                "error_rate": round(m["error_rate"], 5),
                "cpu_pct": round(m["cpu_pct"], 2),
                "mem_pct": round(m["mem_pct"], 2),
                "rps": round(m["rps"], 2),
                "log": _log_line(service, kind, m, version, rng),
                "incident_type": kind,
                # Derived, not painted on — the SLO is the ground-truth definition.
                "label": int(m["latency_ms"] > SLO_LATENCY_MS or m["error_rate"] > SLO_ERROR_RATE),
            })
    return events
