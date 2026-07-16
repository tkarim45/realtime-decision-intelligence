# Real-Time Multi-Modal Decision Intelligence System

> A production system that ingests a live multi-modal stream (events + text, optionally
> image/audio), scores it in real time with classical + small deep models, **explains and
> acts** on it with an LLM agent, and decides **interventions** with causal/uplift
> modeling — all under drift monitoring. Runs on an Apple M1 (8 GB): small on-device
> models for scoring, local + cloud LLM routed for reasoning.

**Status:** 🚧 **M0–M1 of 6 complete.** Domain committed (**AIOps**); labeled synthetic
telemetry stream; durable at-least-once broker with **crash recovery proven (1200/1200, 0
lost)**; train=serve online features at **0.0 skew**, hot path **p99 0.30ms**. 75 tests, ruff
clean. Scoring (M2) onward pending. Build from
[`docs/02-build-plan.md`](docs/02-build-plan.md).

**Capstone #3 of 3.** Siblings: `self-improving-agent-platform`,
`ondevice-model-lifecycle`. Effort: **6–12 months solo**.

---

## The one-sentence pitch

Don't just predict — **decide and act, in real time, with evidence.** A live stream flows
in; a fast classical model scores the hot path (sub-ms); a small deep model adds signal;
an anomaly detector catches unknown-unknowns; conformal prediction gives each decision a
guaranteed-coverage confidence set; an LLM agent explains flagged events and can remediate
via tools; and a **causal uplift engine decides which intervention actually helps** —
validated by a peeking-safe experiment engine, all under drift monitoring with
automated retrain + rollback.

## Why this project exists

It's the maximal expression of the mid-2026 "full-stack AI in production" expectation
(see `../research.md`: 72% of AI-first roles pair GenAI with ops). It wires GenAI into a
real streaming decision loop with **measured outcomes** — the clearest "AI in the real
world with evidence of usage" story. It composes five existing repos
(`realtime-ml-pipeline`, `feature-store`, `timeseries-anomaly-detection`,
`uplift-targeting-engine`, `experimentation-engine`) into one deployed product.

## Domain: AIOps / service reliability (committed at M0)

Six microservices emit a metric tick each per second — latency, error rate, CPU, memory,
request rate, and a **log line**. Four incident types are injected, each with a distinct
*multi-metric* fingerprint; the intervention is the remediation (`restart` / `failover` /
`scale_out` / `rollback` / `none`).

Chosen over fraud (which would re-skin `realtime-ml-pipeline`) and vitals because the LLM has
real work — explaining an incident means reading logs, not narrating a number — and because
the action space is genuinely causal: the *wrong* remediation has near-zero or negative
effect (restarting a service whose upstream database is down costs downtime and fixes
nothing). A binary treat/don't-treat can be faked with a risk threshold; this cannot.

**The schema's one real idea — an anomaly is not an incident:**

| field | means | consumer |
|---|---|---|
| `incident_type` | **what is happening** (`None` = normal) | anomaly detector |
| `label` | **should we act** — 1 iff this tick breaches the SLO | decision layer |

An absorbed traffic spike is a real anomaly (`incident_type="traffic_spike"`) that must
*not* be remediated (`label=0`) — scaling out for it is a pure loss. A memory leak's first
minutes are label=0 too: real, not yet breaching. Collapsing the two fields would let a risk
threshold masquerade as a causal policy and erase the uplift engine's job. The label is
**derived** (`latency_ms > 500 or error_rate > 0.05`), so the SLO *is* the ground truth.

Full schema, fingerprints, and action space: [`docs/04-domain.md`](docs/04-domain.md).

## Headline deliverable (what to demo)

A live dashboard showing the system on a running stream with **all four proofs at once**:
1. **Stated throughput + p99 latency SLOs** met.
2. **Proven zero-loss crash recovery** (kill mid-stream, resume, 0 events lost).
3. A **drift-triggered retrain** firing on a deliberately-drifted stream (not on a clean one).
4. A **validated causal lift** from the system's interventions (Qini / policy value).

## What runs where (Apple M1, 8 GB)

| Role | Where |
|---|---|
| Durable event log, stream workers | File-backed Redis-Streams-contract broker (reuse `realtime-ml-pipeline`) — no Kafka cluster |
| Hot-path scoring (sub-ms) | Classical model (scikit-learn / LightGBM) |
| Richer signal | **Small MLX sequence encoder (1–20M params), trained on-device** |
| Unknown-unknowns | Anomaly detector (IsolationForest / from-scratch) |
| Per-decision uncertainty | Conformal prediction (from-scratch) |
| Explain / remediate | **Local Qwen2.5-1.5B (cheap/offline) + Claude API (hard cases), routed** |
| Intervention decision | Uplift / meta-learners |
| Did it work? | Peeking-safe experimentation engine |
| Cluster scale-out | Optional documented cloud step |

The deep model is deliberately small so it fits 8 GB **and** keeps hot-path latency low.

## Architecture (one glance)

```
 live stream ─▶ durable log (consumer groups, at-least-once) ─▶ online features (train=serve)
                                        │
                 ┌──────────────────────┼───────────────────────┐
                 ▼                      ▼                        ▼
        classical (sub-ms)     small MLX encoder          anomaly detector
                 └──────────── ensemble + calibration ────────────┘
                                        │  + conformal (coverage-guaranteed set)
                                        ▼
                              flag? ─▶ LLM agent (local/Claude routed)
                                        │  pull context (feature store, logs, RAG)
                                        │  grounded explanation + recommended action
                                        ▼
                              causal uplift ─▶ treat / don't-treat policy
                                        ▼
                       experimentation engine (peeking-safe) ─▶ measured lift
                                        ▼
        drift monitor (PSI) ─▶ retrain trigger ─▶ canary + shadow ─▶ rollback
                                        ▼
                              real-time ops dashboard
```

Detail: [`docs/01-architecture.md`](docs/01-architecture.md).

## Repository layout

Subsystems are subpackages of one installable `rdi` package rather than seven top-level
directories — `import broker` is not a name any project should own. They land as milestones
complete; empty placeholder packages aren't committed.

```
realtime-decision-intelligence/
├── README.md · Makefile · pyproject.toml · .gitignore
├── docs/  00-what-it-is · 01-architecture · 02-build-plan · 03-setup · 04-domain
├── src/rdi/
│   ├── events.py       # ✅ M0 — labeled synthetic AIOps stream
│   ├── cli.py          # ✅ M0 — `rdi-stream`, JSONL on stdout
│   ├── broker.py       # ✅ M1 — durable event log (consumer groups, at-least-once)
│   ├── consumer.py     # ✅ M1 — claim → features → ack
│   ├── features.py     # ✅ M1 — online windowed features, train=serve
│   ├── demo.py         # ✅ M1 — `rdi-demo recover|parity`
│   ├── scoring/        # M2 — classical + MLX encoder + anomaly + conformal
│   ├── reasoning/      # M3 — LLM agent: explain + remediate (routed)
│   ├── decision/       # M4 — uplift + policy + experimentation
│   └── ops/            # M5 — drift, retrain, canary/shadow, rollback
├── tests/              # ✅ 75 passing
├── dashboard/          # M6 — Next.js real-time UI
└── data/               # gitignored
```

Single-file modules until a subsystem earns a package — `broker.py` is one class, and a
`broker/__init__.py` re-exporting it would be a directory pretending to be architecture.

## Quickstart

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
make install                     # pip install -e ".[dev]"
make test                        # 36 tests

make summary                     # ground-truth breakdown of the stream
make stream                      # labeled JSONL on stdout
make drift                       # M5 fixture: shifted baseline, zero incidents

make recover                     # M1 — kill a consumer mid-stream, lose nothing
make parity                      # M1 — train=serve, and the leak it prevents

rdi-stream --n 600 | jq -c 'select(.label == 1)'
```

## M1 results

**`make recover` — at-least-once holds (S1).** Ack-*after*-processing is the whole guarantee:
the crashed consumer's in-flight events stay in the broker's pending set and get redelivered.
Ack-before would be faster and would silently drop every event in flight at the crash.

```
appended              1200 events (fsync on)
consumer 'crash'      processed 400, then died
in flight at crash    16 events claimed but never acked
consumer 'rescue'     processed 800 (0 duplicates deduped)
total processed       1200 / 1200      still pending  0
EVENTS LOST: 0  ✅
```

**`make parity` — train=serve at 0.0 skew, and the leak quantified.** `compute_offline`
replays the *online* code rather than reimplementing it, so the two paths agree by
construction. The counterexample is kept and measured: a per-service mean over the whole
series — what you get when training is pandas and serving is a stream worker — is
contaminated by the very incidents it should expose, so **outages read milder than they are**:

| incident | online | leaky |
|---|---|---|
| `dependency_failure` | 5.52 | 3.00 |
| `bad_deploy` | 2.77 | 2.23 |
| `traffic_spike` | 3.80 | 2.69 |
| `memory_leak` | **2.77** | **1.56** |

*(mean `latency_over_baseline`)* A memory leak reading 1.56× instead of 2.77× is a feature
that is least trustworthy exactly where it matters most.

**A bug worth stating: a rolling *mean* baseline eats its own incident.** Over a sustained
outage the baseline climbs to meet the outage and `latency_over_baseline` decays toward 1.0 —
a 3.0× outage measured **1.13× by its end, 89% of the signal gone**. Incidents run 40–300
ticks, so this was the common case, not a corner. Fixed with a **median over a 900s window**:
incidents stay a minority of the window, and a median tolerates 50% contamination where a
mean tolerates none. Costs p99 **0.30ms** at a steady-state 901-deep window — measured, and
inside M2's sub-ms budget with room for the model.

`make summary` on the default seed — note the `label=1` column, which is the
anomaly-vs-incident split made visible:

```
events                3600
SLO                   latency_ms > 500  or  error_rate > 0.05
breaching (label=1)   419  (11.6%)
latency p50 / p99     90ms / 531ms

incident_type          count      of which breaching (label=1)
  normal                 2582           0  (0%)
  memory_leak             659         132  (20%)     ← ramps; most ticks pre-breach
  bad_deploy              159         159  (100%)    ← step change; breaks instantly
  traffic_spike           107          35  (33%)     ← the rest are absorbed: don't act
  dependency_failure       93          93  (100%)
```

Detecting `bad_deploy` and `dependency_failure` is trivial (the SLO alone catches them). The
hard part — and M4's actual job — is telling them *apart*, since they demand opposite
remediations (`rollback` vs `failover`).

## Tech stack

Python 3.12 · file-backed streams broker · asyncio workers · MLX (small encoder) ·
scikit-learn / LightGBM · from-scratch conformal + PSI · uplift/meta-learners ·
local Qwen2.5-1.5B (llama.cpp) + Claude API · RAG + MCP · FastAPI + WebSockets · Redis ·
SQLite/Postgres · Next.js · Prometheus/Grafana · MLflow/DVC · Docker.

## License

Private. All rights reserved.
