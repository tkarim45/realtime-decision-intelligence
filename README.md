# Real-Time Multi-Modal Decision Intelligence System

> A production system that ingests a live multi-modal stream (events + text, optionally
> image/audio), scores it in real time with classical + small deep models, **explains and
> acts** on it with an LLM agent, and decides **interventions** with causal/uplift
> modeling — all under drift monitoring. Runs on an Apple M1 (8 GB): small on-device
> models for scoring, local + cloud LLM routed for reasoning.

**Status:** 📐 Spec / scaffold. No measured results yet. Build from
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

## Pick a domain

One domain with a public-ish stream and a natural intervention:
- **Fraud** — score transactions, agent explains a flag, intervention = block/step-up/allow.
- **AIOps / reliability** — score service metrics, agent explains an incident, remediate.
- **Healthcare monitoring** — score vitals stream, agent explains, intervention = alert tier.

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

## Repository layout (target)

```
realtime-decision-intelligence/
├── README.md · .gitignore · requirements.txt
├── docs/  00-what-it-is · 01-architecture · 02-build-plan · 03-setup
├── broker/         # durable event log (consumer groups, at-least-once)
├── features/       # online windowed features, train=serve
├── scoring/        # classical + MLX encoder + anomaly + conformal
├── reasoning/      # LLM agent: explain + remediate (routed)
├── decision/       # uplift + policy + experimentation
├── ops/            # drift, retrain, canary/shadow, rollback
├── dashboard/      # Next.js real-time UI
└── data/           # gitignored
```

## Quickstart (once Milestone 0 exists)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
pip install -r requirements.txt
make stream         # start the broker + a synthetic stream
make score          # scoring workers
make dashboard      # real-time UI
```

## Tech stack

Python 3.12 · file-backed streams broker · asyncio workers · MLX (small encoder) ·
scikit-learn / LightGBM · from-scratch conformal + PSI · uplift/meta-learners ·
local Qwen2.5-1.5B (llama.cpp) + Claude API · RAG + MCP · FastAPI + WebSockets · Redis ·
SQLite/Postgres · Next.js · Prometheus/Grafana · MLflow/DVC · Docker.

## License

Private. All rights reserved.
