# 01 — Architecture

## Subsystems

### 1. Durable ingestion (`broker/`, `features/`)
- **Event log** with consumer groups + **at-least-once delivery**; crash recovery proven
  (kill mid-stream, resume, 0 events lost). Reuse the file-backed
  Redis-Streams-contract broker from `realtime-ml-pipeline` — no Kafka cluster on 8 GB.
- **Online windowed features** with **train=serve consistency**: the *same* feature code
  runs offline (training) and online (serving). This prevents the train/serve skew the
  `feature-store` repo exists to demonstrate.

### 2. Multi-model scoring (`scoring/`)
- **Classical hot path**: scikit-learn / LightGBM, sub-ms per event.
- **Small deep encoder**: a 1–20M-param **MLX sequence encoder** trained on-device — adds
  temporal/contextual signal without heavy latency or memory.
- **Anomaly detector**: IsolationForest / from-scratch, for unknown-unknowns.
- **Ensemble + calibration**: combine scores, calibrate probabilities.
- **Conformal prediction** (from-scratch split-conformal): each decision gets a
  guaranteed-coverage confidence set. Reuse the `conformal-prediction` repo.

### 3. GenAI reasoning (`reasoning/`)
- Triggered only when an event is **flagged** (slow path — protects the latency budget).
- Agent pulls context: feature store snapshot, recent logs, domain docs via RAG.
- Writes a **grounded explanation + recommended action**; can call **MCP tools** to
  remediate (block, alert, escalate).
- **Routed**: local Qwen2.5-1.5B for cheap/offline explanations, Claude API for hard ones.
- Every explanation **scored for groundedness** (does it cite real feature values / logs?).

### 4. Causal decision engine (`decision/`)
- **Uplift / meta-learners**: estimate the *incremental* effect of each candidate action
  (treat vs not), not just risk. Reuse `uplift-targeting-engine`.
- **Policy**: pick treat / don't-treat per event to maximize expected incremental value.
- **Experimentation engine**: outcomes feed a peeking-safe evaluator (two-proportion /
  Welch, Beta-Binomial Bayesian, always-valid mSPRT). Reuse `experimentation-engine`.
- Validate the policy with **Qini / policy value** vs a naive risk-threshold baseline.

### 5. MLOps & drift (`ops/`)
- **PSI drift** on the live stream (from-scratch); alert fires on drifted data, silent on
  clean. Reuse `realtime-ml-pipeline` PSI.
- **Retrain trigger** on drift; **canary + shadow** deploys; **instant rollback**.
- **Load shedding** (429 + Retry-After) under burst; request tracing; cost/latency/quality
  metrics.

### 6. Product surface (`dashboard/`)
- Real-time **ops dashboard** (Next.js + WebSockets): live event feed, per-decision
  explanation + confidence set + recommended action, drift/health panels, experiment
  results view.

## Latency budget (the core design constraint)

```
event ─▶ features (μs) ─▶ classical score (sub-ms) ─▶ ensemble+conformal (sub-ms) ─▶ decision
                                                              │ only if flagged
                                                              ▼
                                            LLM explain/remediate (100s of ms — SLOW PATH)
```
The hot path (score + decide) must meet the p99 SLO **without** the LLM. The LLM runs only
on flagged events, asynchronously, so it never gates the stream. This routing is the whole
latency trick — get it wrong and throughput collapses.

## Data flow

```
stream ─▶ durable log ─▶ online features ─▶ scoring (classical+MLX+anomaly, conformal)
   │                                              │
   │                                        flag? ├─▶ LLM agent (routed) ─▶ explanation+action
   │                                              ▼
   │                                     uplift policy ─▶ treat/don't-treat
   │                                              ▼
   │                                     experiment engine ─▶ measured lift
   └─▶ PSI drift ─▶ retrain ─▶ canary/shadow ─▶ rollback        ▼
                                                          dashboard
```

## Build-order dependency

```
broker+features ─▶ scoring ─▶ reasoning ─▶ decision ─▶ ops ─▶ dashboard
        (each layer needs the one before it working on a live stream)
```
