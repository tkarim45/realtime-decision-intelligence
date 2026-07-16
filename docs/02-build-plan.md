# 02 — Build plan (step by step)

Build bottom-up: a live stream first, then scoring, then reasoning, then the decision
layer, then ops, then the dashboard. Each layer must work on a **running stream** before
the next. Everything fits the M1; cluster scale-out is an optional documented cloud step.

---

## Milestone 0 — Foundations ✅ DONE

**Goal:** repo skeleton, env, domain chosen, a synthetic stream flowing.

Steps:
1. ✅ Domain committed: **AIOps / service reliability**; intervention = remediation
   (`restart` / `failover` / `scale_out` / `rollback` / `none`). See `docs/04-domain.md`.
2. ✅ `personal` conda env; `src/rdi/` package (subsystems land as milestones complete —
   no empty placeholder packages), Makefile, pytest, ruff, hatchling. **No black** — every
   sibling repo here is hand-formatted compact and black would reformat all 18 of their
   source files; `make lint` enforces line length + import order instead.
3. ✅ **Synthetic stream generator** (`src/rdi/events.py`): six services × one tick/sec,
   four incident types with distinct multi-metric fingerprints, a log line per event, and
   two separate ground-truth fields (`incident_type` = what's happening, `label` = SLO
   breach). Label is derived from the SLO, not painted on.

**Artifact:** ✅ `make stream` emits labeled JSONL; `make summary` prints the ground-truth
breakdown; `make drift` emits the M5 fixture. 36 tests, ruff clean.

**Carried forward — two things M0 established that later milestones must respect:**
- **Anomaly ≠ incident.** `incident_type` and `label` are deliberately non-redundant
  (absorbed spikes and pre-breach leaks are label=0). M4's honest question ("does the causal
  policy beat a naive risk threshold?") is only answerable because of this.
- **A PSI confound, found by a failing test.** The clean stream's *mean* latency is higher
  than the drifted stream's — a few big incident spikes outweigh a fleet-wide 1.8× baseline
  shift. So naive PSI against a training reference reads **incidents as drift**. M5 must
  exclude incident ticks from the reference or use a window that outlasts an episode.
  Pinned by `test_incidents_inflate_clean_mean_above_drifted_mean`.

---

## Milestone 1 — Durable ingestion + features (Week 3–6)

**Goal:** at-least-once stream with proven crash recovery + train=serve features.

Steps:
1. Stand up the **file-backed streams broker** (reuse `realtime-ml-pipeline`): consumer
   groups, offsets, at-least-once delivery.
2. **Crash-recovery test**: kill a consumer mid-stream, restart, assert **0 events lost**.
3. Implement **online windowed features** with the *same code* used offline for training
   (train=serve). Write a test that proves offline and online feature vectors match.

**Artifact:** crash-recovery test (0 lost) + a train=serve feature-parity test.

---

## Milestone 2 — Scoring layer (Week 7–12)

**Goal:** real-time scoring meeting a latency SLO, with calibrated uncertainty.

Steps:
1. Train the **classical hot-path model** (LightGBM/sklearn) offline; serve it sub-ms.
2. Train the **small MLX sequence encoder** (1–20M params) on-device for richer signal;
   keep it small enough to run inline. (See `docs/03-setup.md` for MLX.)
3. Add the **anomaly detector** (IsolationForest / from-scratch). Reuse
   `timeseries-anomaly-detection`.
4. **Ensemble + calibrate**; add **conformal prediction** (reuse `conformal-prediction`) —
   verify empirical coverage hits the target on the stream.
5. **Load-test** the hot path: measure throughput + p50/p99; state SLOs.

**Artifact:** load-test report (throughput + p99) + a conformal coverage plot.

---

## Milestone 3 — GenAI reasoning (Week 13–17)

**Goal:** grounded explanation + remediation on flagged events — on the slow path.

Steps:
1. On a flag, assemble context: feature snapshot + recent logs + domain docs via RAG.
2. Agent writes a **grounded explanation + recommended action**; wire an **MCP tool** to
   remediate (e.g. block / escalate).
3. **Route**: local Qwen2.5-1.5B for cheap/offline, Claude for hard cases.
4. **Score groundedness** of each explanation (does it reference real feature values?).
5. **Prove the latency isolation**: confirm the LLM (slow path) does **not** degrade the
   hot-path p99 (run LLM async, off the stream-critical path).

**Artifact:** a trace showing explanation + tool-call remediation + a p99 chart proving
the hot path is unaffected by LLM load.

---

## Milestone 4 — Causal decision engine (Week 18–23)

**Goal:** decide *interventions* causally and validate they help.

Steps:
1. **Uplift / meta-learners** (reuse `uplift-targeting-engine`): estimate incremental
   effect of each action per event.
2. **Policy**: treat/don't-treat to maximize expected incremental value.
3. **Validate** with Qini / policy value vs a naive risk-threshold policy — the causal
   policy should win on *incremental* outcome.
4. **Experimentation engine** (reuse `experimentation-engine`): route outcomes into a
   peeking-safe evaluator (Bayesian + mSPRT); report measured lift.

**Artifact:** uplift eval (Qini curve) + an experiment report showing validated lift,
peeking-safe.

---

## Milestone 5 — MLOps & drift (Week 24–28)

**Goal:** keep it alive — drift, retrain, canary/shadow, rollback.

Steps:
1. **PSI drift monitor** (from-scratch, reuse `realtime-ml-pipeline`) on the live stream.
   ⚠️ **M0 proved naive PSI fails here**: incident spikes move the mean more than the
   fleet-wide baseline shift does, so PSI-vs-training-reference fires on incidents, not
   drift. Exclude incident ticks from the reference or window past an episode — and keep
   `test_incidents_inflate_clean_mean_above_drifted_mean` green.
2. **Drift test**: run a clean stream (no alert) and a deliberately-drifted stream (alert
   fires) — prove specificity. `make drift` is the fixture (zero incidents injected).
3. **Retrain trigger** on drift → **canary** (small % of traffic) + **shadow** (score, don't
   act) → **instant rollback** on canary regression.
4. **Load shedding** (429 + Retry-After) under burst; request tracing; metrics endpoint.

**Artifact:** drift test (fires on drifted, silent on clean) + a canary→rollback demo.

---

## Milestone 6 — Real-time dashboard & polish (Week 29–36)

**Goal:** make it a live product.

Steps:
1. Next.js + WebSocket **ops dashboard**: live event feed, per-decision explanation +
   confidence set + action, drift/health panels, experiment-results view.
2. `docker-compose` for the full local stack; working README quickstart.
3. A 60–90s screen recording of the full loop (event → score → explain → decide → measure
   → drift → rollback) for the portfolio.

**Artifact:** deployed demo (local or small cloud instance) + recording.

---

## Optional cloud burst (any time after M2)

Want real throughput at scale or a Kafka cluster? Script a cloud deploy (managed Kafka +
K8s). **Document it as optional** — the laptop version stands alone with the file-backed
broker.

---

## Milestone checklist

- [x] **M0 Foundations — AIOps domain committed + labeled synthetic stream (36 tests)**
- [ ] M1 Ingestion — 0-loss crash recovery + train=serve parity
- [ ] M2 Scoring — p99 SLO + conformal coverage
- [ ] M3 Reasoning — grounded explanation + remediation, hot path unaffected
- [ ] M4 Decision — validated causal lift, peeking-safe
- [ ] M5 Ops — drift specificity + canary/rollback
- [ ] M6 Dashboard — deployed demo + recording
