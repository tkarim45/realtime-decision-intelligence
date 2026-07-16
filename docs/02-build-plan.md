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

## Milestone 1 — Durable ingestion + features ✅ DONE

**Goal:** at-least-once stream with proven crash recovery + train=serve features.

Steps:
1. ✅ **File-backed streams broker** (`src/rdi/broker.py`, adapted from
   `realtime-ml-pipeline`): consumer groups, offsets, at-least-once delivery. Two changes
   from the sibling — the file handle stays open (reopening per append would have put a
   syscall floor under M2's throughput that has nothing to do with the model), and `fsync`
   is a real option rather than a docstring promise.
2. ✅ **Crash recovery**: `make recover` kills a consumer mid-stream (400 processed, 16 in
   flight), restarts, and finishes **1200/1200 — 0 events lost**, 0 pending. Ack-after-
   processing is what buys this; the duplicates it implies are deduped by offset.
3. ✅ **Online windowed features** (`src/rdi/features.py`), 8 features each earning its place
   against a documented discriminator. `compute_offline` **replays the online code** rather
   than reimplementing it — `make parity` shows max abs difference **0.0**.

**Artifact:** ✅ `make recover` (0 lost) + `make parity` (0.0 skew, and the leak quantified).
75 tests, ruff clean.

**What M1 established:**

- **Train=serve is structural, not asserted.** One implementation, replayed offline. The
  parity test is near-tautological *by design* and exists to fail loudly if someone later
  "optimizes" training into a second implementation. The test with teeth is
  `test_truncating_the_future_does_not_change_the_past` — a causality property that any
  vectorized batch rewrite fails.
- **The leak, quantified.** `compute_offline_leaky` (a per-service mean over the whole
  series — what you write when training is pandas and serving is a stream worker) is kept as
  a *measured counterexample*. It understates every incident because a service's mean latency
  includes its own outages: `dependency_failure` 5.52 → 3.00, `memory_leak` **2.77 → 1.56**.
  The feature is least trustworthy exactly where it matters.
- **A real bug, found by a failing test: a rolling *mean* baseline eats its own incident.**
  Over a sustained outage the baseline climbs to meet the outage and
  `latency_over_baseline` decays toward 1.0 — measured, a 3.0× outage read as **1.13× by its
  end (89% of the signal gone)**. Since incidents run 40–300 ticks, this was the common case.
  Fixed with a **median over a 900s window**: incidents stay a minority of the window, and a
  median tolerates 50% contamination where a mean tolerates none. (p25 also resists poisoning
  but the diurnal cycle biases it low, so a healthy service would read ~1.3×.)
- **Hot-path budget measured, not assumed.** The median is O(window) per metric; at a
  steady-state 901-deep window it costs **p99 0.30ms** (~6,755 events/s single-threaded),
  leaving room for M2's model inside the sub-ms SLO. Guarded loosely at p99 < 2ms.

---

## Milestone 2 — Scoring layer (Week 7–12) — 🚧 step 1 of 5 done

**Goal:** real-time scoring meeting a latency SLO, with calibrated uncertainty.

**What step 1 established (and had to fix first):**

- **The target is `incident_type`, not `label`.** The SLO breach is a deterministic function
  of two features (`latency_ms > 500 or error_rate > 0.05`) — 0/3600 disagreements with the
  raw if-statement. A classifier on it would score ~1.0 and be an if-statement in disguise.
  `label` is an *observation* for M4 (was an intervention warranted?), never a target.
- **The generator was too clean, and it would have made M2–M4 vacuous.** With the M0
  fingerprints the honest macro-F1 was 0.970 and `dependency_failure`/`traffic_spike` both
  scored 1.00 — `cpu`, `rps` and `error_rate` each separated them *perfectly and
  independently*. A saturated classifier means always-singleton conformal sets (M2), no
  ambiguity for the agent (M3), and trivial uplift (M4). Fixed with realism, each change
  justified on its own: **retry storms** (retries burn CPU, so a third of dependency failures
  run CPU *up*), **partial upstream degradation** (errors 0.08–0.60), **saturation errors** on
  breaking spikes, **modest 1.5× spikes**, and **benign deploys** (~93% of deploys are
  uneventful — without them `since_deploy_s` was a perfect `bad_deploy` oracle).
- **Honest score after realism: macro-F1 0.863** (temporal split). The hard class is
  `dependency_failure` at **recall 0.50**, and it is mistaken for **`bad_deploy`** (84/184) —
  the metrics genuinely cannot settle it, which is M3's reason to read logs.
- **A leak only shows where there's headroom.** The random-split leak was worth **+0.020**
  macro-F1 on the saturated generator and **+0.121** on the realistic one. Same leak; the
  saturated task had no room to show it.
- **A silent eval bug:** a plain time cut left `bad_deploy` with *zero* test examples and
  `evaluate` quietly averaged over the survivors, reporting a 4-class number as macro-F1.
  Both `temporal_split` and `evaluate` now raise instead.
- **Warnings are errors now.** A sklearn feature-name warning fired on every predict for a
  whole milestone unread; an attempted fix was a silent no-op (`feature_names_in_` is a
  read-only property) and only warnings-as-errors caught that.

Steps:
1. ✅ **Classical hot-path model** (`src/rdi/model.py`, LightGBM): 5-class `incident_type`,
   temporal split, macro-F1 **0.863**. `make score`. Sub-ms serving measured in step 5.
2. Add the **anomaly detector** (IsolationForest / from-scratch). Reuse
   `timeseries-anomaly-detection`. The test that earns it: hold out one incident type from
   training, show the classifier mislabels it (it can only name what it has seen) while the
   detector still flags it. That is what "unknown-unknowns" has to mean to be worth the code.
3. **Ensemble + calibrate**; add **conformal prediction** (reuse `conformal-prediction`) —
   verify empirical coverage hits the target on the stream. The payoff is now real: on the
   `dependency_failure`/`bad_deploy` confusion the set should come back
   `{dependency_failure, bad_deploy}` — an explicit "the metrics can't settle this", which is
   exactly when M3 should read the log and M4 should not auto-remediate.
4. Train the **small MLX sequence encoder** (1–20M params) on-device for richer signal; keep
   it small enough to run inline. (See `docs/03-setup.md` for MLX.) Honest bar: it must beat
   0.863 macro-F1 on the *temporal* split to justify its latency and memory, or it gets cut.
5. **Load-test** the hot path: measure throughput + p50/p99; state SLOs. Budget so far —
   features cost p99 0.30ms of a sub-ms target.

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
- [x] **M1 Ingestion — 0-loss crash recovery (1200/1200) + train=serve parity (0.0 skew)**
- [ ] M2 Scoring — p99 SLO + conformal coverage
- [ ] M3 Reasoning — grounded explanation + remediation, hot path unaffected
- [ ] M4 Decision — validated causal lift, peeking-safe
- [ ] M5 Ops — drift specificity + canary/rollback
- [ ] M6 Dashboard — deployed demo + recording
