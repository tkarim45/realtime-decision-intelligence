# Real-Time Multi-Modal Decision Intelligence System

> A production system that ingests a live multi-modal stream (events + text, optionally
> image/audio), scores it in real time with classical + small deep models, **explains and
> acts** on it with an LLM agent, and decides **interventions** with causal/uplift
> modeling — all under drift monitoring. Runs on an Apple M1 (8 GB): small on-device
> models for scoring, local + cloud LLM routed for reasoning.

**Status:** 🚧 **M0–M2 of 6 complete.** Domain committed (**AIOps**); labeled synthetic
telemetry stream; durable at-least-once broker with **crash recovery proven (1200/1200, 0
lost)**; train=serve online features at **0.0 skew**; incident classifier at **macro-F1
0.863**; unsupervised detector that catches an incident class the classifier was never
trained on; conformal sets whose **coverage guarantee holds**; hot path at **p99 0.919ms,
2,513 events/s — sub-ms SLO met**. 120 tests, ruff clean.

**Still to build: M3 reasoning (LLM agent), M4 causal uplift, M5 drift/canary, M6 dashboard.**
Those are the reason this is a 6–12-month capstone, and they are not started. Build from
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
│   ├── model.py        # ✅ M2 — LightGBM incident classifier + honest splits
│   ├── anomaly.py      # ✅ M2 — IsolationForest on normal ticks only
│   ├── conformal.py    # ✅ M2 — split-conformal prediction sets
│   ├── scoring.py      # ✅ M2 — the assembled hot path + near-line detector
│   ├── demo.py         # ✅ runnable artifacts for every milestone
│   ├── reasoning/      # M3 — LLM agent: explain + remediate (routed)
│   ├── decision/       # M4 — uplift + policy + experimentation
│   └── ops/            # M5 — drift, retrain, canary/shadow, rollback
├── tests/              # ✅ 120 passing
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

make score                       # M2 — incident classifier, temporal vs leaky split
make uncertainty                 # M2 — unknown-unknowns detector + conformal sets
make loadtest                    # M2 — hot-path latency, and what came off it

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

## M2 results (step 1 of 5)

**The model does not predict `label`.** The SLO breach is a deterministic function of two
features — 0/3600 disagreements with the raw if-statement — so a classifier on it would score
~1.0 and be an if-statement in disguise. The target is `incident_type`, because that is what
picks the remediation. `label` is an observation for M4, not a target.

**The generator was too clean, and it would have made M2–M4 vacuous.** With the M0
fingerprints the honest macro-F1 was 0.970 and `dependency_failure`/`traffic_spike` both hit
1.00 — `cpu`, `rps` and `error_rate` each separated them perfectly *and independently*. A
saturated classifier means always-singleton conformal sets, no ambiguity for the agent, and
trivial uplift. Fixed with realism (retry storms, partial upstream degradation, saturation
errors, modest spikes, benign deploys), each justified on its own terms.

```
TEMPORAL SPLIT (train on the past, test on the future)   macro-F1 0.863
  class                    P     R     F1      n
  normal                0.97  0.99   0.98   8588
  memory_leak           0.93  0.84   0.88   1321
  dependency_failure    0.83  0.50   0.62    184     ← the hard class
  traffic_spike         1.00  0.89   0.94    238
  bad_deploy            0.80  1.00   0.89    361

RANDOM SPLIT (leaky)                                     macro-F1 0.983
  inflation             +0.121 macro-F1 of illusion
```

**A leak only shows where there's headroom.** The random-split leak (an episode spans 40–300
near-identical ticks, so shuffling scores the model on ticks whose neighbours it memorized)
was worth **+0.020** on the saturated generator and **+0.121** on the realistic one. Same
leak — the easy task had no room to show it.

**The honest weakness: `dependency_failure` recall 0.50, and 84/184 are misread as
`bad_deploy`.** A retry storm drives CPU, latency and errors up together — a bad deploy's
exact shape — and benign deploys ship often enough that one plausibly landed just before. The
remediations are opposite (`failover` vs `rollback`). **The metrics cannot settle it**; the
log line names the failing upstream, which is why M3's agent reads text.

**Corrections this forced to earlier claims:** CPU is *not* the discriminator (retry storms
make it bimodal — its mean of ~0.9 describes no actual tick); `rps` was doing the work. And a
plain time cut left `bad_deploy` with **zero test examples** while `evaluate` quietly averaged
over the survivors — both now raise instead of reporting a 4-class number as macro-F1.

### Uncertainty: one layer earned its place, one prediction failed (`make uncertainty`)

**The anomaly detector earns its place.** It trains on **normal ticks only**, so it can flag
what it has never seen. Hide `memory_leak` from both models: the classifier — which cannot say
"I don't know" — calls it `normal` **39%** of the time, while the detector still flags **77%**
of it. Their per-class strengths are **anti-correlated (−0.66)**:

| class | classifier recall | detector flagged |
|---|---:|---:|
| `normal` | 0.99 | 2.6% |
| `memory_leak` | 0.84 | 76.5% |
| `dependency_failure` | **0.50** | **98.9%** |
| `traffic_spike` | 0.89 | 100.0% |
| `bad_deploy` | **1.00** | **30.2%** |

The classifier's worst class is the detector's best and vice versa — that, not intuition, is
what justifies a second model.

**But it is not a safety net, and the difference matters.** Class-level complementarity does
not imply event-level complementarity. The detector rescues only **18.8%** of the classifier's
misses, and "classifier says normal, detector disagrees" is **15.5% incidents against a 19.7%
base rate — worse than guessing**. Both fail on the same mild, pre-breach ticks.

**Conformal delivers its guarantee** (0.919 at a 90% target, 0.962 at 95%), while naive
softmax thresholding cannot be dialled at all (0.960 regardless of α).

**And the prediction in the build plan was wrong.** Conformal was supposed to hedge the
`dependency_failure`/`bad_deploy` confusion so an ambiguous set could route to the LLM. It
does not — when the model misreads, it assigns **P(bad_deploy)=0.973** and
**P(dependency_failure)=0.017**. It is *confidently wrong, not uncertain*, and conformal can
only widen a set the model already hesitates on; it cannot manufacture doubt. Forcing the
hedge needs a 99% target, which makes **42% of the whole stream** ambiguous. Uncertainty
quantification is not a substitute for information the metrics never contained.

### The hot path, and what had to come off it (`make loadtest`)

```
HOT PATH  features -> classifier -> conformal set -> act/escalate
  p50 / p95 / p99     0.366 / 0.572 / 0.919 ms
  throughput          2,513 events/s single-threaded
  sub-ms p99 SLO      MET
```

**The anomaly detector had to leave the hot path to get there.** Measured per call: features
0.058ms, classifier 0.205ms, `detector.flag` **5.9ms** — 29× the classifier and 99% of the
budget, blowing the SLO by 7×. It is sklearn per-call dispatch, not compute: the same forest
costs **0.032ms/event at batch 256 (186× cheaper)**. So it runs batched on its own consumer
group — and still catches the unknown-unknowns it exists for, one batch later.

**And off the path was not enough — it had to leave the thread.** Co-locating the batched
detector, *with its own cost excluded from the timing*, still inflated p99 **0.919 → 2.166ms**
and max **3.04 → 38.29ms**, because a 6ms tree walk every 256 events evicts cache under the
events that follow. (Not GC — disabling it doesn't help.) **"Don't count it" is not latency
isolation; only a separate process is.** The architecture always said that about the LLM;
measurement said it one layer further down.

### The MLX encoder was cut, on evidence

The build plan set the bar: *beat 0.863 macro-F1 or be cut*. Rather than assert it was
unnecessary, I tested the premise — a 5× richer temporal feature space (8 → 40 features, lags
1/2/4/8 per service, a cheap proxy for what a sequence encoder learns) moved macro-F1 only
0.863 → 0.880 **and made the target class worse**: `dependency_failure` recall 0.50 → 0.41.
The bottleneck is not model capacity or temporal context. The discriminating information is
not in the metrics at all — it is in the log line, which is M3's job.

> **On the latency numbers:** they are machine-dependent and deliberately *not* asserted in
> the test suite. An earlier version asserted p99 < 2ms and failed at 2.83ms purely from
> contention with other tests; the co-location effect flipped sign under the same noise. The
> suite keeps an order-of-magnitude regression guard; the real numbers come from
> `make loadtest` in a clean process. A flaky test that encodes a claim is worse than no test.

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
breaching (label=1)   442  (12.3%)
latency p50 / p99     89ms / 552ms

incident_type          count      of which breaching (label=1)
  normal                 2599           0  (0%)
  memory_leak             621         144  (23%)     ← ramps; most ticks pre-breach
  bad_deploy              159         159  (100%)    ← step change; breaks instantly
  traffic_spike           129          47  (36%)     ← the rest are absorbed: don't act
  dependency_failure       92          92  (100%)
```

*Detecting* `bad_deploy` and `dependency_failure` is trivial — the SLO alone catches them, at
100% each. The hard part is telling them *apart*, since they demand opposite remediations
(`rollback` vs `failover`), and **M2 measured that as the real weakness**: recall 0.50, with
84/184 dependency failures misread as bad deploys. See [M2 results](#m2-results-step-1-of-5).

## Tech stack

**Built (M0–M2), and it is a short list on purpose:** Python 3.12 · NumPy · scikit-learn ·
LightGBM · a from-scratch file-backed streams broker (Redis-Streams contract) · from-scratch
online windowed features · split-conformal from scratch · pytest · ruff. The core installs
with three dependencies and runs offline on an M1.

**Planned (M3–M6), not in the repo yet:** local Qwen2.5-1.5B (llama.cpp) + Claude API routed ·
RAG + MCP · uplift/meta-learners · from-scratch PSI · FastAPI + WebSockets · Next.js ·
Prometheus/Grafana · Docker. Each arrives with the milestone that needs it — see
[`docs/03-setup.md`](docs/03-setup.md) for the per-milestone dependency table.

## License

MIT — see [`LICENSE`](LICENSE). Matches `pyproject.toml` and the sibling repos.
