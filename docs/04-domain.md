# 04 — Domain decision (committed)

**Domain: AIOps / service reliability.** Chosen at Milestone 0 over fraud and healthcare
vitals. This is a *committed* decision — every layer downstream (features, action space,
uplift labels, dashboard) is built against the schema below.

## Why AIOps

- **It is not a re-skin of an existing repo.** `realtime-ml-pipeline` already ships fraud
  scoring; a fraud capstone would read as "the same project, bigger." AIOps shares the
  *machinery* (broker, PSI, conformal, uplift) and none of the domain.
- **The LLM has real work.** Every event carries a log line. Explaining an incident means
  reading text, not narrating a number — so RAG and groundedness scoring are load-bearing
  rather than decorative.
- **The action space is genuinely causal.** Four incident types × four remediations, where
  the *wrong* remediation has near-zero or negative effect (restarting a service whose
  upstream database is down costs downtime and fixes nothing). A binary treat/don't-treat
  can be faked with a risk threshold; this cannot. That is the whole point of Milestone 4.

## Event schema

One event per service per tick. Ground truth ships inline so every layer is measurable.

| Field | Type | Notes |
|---|---|---|
| `ts` | float | seconds since stream start |
| `service` | str | one of six fleet services |
| `version` | str | `v<major>.<minor>.<patch>`; bumps on deploy |
| `latency_ms` | float | p99 latency this tick |
| `error_rate` | float | 0..1 |
| `cpu_pct` | float | 0..100 |
| `mem_pct` | float | 0..100 |
| `rps` | float | request rate |
| `log` | str | a log line — the LLM's raw material |
| `incident_type` | str \| None | **what is happening** (ground truth); `None` = normal |
| `label` | int | **should we act** — 1 iff this tick breaches the SLO |

### `incident_type` and `label` are deliberately not the same thing

This is the schema's one real idea. An **anomaly** is not an **incident**:

- A traffic spike the service absorbs has `incident_type="traffic_spike"` and `label=0`.
  The detector *should* notice it. The decision layer should *not* pay to scale out.
- A memory leak in its first minutes has `incident_type="memory_leak"` and `label=0` — the
  leak is real but has not breached yet. Catching it here is exactly what early detection
  is worth.

Collapsing the two would let a risk-threshold policy look optimal and would erase the
uplift engine's job. Keeping them apart is what makes Milestone 4's honest research
question ("does the causal policy beat a naive risk threshold?") answerable.

### The label is derived, not hand-set

```
label = 1  iff  latency_ms > 500  or  error_rate > 0.05
```

The SLO *is* the ground truth definition — labels fall out of the emitted metrics rather
than being painted on by the generator. Two consequences worth stating: a noisy normal tick
can breach on its own (a real blip, not a bug), and incident ticks that stay under the SLO
are honestly labeled 0.

## Incident types and their signatures

Each type has a distinct multi-metric fingerprint. Any single metric is ambiguous; the
combination is not. That is what the scoring layer must learn.

| Type | Onset | Fingerprint | Correct remediation |
|---|---|---|---|
| `memory_leak` | slow (long ramp) | `mem_pct` climbs monotonically → `latency_ms` creeps up quadratically → `error_rate` rises late | `restart` |
| `dependency_failure` | fast | `error_rate` jumps (0.08–0.60 — upstreams degrade partially more often than they die), `latency_ms` spikes, `rps` dips slightly. **CPU is bimodal:** down when threads block on I/O, *up* when clients retry | `failover` |
| `traffic_spike` | fast | `rps` up 1.5–6×, `cpu_pct` up, `latency_ms` up, `error_rate` up 0.02–0.15 when the service saturates and starts returning 503s | `scale_out` |
| `bad_deploy` | step | `version` bumps, then `latency_ms` and `error_rate` step up **together**, sustained | `rollback` |

### Correction (M2): CPU is *not* the discriminator, and the hard pair isn't the one we named

This doc originally claimed `dependency_failure` shows flat/low CPU and that this is what
separates it from `traffic_spike`. **Measured, that was wrong**, for two reasons:

- **Retry storms.** When an upstream fails, clients retry, and retries burn CPU. About a
  third of dependency-failure ticks run CPU *above* baseline, so the distribution is bimodal
  and its mean (~0.9) describes no actual tick. A test asserting "dep CPU is low" passed on
  an average that misrepresented a third of the data.
- **`rps` was doing the work all along.** A traffic spike genuinely has traffic; a failing
  dependency does not. Boring, and true.

The real hard pair is **`dependency_failure` vs `bad_deploy`** (M2 recall 0.50, 84/184
misread). A retry storm drives CPU, latency and errors up together — a bad deploy's exact
shape — and benign deploys ship often enough that one plausibly landed just before. The
remediations are opposite (`failover` vs `rollback`), and **the metrics cannot settle it**.
The log line names the failing upstream, which is why M3's agent reads text rather than
narrating numbers.

### Benign deploys (added M2)

Real fleets ship constantly and ~93% of deploys are uneventful — a version bump and nothing
else. Without them, every version bump in the stream would be one that broke something and
`since_deploy_s` would be a perfect `bad_deploy` oracle: the model would learn
"a deploy just landed ⇒ bad deploy", score ~1.0, and have learned nothing. With them, deploy
recency is **necessary but not sufficient**.

## Action space

```
none · restart · failover · scale_out · rollback
```

The correct-remediation mapping above is ground truth for Milestone 4 — it is **not** a
field on the event. The uplift model has to recover it from the metrics, and the policy has
to weigh it against the cost of acting (every remediation but `none` costs something, so
acting on an absorbed spike is a real loss).

## Drift (Milestone 5)

`drifted=True` inflates fleet-wide baseline latency and shifts the request-rate
distribution — **no incidents are injected**. The world changed; nothing broke. PSI on
`latency_ms` must fire, and the incident alert must stay silent. That separation is the
specificity test in S7: a drift monitor that fires on incidents is just a bad detector.
