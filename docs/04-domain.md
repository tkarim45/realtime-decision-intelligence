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
| `dependency_failure` | fast | `error_rate` jumps hard, `latency_ms` spikes, **`cpu_pct` flat or *down*** (threads blocked on I/O, not working) | `failover` |
| `traffic_spike` | fast | `rps` up 3–6×, `cpu_pct` up, `latency_ms` up, `error_rate` barely moves | `scale_out` |
| `bad_deploy` | step | `version` bumps, then `latency_ms` and `error_rate` step up **together**, sustained | `rollback` |

`dependency_failure` having *flat CPU* is the discriminator that separates it from
`traffic_spike` — both show high latency, only one shows the service actually working.

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
