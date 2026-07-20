# Data model

The generator emits one event per service per tick, six services, one tick per second. Every
event carries ground truth so downstream results can be scored.

## Event schema

| Field | Type | Notes |
|---|---|---|
| `ts` | float | seconds since stream start |
| `service` | str | one of six fleet services |
| `version` | str | `v<major>.<minor>.<patch>`, bumps on deploy |
| `latency_ms` | float | p99 latency for this tick |
| `error_rate` | float | 0 to 1 |
| `cpu_pct` | float | 0 to 100 |
| `mem_pct` | float | 0 to 100 |
| `rps` | float | request rate |
| `log` | str | one log line |
| `incident_type` | str or None | what is happening, `None` when healthy |
| `label` | int | 1 when this tick breaches the SLO |

## Why there are two ground-truth fields

`incident_type` and `label` answer different questions, and keeping them apart is the point.

An anomaly is not automatically an incident. A traffic spike the service absorbs has
`incident_type="traffic_spike"` and `label=0`. The detector should notice it. Nothing should
pay to scale out for it. A memory leak in its first minutes is the same story: real, not yet
breaching, and worth catching early precisely because it hasn't broken anything yet.

Collapse the two fields and a plain risk threshold starts to look like a decision policy,
because every anomaly would be worth acting on by construction.

The label is derived, never hand-set:

```
label = 1  if  latency_ms > 500  or  error_rate > 0.05
```

The SLO constant is the definition. Two consequences follow: a healthy tick can breach on
noise alone, which is realistic, and incident ticks that stay inside the SLO are honestly
labelled 0.

## Incident types

Each type has a multi-metric fingerprint. No single metric identifies any of them.

| Type | Onset | Fingerprint | Remediation |
|---|---|---|---|
| `memory_leak` | slow ramp | `mem_pct` climbs steadily, latency creeps up quadratically, errors arrive late | restart |
| `dependency_failure` | fast | errors jump to somewhere between 0.08 and 0.60, latency spikes, `rps` dips. CPU is bimodal: down when threads block on I/O, up when clients retry | failover |
| `traffic_spike` | fast | `rps` up 1.5x to 6x, CPU up, latency up, errors up to 0.15 once the service saturates | scale_out |
| `bad_deploy` | step change | version bumps, then latency and errors step up together and stay there | rollback |

The remediation column is ground truth for evaluating a decision policy. It's deliberately not
a field on the event, so a model has to recover it from the metrics.

## Two things that make the data harder on purpose

**Retry storms.** When an upstream fails, clients retry, and retries burn CPU. About a third
of dependency-failure ticks run CPU above baseline rather than below. Without this, CPU alone
separates dependency failures from traffic spikes perfectly and the classification task stops
being a task. With it, the two classes overlap on CPU, latency and error rate, and only `rps`
still tells them apart.

**Benign deploys.** Roughly 93% of version bumps ship without incident. Without them, every
deploy in the stream would be a bad one and deploy recency would be a perfect oracle: a model
would learn "a deploy just landed means bad deploy", score near 1.0, and have learned nothing.
With them, deploy recency is necessary but not sufficient.

## Drift mode

`generate(drifted=True)` inflates fleet-wide baseline latency and shifts the request-rate
distribution while injecting no incidents at all. The world changed, nothing broke.

That separation matters for testing a drift monitor. Incident spikes move mean latency more
than a fleet-wide 1.8x baseline shift does, so a monitor comparing live traffic against a
training reference will read incidents as drift unless incident ticks are excluded from the
reference or the window outlasts an episode.
