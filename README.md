# realtime-decision-intelligence

Detects incidents in service telemetry, works out which ones are worth acting on, and picks
the remediation. Runs on a laptop.

Six services emit metrics once a second. The pipeline ingests them through a durable log,
computes windowed features, classifies the incident, attaches a prediction set with a coverage
guarantee, reads the log line when the metrics are ambiguous, and decides an action by
estimating what that action is actually worth. The hot path holds a p99 of 0.919ms at roughly
2,500 events per second on one core.

It's a benchmark and a reference implementation, not a production service. The telemetry is
synthetic, generated with labelled incidents so every claim below can be measured against
ground truth.

## Install

Needs Python 3.12. Three runtime dependencies: NumPy, scikit-learn, LightGBM.

```bash
pip install -e ".[dev]"
make test
```

## Try it

```bash
make summary       # ground-truth breakdown of the generated stream
make stream        # labelled telemetry as JSONL on stdout
make recover       # kill a consumer mid-stream, show nothing is lost
make parity        # offline and online features are byte-identical
make score         # train the classifier, temporal vs shuffled split
make uncertainty   # anomaly detection and conformal prediction sets
make loadtest      # hot-path latency breakdown
make decide        # the intervention policy against its baselines
make pipeline      # every layer end to end on one stream
```

Pipe the stream anywhere:

```bash
rdi-stream --n 600 | jq -c 'select(.label == 1)'
```

## What's in the box

**Durable event log** (`broker.py`) implementing the Redis Streams contract on an append-only
file. `append`, `claim`, `ack` and `redeliver` map to `XADD`, `XREADGROUP`, `XACK` and
`XAUTOCLAIM`, with consumer groups and a pending ledger for in-flight messages. Swapping in
real Redis is one adapter behind the same five methods.

**Online windowed features** (`features.py`) computed per service against that service's own
recent baseline. A 400ms response is fine for a service that always runs at 380ms and a fire
for one that idles at 60ms, so most features are ratios rather than absolutes.

**Incident classifier** (`model.py`), a LightGBM model over five classes: normal, memory
leak, dependency failure, traffic spike, bad deploy. The class matters because each one calls
for a different remediation, and picking the wrong remediation is worse than picking none.

**Unsupervised detector** (`anomaly.py`), an IsolationForest fitted only on healthy traffic so
it can flag failure modes that were never labelled.

**Conformal prediction** (`conformal.py`), split-conformal sets carrying a distribution-free,
finite-sample coverage guarantee. Ambiguous sets route events away from automatic action.

**Log reading** (`reasoning.py`) for the events the metrics can't separate. A retry storm and a
bad release look identical on every metric, and the classifier is confidently wrong rather than
uncertain, so no amount of calibration helps. The log line names the failing upstream. Ships a
deterministic offline reader and a real-model reader behind the same interface.

**Intervention policy** (`decision.py`). Knowing what broke isn't knowing what to do. Every
remediation costs something and the wrong one has negative effect, so a T-learner estimates the
incremental effect of each action per event and the policy acts only when that clears the cost.

**Drift and rollout** (`ops.py`), PSI against a healthy-only reference, plus shadow scoring,
canary releases and one-step rollback.

**The assembled loop** (`pipeline.py`), all of the above running as one system.

## How it works

```
telemetry ─▶ durable log ─▶ features ─▶ classifier ─▶ conformal set ─▶ act / escalate
                  │                          │              │
                  │                          │              └─▶ read the log line
                  │                          │                    (ambiguous events only)
                  │                          ├─▶ anomaly detector (batched, off-thread)
                  │                          └─▶ uplift ─▶ action, or none
                  └─▶ at-least-once delivery      │
                                                  └─▶ drift monitor ─▶ retrain signal
```

The anomaly detector and the log reader both sit off the request path. More on why below.

## Results

Measured on generated telemetry, 6,000 ticks across six services.

| Property | Result |
|---|---|
| Crash recovery | consumer killed with 16 events in flight, 0 of 1200 lost, pending ledger empty |
| Training/serving feature skew | 0.0 (max absolute difference) |
| Incident classification | macro-F1 0.863 on a temporal split |
| Hot path latency | p50 0.366ms, p95 0.572ms, p99 0.919ms |
| Throughput | 2,513 events/sec, single-threaded |
| Conformal coverage | 0.962 against a 95% target, 0.919 against 90% |

Per-class results, and how the two detection layers compare:

| Class | Classifier recall | Detector flag rate |
|---|---:|---:|
| normal | 0.99 | 2.6% |
| memory_leak | 0.84 | 76.5% |
| dependency_failure | 0.50 | 98.9% |
| traffic_spike | 0.89 | 100.0% |
| bad_deploy | 1.00 | 30.2% |

Their strengths run opposite to each other (correlation -0.66). The classifier's worst class
is the detector's best, and vice versa, which is the reason both are here.

Reading the log line for the 5% of events the metrics can't separate:

| | macro-F1 | dependency_failure recall |
|---|---:|---:|
| classifier alone | 0.863 | 0.50 |
| plus log reading | 0.968 | 0.96 |

Choosing an action, scored as breaches avoided net of what the actions cost:

| Policy | Value | Events treated |
|---|---:|---:|
| causal, T-learner, cost-aware | **245.1** | 845 |
| risk threshold, top 10%, runbook action | 206.3 | 1,070 |
| causal, per-class table | 108.6 | 564 |
| treat every alert, runbook action | 105.7 | 1,974 |
| treat nobody | 0.0 | 0 |
| risk threshold, top 25%, runbook action | **-104.9** | 2,673 |

Drift detection, on windows of 600 events:

| Stream | Windows alerting |
|---|---:|
| clean, contains real incidents | 8% |
| shifted baseline, no incidents | 100% |

## Design notes

A few results worth writing down, mostly because they contradicted what the design assumed.

**The anomaly detector can't sit on the hot path.** Per-call cost broke down as 0.058ms for
features, 0.205ms for the classifier, and 5.9ms for `IsolationForest.flag`. That last number
is 29x the classifier and would blow a sub-millisecond budget seven times over. It isn't
compute, it's scikit-learn's per-call dispatch: the same forest costs 0.032ms per event when
scored in batches of 256, about 186x cheaper. So the detector runs batched on its own consumer
group and still catches what it's there to catch, one batch later.

**Moving it off the path wasn't enough. It had to leave the thread.** Running the batched
detector in the same loop, with its cost excluded from the measurement, still pushed hot-path
p99 from 0.919ms to 2.166ms and the max from 3.04ms to 38.29ms. A 6ms tree walk every 256
events evicts cache under whatever runs next. It isn't garbage collection either, since
disabling the collector changes nothing.

**Conformal sets can't hedge a confident mistake.** The design expected ambiguous sets to
cover the dependency_failure/bad_deploy confusion, since both look similar in the metrics. They
don't. When the model gets that pair wrong it assigns 0.973 to the wrong class and 0.017 to the
right one, so it's wrong rather than uncertain, and conformal only widens sets the model
already hesitates on. Forcing the hedge needs a 99% target, which turns 42% of the stream
ambiguous. Uncertainty quantification won't recover information the inputs never carried.

**Shuffled splits look better than they are.** An incident episode spans 40 to 300 
near-identical ticks, so a random split scores the model on ticks whose neighbours it
memorised. That's worth +0.121 macro-F1 of nothing. It was only +0.020 on an earlier, easier
version of the generator, which is its own lesson: a leak only shows up where there's headroom
for it to show up.

**A risk threshold works right up until it doesn't.** Ranking by risk and applying the runbook
action is a strong baseline, and at the top 10% it scores 206.3. Move the cutoff to the top 25%
and the same rule scores -104.9, because it starts acting on events that were never going to
breach and the costs outrun the benefit. The causal policy scores higher than either while
treating 21% fewer events, since it picks its own operating point instead of being handed one.

**Averaging effects over a class is too coarse to act on.** The first version of the policy
estimated one uplift number per incident type. Most events in a class don't breach, so the
average understated the gain where action mattered, and a cost-aware rule then declined to act
at all: memory-leak restarts came out at 0.134 against a cost of 0.30, so it never restarted
anything. Conditioning on features instead moved the policy from 108.6 to 245.1.

**Naive PSI reads incidents as drift.** A handful of large incident spikes move mean latency
further than a fleet-wide 1.8x baseline shift does, so a monitor comparing live traffic against
a training reference fires on outages, which is exactly backwards: an outage wants remediation,
drift wants retraining. Measured, the naive version alerted on 13 clean windows and 14 drifted
ones, which is no discrimination at all. Building the reference from healthy ticks only and
requiring the median to move as well gets it to 8% against 100%.

**More temporal context didn't help.** Widening the feature space five times over (8 to 40
features, lags at 1, 2, 4 and 8 ticks) moved macro-F1 from 0.863 to 0.880 and made the hardest
class worse, dropping dependency_failure recall from 0.50 to 0.41. The limit isn't model
capacity. What separates those two classes isn't in the metrics at all, it's in the log line.

## Limitations

The telemetry is synthetic. Fingerprints for each incident type are hand-designed, so the
classifier's numbers describe this generator rather than any real fleet. Treat them as a
harness for comparing approaches, not as a claim about production accuracy.
[docs/data-model.md](docs/data-model.md) covers the event schema and what each incident type
looks like.

Latency figures come from a single M1 laptop and vary with machine load. The test suite only
guards against order-of-magnitude regressions; the real numbers come from `make loadtest` in a
quiet process.

## Development

```bash
make test    # 120 tests
make lint    # ruff
```

Warnings are errors. A scikit-learn feature-name warning went unnoticed for a while, so the
suite now fails on anything it hasn't been told to ignore by name.

## License

MIT. See [LICENSE](LICENSE).
