# 00 — What this project is

## Problem

Most ML systems predict a number and stop. Real value comes from **acting on the
prediction, in real time, and knowing the action worked.** That requires four things
almost no portfolio project has together: (1) low-latency streaming ML, (2) trustworthy
uncertainty, (3) a decision layer that models the *incremental* effect of an intervention
(not just risk), and (4) proof — a valid experiment — that interventions actually help.
Wrap GenAI around it for human-readable explanations and remediation, keep it alive under
failure and drift, and you have a production decision-intelligence system.

## What we build

A live pipeline for one domain (fraud / AIOps / vitals) that:

1. **Ingests a durable event stream** with at-least-once delivery and crash recovery.
2. **Scores in real time** — classical hot path (sub-ms) + a small on-device deep encoder
   + an anomaly detector, calibrated, with **conformal prediction** giving each decision a
   guaranteed-coverage confidence set.
3. **Explains and remediates** — when an event is flagged, an LLM agent pulls context
   (feature store, logs, RAG), writes a grounded explanation + recommended action, and can
   call tools to remediate. Explanations are scored for groundedness.
4. **Decides interventions causally** — uplift/meta-learners estimate the incremental
   effect of each action; a policy picks treat/don't-treat.
5. **Proves it worked** — outcomes feed a peeking-safe experimentation engine
   (Bayesian + always-valid sequential tests).
6. **Stays alive** — PSI drift alerts, retrain trigger, canary + shadow deploys, instant
   rollback, load shedding.

All on an Apple M1 (8 GB): small on-device models for scoring, routed local+cloud LLM for
reasoning.

## Goals

- Meet stated **throughput + p99 latency SLOs** on a live stream.
- **Zero event loss** across a mid-stream crash.
- A **drift-triggered retrain** that fires on drifted data and stays quiet on clean data.
- A **validated causal lift** from interventions (Qini / policy value).
- Keep GenAI on the *slow path only* so it never blows the latency budget.

## Non-goals

- Not a big-data infra demo — no Kafka cluster; a file-backed streams broker on the laptop.
- Not a model-training project — the deep encoder is deliberately tiny (that's the sibling
  `ondevice-model-lifecycle` repo's territory).
- Not a chatbot — the LLM is a scoped explain/remediate component, not the product.

## Success criteria (definition of done)

| # | Criterion | Evidence |
|---|---|---|
| S1 | Durable stream with at-least-once + proven crash recovery (0 lost) | recovery test log |
| S2 | Real-time scoring meets throughput + p99 SLO | load-test report |
| S3 | Conformal sets hit target coverage on the stream | coverage plot |
| S4 | LLM agent produces grounded explanation + can remediate via a tool | trace |
| S5 | Uplift policy picks interventions; validated by Qini/policy value | uplift eval |
| S6 | Experiment engine measures intervention lift, peeking-safe | experiment report |
| S7 | PSI drift alert fires on drifted stream, silent on clean stream | drift test |
| S8 | Everything visible on a real-time dashboard | screen recording |

## The honest research questions

- Can the LLM stay on the slow path without breaking the end-to-end latency budget?
- Is the online feature code truly train=serve (the leak `feature-store` exists to prevent)?
- Does the causal policy beat a naive risk-threshold policy on *incremental* outcome?
