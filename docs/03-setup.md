# 03 — Setup (Apple M1, 8 GB)

## 0. Golden rules

- Never conda `base` / system Python (hook-enforced). Use `personal`; `claude` for scratch.
- Usable memory ≈ 4–5 GB after the OS. The deep encoder is deliberately tiny (1–20M) so
  scoring stays memory- and latency-cheap.
- Keep the LLM on the **slow path only** — it must never gate the hot path.

## 1. Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
make install    # pip install -e ".[dev]"
make test
```

Dependencies live in `pyproject.toml`, not a `requirements.txt` — matching the sibling repos.
**M0 needs only `numpy`** (plus pytest/ruff as dev extras — no black; see `02-build-plan.md`).
Heavy deps are added to
`pyproject.toml` by the milestone that actually needs them, so a fresh clone stays installable
on an 8 GB M1 and nobody compiles `llama-cpp-python` to look at a JSONL stream:

| Milestone | Adds |
|---|---|
| M1 broker/features | — (stdlib + numpy) |
| M2 scoring | `scikit-learn`, `lightgbm`, `mlx`, `mlx-lm` |
| M3 reasoning | `llama-cpp-python`, `anthropic[bedrock]`, `mcp`, `httpx` |
| M4 decision | `pandas`, `statsmodels` |
| M5 ops | `prometheus-client` |
| M6 dashboard | `fastapi`, `uvicorn`, `websockets` (+ Next.js in `dashboard/`) |

## 2. Broker (no Kafka on 8 GB)

Reuse the file-backed Redis-Streams-contract broker from the `realtime-ml-pipeline` repo
(copy the module in). It gives consumer groups + at-least-once + crash recovery without a
cluster. Redis itself is optional (the file-backed variant needs no server); if you use
Redis, `brew install redis && redis-server` is fine on the M1.

## 3. Small deep encoder (MLX)

```bash
pip install mlx mlx-lm
```
Build a **1–20M-param sequence encoder** in MLX (small transformer or GRU over the event
window). Train on-device offline; export weights; load for inline scoring. Keep it small —
it must run inside the hot-path latency budget. Sanity-check inference latency before
wiring it into the stream.

## 4. Local + cloud LLM (slow path)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
# small GGUF for cheap/offline explanations
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models')"

set -a; source ~/.env; set +a   # Claude/Bedrock creds for hard cases
```
Route: local model for routine explanations, Claude for hard/ambiguous ones. The router
decision is cheap (rule/heuristic on flag severity + context size).

## 5. Latency discipline (the make-or-break)

- Measure hot-path p50/p99 **without** the LLM first; that's your SLO.
- Run the LLM **async, off the stream-critical path** (a separate worker/queue).
- Re-measure hot-path p99 **under LLM load** — it must not move. If it does, the LLM is
  leaking into the hot path; fix the async boundary.

## 6. First run

```bash
make summary      # ✅ M0 — ground-truth breakdown of the labeled stream
make stream       # ✅ M0 — labeled JSONL on stdout
make drift        # ✅ M0 — the M5 fixture: shifted baseline, zero incidents
make test         # ✅ M0 — 36 tests
make score        # M2 — scoring workers
make dashboard    # M6 — real-time UI
```

## 7. Troubleshooting

- **Hot-path p99 spikes** → LLM leaking into the hot path; move it fully async.
- **Encoder too slow/big** → shrink it (fewer params/layers), or drop to features-only for
  the hot path and use the encoder on the slow path.
- **Events lost on crash** → offsets committed before processing; commit *after*
  at-least-once processing instead.
- **Drift alert too noisy** → tune PSI window/threshold; verify it stays silent on a known
  clean stream before trusting it on a drifted one.
