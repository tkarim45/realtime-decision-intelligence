# 03 — Setup (Apple M1, 8 GB)

## 0. Golden rules

- Never conda `base` / system Python (hook-enforced). Use `personal`; `claude` for scratch.
- Usable memory ≈ 4–5 GB after the OS. The deep encoder is deliberately tiny (1–20M) so
  scoring stays memory- and latency-cheap.
- Keep the LLM on the **slow path only** — it must never gate the hot path.

## 1. Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
pip install -r requirements.txt
```

`requirements.txt`: `fastapi uvicorn websockets redis scikit-learn lightgbm numpy pandas
statsmodels mlx mlx-lm llama-cpp-python anthropic[bedrock] mcp httpx prometheus-client
pytest ruff black python-dotenv`.

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

## 6. First run (once Milestone 0+ exists)

```bash
make stream       # broker + synthetic stream
make score        # scoring workers
make dashboard    # real-time UI
make test         # pytest (includes crash-recovery + feature-parity tests)
```

## 7. Troubleshooting

- **Hot-path p99 spikes** → LLM leaking into the hot path; move it fully async.
- **Encoder too slow/big** → shrink it (fewer params/layers), or drop to features-only for
  the hot path and use the encoder on the slow path.
- **Events lost on crash** → offsets committed before processing; commit *after*
  at-least-once processing instead.
- **Drift alert too noisy** → tune PSI window/threshold; verify it stays silent on a known
  clean stream before trusting it on a drifted one.
