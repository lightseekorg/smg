# τ²-bench nightly A/B — `scripts/tau2/`

**Track B** of the multi-turn parser-verification suite: run **τ²-bench** (`tau2-bench`) against two serving "arms" and diff pass^1 and pass^k. The companion **Track A** (offline, deterministic parser-conformance gate) lives in `crates/parser_conformance/`.

## The experiment

Two arms expose an identical OpenAI `/v1` endpoint. The same `tau2` CLI is pointed at each arm for the **agent LLM**; the **user-simulator is always the external `gpt-5.2`** (via `OPENAI_API_KEY`), identical on both arms. **Everything is held fixed except the agent frontend**, so any pass^k delta is attributable to the tokenization + tool/reasoning parsing layer — the number that argues for an engine adopting SMG's frontend.

| | baseline | candidate |
|---|---|---|
| arm | **pure vLLM** | **SMG → vLLM (gRPC)** |
| who renders the chat template + tokenizes | vLLM | SMG |
| who parses tool calls / reasoning | vLLM (`--tool-call-parser qwen3_xml`) | SMG (`--tool-call-parser qwen_xml`) |
| user-simulator | **gpt-5.2 (fixed, identical)** | **gpt-5.2 (fixed, identical)** |
| model · engine · checkpoint · sampling | **identical** | **identical** |

**Why native FC mode puts SMG's parser on the critical path.** τ²-bench's agent instructs the model via native `tools` and reads `response.choices[].message.tool_calls` — the *server's parsed output*. This means SMG's Rust tool-call parser (or vLLM's) is on the critical path for every agent step, and any parsing error shows up directly as a failed task. The user-sim side is purely the OpenAI API and is identical on both arms.

## Files

| file | what |
|---|---|
| `launch_arms.sh` | bring up one arm (`a` = pure vLLM, `b` = vLLM gRPC worker + SMG gateway); prints its base_url on stdout; `stop` tears down all arms via pidfiles. Fully env-parameterised. |
| `run_ab.py` | point `tau2 run` at both arms (one domain at a time), read back `results.json`, compute pass^1 and pass^k per domain, emit a markdown + JSON comparison table, and apply a regression gate. Arms must already be serving. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: install tau2-bench (uv required); and ninja in the vLLM env (see Gotchas)
git clone https://github.com/sierra-research/tau2-bench ~/tau/tau2-bench
cd ~/tau/tau2-bench && uv sync
~/vllm-env/bin/pip install ninja          # then ensure ~/vllm-env/bin is on PATH

# 1) bring up both arms (here: Qwen3.6-27B, TP=2, one arm per GPU pair)
export TAU2_MODEL=Qwen/Qwen3.6-27B VLLM_BIN=~/vllm-env/bin/vllm \
       VLLM_PYTHON=~/vllm-env/bin/python SMG_LAUNCH="$HOME/smg/target/ci/smg launch" \
       TAU2_TP=2 TAU2_MAX_MODEL_LEN=16384 PATH=~/vllm-env/bin:$PATH
A_URL=$(TAU2_GPU=0,1 TAU2_VLLM_TOOL_PARSER=qwen3_xml TAU2_VLLM_REASONING_PARSER=qwen3 bash launch_arms.sh a)
B_URL=$(TAU2_GPU=2,3 TAU2_SMG_TOOL_PARSER=qwen_xml   TAU2_SMG_REASONING_PARSER=qwen3  bash launch_arms.sh b)

# 2) set your OpenAI key so the user-sim (gpt-5.2) can reach the API
export OPENAI_API_KEY=sk-...

# 3) run the A/B
~/tau/tau2-bench/.venv/bin/python run_ab.py \
    --baseline  "vllm=$A_URL" \
    --candidate "smg=$B_URL" \
    --agent-model Qwen/Qwen3.6-27B \
    --user-llm gpt-5.2 \
    --tau2 ~/tau/tau2-bench/.venv/bin/tau2 \
    --data-dir ~/tau/tau2-bench/data \
    --domains retail,airline,telecom \
    --num-trials 2 \
    --out ~/tau2_ab.md --json-out ~/tau2_ab.json

# 4) teardown
bash launch_arms.sh stop
```

Key env knobs for `launch_arms.sh`: `TAU2_GPU` (CUDA_VISIBLE_DEVICES, e.g. `0,1`), `TAU2_TP` (tensor-parallel size — match the GPU count), `TAU2_MAX_MODEL_LEN`, `TAU2_{VLLM,SMG}_{TOOL,REASONING}_PARSER`, and `TAU2_VLLM_EXTRA` for extra vLLM flags.

`run_ab.py` exits non-zero if the candidate's overall pass^k drops more than `--tolerance` (default 2pp) below the baseline.

## Agent-vs-user routing (no config file)

τ²-bench routes the agent and user LLMs **per-call** — no LiteLLM config file is needed. `run_ab.py` passes the arm's URL directly via `--agent-llm-args`:

```
--agent-llm openai/<model>
--agent-llm-args '{"api_base": "<arm>/v1", "api_key": "smg-local"}'
--user-llm gpt-5.2
```

The user-simulator (`gpt-5.2`) is reached via the standard OpenAI API using `OPENAI_API_KEY`. The agent is routed to whichever arm is under test. `run_ab.py` constructs and passes these arguments automatically — you only need to supply `--agent-model`, `--user-llm`, and `--baseline`/`--candidate`.

Results land at:

```
<DATA_DIR>/simulations/ab_<arm_name>_<domain>/results.json
```

where `DATA_DIR` defaults to `tau2-bench/data`. Pass `--data-dir` to override.

## Per-model parser flags (the nightly matrix)

| model (matrix leg) | pure-vLLM `--tool-call-parser` / `--reasoning-parser` | SMG `--tool-call-parser` / `--reasoning-parser` |
|---|---|---|
| Qwen3.6-27B (`qwen3.6`) | `qwen3_xml` / `qwen3` | `qwen_xml` / `qwen3` |

> The mid-2026 SKU ids and vLLM parser names may shift; confirm against the installed
> vLLM build: `vllm serve --help | grep -A40 tool-call-parser`.

## Gotchas discovered while bringing this up (read before debugging)

- **Install `ninja` in the vLLM env (do NOT reach for `--enforce-eager`).** vLLM's torch.compile / CUDA-graph path shells out to `ninja` to build kernels; if it's missing the engine dies with `No such file or directory: 'ninja'`. `--enforce-eager` only *hides* this by skipping compilation (slower). Real fix: `pip install ninja` in the vLLM env **and put its bin on `PATH`** (vLLM execs `ninja` by name).
- **Cap the context.** Qwen3 models default to a very large `max_model_len` → OOM on init. Pass `--max-model-len 16384` (the launch helper defaults to this); use the **same** value on both arms to keep conditions identical.
- **SMG auto model→parser mapping lags new SKUs.** SMG's factory doesn't yet map `Qwen3.6*` (it falls back to the JSON `qwen` parser, wrong for the XML format), so pass `--tool-call-parser qwen_xml` explicitly via `TAU2_SMG_TOOL_PARSER=qwen_xml`. Adding a `Qwen3.6*`→`qwen_xml` mapping to `crates/tool_parser` is a good follow-up.
- **`tau2-bench` install: use `uv sync`.** Clone the repo and run `uv sync` inside it — this resolves all deps (including any optional extras) into `.venv`. The executable is `tau2-bench/.venv/bin/tau2`; pass its full path to `--tau2`.
- **Results path.** Results land at `<DATA_DIR>/simulations/<save-to>/results.json`. `run_ab.py` sets `save-to` to `ab_<arm>_<domain>` automatically. If a tau2 run exits non-zero (e.g. API error), `run_ab.py` warns and attempts to read whatever partial results exist at that path.
- **Exclude `banking_knowledge`.** The `banking_knowledge` domain requires OPENAI_API_KEY-authenticated embeddings to set up its retrieval fixture — it will fail without dedicated infra. Omit it from `--domains`; the default (`retail,airline,telecom`) already excludes it.
- **`OPENAI_API_KEY` is required.** The user-simulator (`gpt-5.2`) goes through the OpenAI API. If the key is unset or invalid, every user turn fails and pass^k collapses to 0 on both arms equally — a bad signal to debug.

## Validation status

> **Placeholder — numbers will be filled in after the H100 validation run (Task 9).**
>
> The table below is reserved for end-to-end validation results on a dev H100 box with
> `Qwen/Qwen3.6-27B` at TP=2. No benchmark numbers have been fabricated here.

| domain | pure vLLM pass^1 | SMG → vLLM gRPC pass^1 | Δ | pure vLLM pass^k | SMG → vLLM gRPC pass^k | Δ |
|---|---|---|---|---|---|---|
| retail | — | — | — | — | — | — |
| airline | — | — | — | — | — | — |
| telecom | — | — | — | — | — | — |
| **overall** | — | — | — | — | — | — |
