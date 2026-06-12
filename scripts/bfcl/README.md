# BFCL nightly A/B — `scripts/bfcl/`

**Track B** of the parser-verification proposal
(`docs/proposals/2026-06-10-bfcl-nightly-parser-verification.md`): run the
**official** Berkeley Function Calling Leaderboard (`bfcl-eval`) against two
serving "arms" and diff the scores. The companion **Track A** (offline,
deterministic parser-conformance gate) lives in `crates/parser_conformance/`.

## The experiment

Two arms expose an identical OpenAI `/v1` endpoint. The same official `bfcl`
CLI (FC mode) is pointed at each; **everything is held fixed except the
frontend**, so any score delta is attributable to the tokenization + parsing
layer — the number that argues for an engine adopting SMG's frontend.

| | baseline | candidate |
|---|---|---|
| arm | **pure vLLM** | **SMG → vLLM (gRPC)** |
| who renders the chat template + tokenizes | vLLM | SMG |
| who parses tool calls / reasoning | vLLM (`--tool-call-parser hermes`) | SMG (`--tool-call-parser qwen`) |
| model · engine · checkpoint · sampling | **identical** | **identical** |

**Why FC mode is mandatory.** BFCL's `…-FC` model handlers send the native
`tools` param and score `response.choices[].message.tool_calls` — i.e. the
*server's parsed output*. The non-FC (prompt) handlers format tools into the
prompt and parse the text themselves, bypassing the server parser. Only FC mode
puts SMG's / vLLM's parser on the critical path, so the driver always uses the
`-FC` handler (e.g. `Qwen/Qwen3-4B-Instruct-2507-FC`).

## Files

| file | what |
|---|---|
| `launch_arm.sh` | bring up one arm (`a` = pure vLLM, `b` = vLLM-gRPC + SMG); prints its base_url; `stop` tears down via pidfiles. Fully env-parameterised. |
| `run_ab.py` | point official `bfcl generate`+`evaluate` (FC mode) at both arms, parse per-category accuracy, emit a markdown + JSON comparison table, and a regression gate. Arms must already be serving. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: a venv with bfcl-eval (+ soundfile, see Gotchas)
python -m venv ~/bfcl-env && ~/bfcl-env/bin/pip install bfcl-eval soundfile

# 1) bring up both arms (small model fits one H100 at mem-util ~0.4 each;
#    or pin each arm to its own GPU via BFCL_GPU)
export BFCL_MODEL=Qwen/Qwen3-4B-Instruct-2507 VLLM_BIN=~/vllm-env/bin/vllm \
       VLLM_PYTHON=~/vllm-env/bin/python SMG_LAUNCH="$HOME/smg/target/ci/smg launch"
BFCL_GPU=0 BFCL_ARM_A_PORT=31199 A_URL=$(bash launch_arm.sh a)
BFCL_GPU=1 BFCL_ARM_B_GRPC_PORT=50081 BFCL_ARM_B_GW_PORT=31200 B_URL=$(bash launch_arm.sh b)

# 2) run the official A/B
~/bfcl-env/bin/python run_ab.py \
    --baseline  "vllm=$A_URL" \
    --candidate "smg=$B_URL" \
    --bfcl-model Qwen/Qwen3-4B-Instruct-2507-FC \
    --categories simple_python,multiple,parallel,irrelevance \
    --bfcl ~/bfcl-env/bin/bfcl --project-root ~/bfcl_ab \
    --out ~/bfcl_ab.md --json-out ~/bfcl_ab.json

# 3) teardown
bash launch_arm.sh stop
```

`run_ab.py` exits non-zero if the candidate's overall accuracy drops more than
`--tolerance` (default 2pp) below the baseline.

## Per-model parser flags (vLLM ~0.22.x)

| model family | pure-vLLM `--tool-call-parser` / `--reasoning-parser` | SMG `--tool-call-parser` / `--reasoning-parser` |
|---|---|---|
| Qwen3 dense Instruct (e.g. Qwen3-4B-Instruct-2507) | `hermes` / — | `qwen` / — |
| Qwen3 thinking | `hermes` / `qwen3` | `qwen` / `qwen3` |
| Qwen3-Coder / 3.5 / 3.6 | `qwen3_coder` / `qwen3` | `qwen_xml` / `qwen3` |
| DeepSeek V3/R1 | `deepseek_v3` / `deepseek_r1` | `deepseek` / `deepseek_r1` |
| DeepSeek V3.1/V3.2/V4 | `deepseek_v31` / `deepseek_v3` | `deepseek31` / `deepseek_v4` (DSML) / … |
| Kimi K2 | `kimi_k2` / — | `kimik2` / `kimi` |
| MiniMax M2 | `minimax_m2` / `minimax_m2` | `minimax_m2` / `minimax` |

> The mid-2026 SKUs (DeepSeek V4, Kimi K2.6, Qwen3.6, MiniMax M2.7) may use newer
> parser names; confirm against the installed vLLM build:
> `vllm serve --help | grep -A40 tool-call-parser`.

## Gotchas discovered while bringing this up (read before debugging)

- **`bfcl-eval` needs `soundfile`.** Its Qwen handler imports `qwen_agent` →
  `soundfile`; without it `bfcl --help` itself crashes. `pip install soundfile`.
- **Cap the context.** Qwen3-4B defaults to a 256K `max_model_len` → ~36 GiB KV
  cache → engine init OOM. Pass `--max-model-len 16384` (the launch helper
  defaults to this); use the **same** value on both arms.
- **Force HF offline once the model is cached.** Without `HF_HUB_OFFLINE=1`
  (set by `run_ab.py`), bfcl round-trips to the HF Hub per request and crawls
  (and rate-limits without a token).
- **Use the `-FC` handler.** `Qwen/Qwen3-4B-Instruct-2507-FC`, not the bare name
  (which is prompt mode and bypasses the server parser).
- **`bfcl generate --skip-server-setup`** points at `LOCAL_SERVER_ENDPOINT` /
  `LOCAL_SERVER_PORT`. (Custom full base_urls behind a proxy are still rigid —
  gorilla issue #1280.)
- **vLLM engine stability under load.** On a shared/contended GPU, vLLM 0.22.1's
  V1 engine has thrown `EngineDeadError` under concurrent bfcl load. Prefer a
  dedicated GPU per arm and a modest `--num-threads`.

## Validation status — ran end-to-end ✅

Brought up on a dev H100 box against `Qwen/Qwen3-4B-Instruct-2507`, BFCL
`simple_python` (400 cases), FC mode, temp 0.001, both arms on the same GPU
(sequentially), `--enforce-eager` + `HF_HUB_OFFLINE=1`:

| arm | accuracy |
|---|---|
| pure vLLM (`--tool-call-parser hermes`) | **95.50%** (382/400) |
| SMG → vLLM gRPC (`--tool-call-parser qwen`) | **95.25%** (381/400) |
| **Δ (SMG − vLLM)** | **−0.25pp** (1 case — parity, within noise) |

So on this slice SMG's Rust frontend is **at parity** with vLLM's native
parser. Both arms served native `tool_calls` (FC mode confirmed end to end), and
`run_ab.py` produced the table above. Scale to more categories + multiple runs
(and the five target models) to characterise the delta with confidence; that's
what `nightly-bfcl.yml` is for.

Two things were needed to run cleanly on a shared/contended GPU (both folded
into the tooling): `BFCL_VLLM_EXTRA=--enforce-eager` (vLLM's V1 engine threw
`EngineDeadError` under load with CUDA graphs) and `HF_HUB_OFFLINE=1` (set by
`run_ab.py`; otherwise per-request HF Hub round-trips throttle the run to a
crawl).
