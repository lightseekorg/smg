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
| `register_bfcl_model.py` | register a model that bfcl-eval doesn't ship a handler for yet (new SKUs), by cloning an existing FC entry. Idempotent. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: a venv with bfcl-eval (+ soundfile), and ninja in the *vLLM* env
#    for torch.compile / CUDA-graph kernel builds (see Gotchas).
python -m venv ~/bfcl-env && ~/bfcl-env/bin/pip install bfcl-eval soundfile
~/vllm-env/bin/pip install ninja          # then ensure ~/vllm-env/bin is on PATH

# teach bfcl about a model it doesn't ship a handler for yet (new SKUs)
~/bfcl-env/bin/python register_bfcl_model.py --model-id Qwen/Qwen3.6-27B

# 1) bring up both arms (here: Qwen3.6-27B, TP=2, one arm per GPU pair)
export BFCL_MODEL=Qwen/Qwen3.6-27B VLLM_BIN=~/vllm-env/bin/vllm \
       VLLM_PYTHON=~/vllm-env/bin/python SMG_LAUNCH="$HOME/smg/target/ci/smg launch" \
       BFCL_TP=2 BFCL_MAX_MODEL_LEN=16384 PATH=~/vllm-env/bin:$PATH
A_URL=$(BFCL_GPU=0,1 BFCL_VLLM_TOOL_PARSER=qwen3_xml BFCL_VLLM_REASONING_PARSER=qwen3 bash launch_arm.sh a)
B_URL=$(BFCL_GPU=2,3 BFCL_SMG_TOOL_PARSER=qwen_xml  BFCL_SMG_REASONING_PARSER=qwen3 bash launch_arm.sh b)

# 2) run the official A/B
~/bfcl-env/bin/python run_ab.py \
    --baseline  "vllm=$A_URL" \
    --candidate "smg=$B_URL" \
    --bfcl-model Qwen/Qwen3.6-27B-FC \
    --categories simple_python,multiple,parallel,irrelevance \
    --bfcl ~/bfcl-env/bin/bfcl --project-root ~/bfcl_ab \
    --out ~/bfcl_ab.md --json-out ~/bfcl_ab.json

# 3) teardown
bash launch_arm.sh stop
```

Key env knobs for `launch_arm.sh`: `BFCL_GPU` (CUDA_VISIBLE_DEVICES, e.g. `0,1`),
`BFCL_TP` (tensor-parallel size — match the GPU count), `BFCL_MAX_MODEL_LEN`,
`BFCL_{VLLM,SMG}_{TOOL,REASONING}_PARSER`, and `BFCL_VLLM_EXTRA` for extra vLLM
flags.

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
- **Install `ninja` in the vLLM env (do NOT reach for `--enforce-eager`).**
  vLLM's torch.compile / CUDA-graph path shells out to `ninja` to build kernels
  (required for newer archs like Qwen3.6's `qwen3_5`); if it's missing the
  engine dies with `No such file or directory: 'ninja'`. `--enforce-eager` only
  *hides* this by skipping compilation (slower). Real fix:
  `pip install ninja` in the vLLM env **and put its bin on `PATH`** (vLLM execs
  `ninja` by name) — then run with CUDA graphs, no `--enforce-eager`.
- **Don't force HF offline.** With the model cached, bfcl runs fine online
  (~7 req/s measured); a one-off slow run is usually a transient HF hiccup, not
  a systematic per-request throttle. `run_ab.py` does **not** set
  `HF_HUB_OFFLINE`; set it yourself only for air-gapped boxes. No HF token is
  needed for public models.
- **New models need a bfcl handler.** bfcl-eval pins a fixed model list; a
  brand-new SKU (e.g. `Qwen/Qwen3.6-27B`) isn't in it, so
  `bfcl generate --model <id>-FC` fails with "Unknown model_name". Run
  `register_bfcl_model.py --model-id <id>` first.
- **SMG auto model→parser mapping lags new SKUs.** SMG's factory doesn't yet map
  `Qwen3.6*` (it falls back to the JSON `qwen` parser, wrong for the XML format),
  so pass `--tool-call-parser qwen_xml` explicitly. Adding a `Qwen3.6*`→`qwen_xml`
  mapping to `crates/tool_parser` is a good follow-up.
- **Use the `-FC` handler.** `Qwen/Qwen3-4B-Instruct-2507-FC`, not the bare name
  (which is prompt mode and bypasses the server parser).
- **`bfcl generate --skip-server-setup`** points at `LOCAL_SERVER_ENDPOINT` /
  `LOCAL_SERVER_PORT`. (Custom full base_urls behind a proxy are still rigid —
  gorilla issue #1280.)
- **`EngineDeadError` on startup is usually the missing-`ninja` issue above**
  (the engine dies around CUDA-graph capture / first compile). Install `ninja`
  and put it on `PATH` rather than falling back to `--enforce-eager`.

## Validation status — ran end-to-end ✅

Run on a dev H100 box, **`Qwen/Qwen3.6-27B` at TP=2** (one arm per GPU pair),
the **full `non_live` set** (1390 cases), FC mode, temp 0.001 — **no
`--enforce-eager`, online HF** (the clean config, after `pip install ninja`):

| category | pure vLLM (`qwen3_xml`) | SMG → vLLM gRPC (`qwen_xml`) | Δ |
|---|---|---|---|
| simple_python | 95.00 | 94.75 | −0.25 |
| simple_java | 64.00 | 64.00 | 0.00 |
| simple_javascript | 72.00 | 72.00 | 0.00 |
| multiple | 91.00 | 91.50 | +0.50 |
| parallel | 89.50 | 90.00 | +0.50 |
| parallel_multiple | 91.00 | 91.50 | +0.50 |
| irrelevance | 84.58 | 84.58 | 0.00 |
| **overall (unweighted)** | **83.87** | **84.05** | **+0.18** |

So SMG's Rust frontend is **at parity** with vLLM's native parser across the
full non-live set — marginally ahead overall (+0.18pp), and never worse than
−0.25pp on any category. Both arms reasoning-parse `<think>` into
`reasoning_content` and emit native `tool_calls` (FC confirmed end to end). The
low java/js numbers are the *model's* non-Python ability (identical on both
arms) — confirming the A/B isolates the frontend, not model quality. (An earlier
`simple_python`-only run on Qwen3-4B gave 95.50 vs 95.25, same parity story.)

Scope note: the table above is the `non_live` slice. The **nightly** runs the
broader reproducible set — **`non_live` + `live` + `multi_turn`** (17 categories;
`live` is real-user data, `multi_turn` is state-based simulation — both static,
no internet). It excludes the agentic/executable categories (`web_search_*`,
`memory_*`, `exec_*`) that need web-search/memory/sandbox infra. **PRs** that
touch the pipeline run only a quick non-live sanity subset
(`simple_python,irrelevance`); per-PR correctness is the cheap CPU Track-A
parser-conformance gate (`crates/parser_conformance`). Scale to multiple runs ×
the five target models for tight confidence intervals.

**On the two "fixes" from the first cut:** neither `--enforce-eager` nor
`HF_HUB_OFFLINE` is actually needed. The crash was a missing `ninja` (install it
in the vLLM env + PATH → CUDA graphs work); the slowness was a transient
unauthenticated-HF hiccup (cached + online runs at ~7 req/s). Both band-aids
have been removed from the defaults.
