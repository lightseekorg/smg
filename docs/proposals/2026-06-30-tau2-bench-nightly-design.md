# tau2-bench nightly A/B — parser verification (Track B, multi-turn)

**Status:** design — approved in brainstorming, pending spec review.
**Date:** 2026-06-30.
**Audience:** SMG router + parser team.
**Relation:** extends `docs/proposals/2026-06-10-bfcl-nightly-parser-verification.md`
(the BFCL nightly). This is the **τ²-bench** sibling of that pipeline: same
"hold everything fixed except SMG's frontend" A/B, but over multi-turn,
LLM-simulated-user, DB-state-reward dialogues instead of single-turn AST matching.

---

## TL;DR

Mirror `nightly-bfcl.yml` for [τ²-bench](https://github.com/sierra-research/tau2-bench):
a live, end-to-end **A/B** that holds model + engine + checkpoint + sampling +
user-simulator **fixed** and varies only SMG's frontend (chat template +
tokenization + tool/reasoning parsing). τ²-bench's agent calls tools via native
FC, so SMG parses the agent's `tool_calls` — the same critical path BFCL FC mode
exercises, but now across **multi-turn** tool-agent-user conversations graded by
**final database-state reward** and **pass^k** reliability.

- **Arm A (baseline):** pure vLLM (vLLM owns template+tokenize+parse).
- **Arm B (candidate):** SMG → vLLM gRPC (SMG owns template+tokenize+parse).
- **User-simulator:** **GPT-5.2** (pinned, temp 0) via the OpenAI API — the same
  user-sim the public τ²-bench leaderboard now recommends, identical across both
  arms.
- Any score Δ between arms is attributable to SMG's parser. **Informational /
  non-blocking nightly signal — never a merge gate** (this is "Track B").

First deliverable: validate the whole thing end-to-end on the H100 box with
Qwen3.6-27B (exactly how BFCL was brought up), capture the parity numbers, *then*
commit the scripts + workflow + docs.

---

## 1. Goal & premise

SMG sits in front of inference engines over gRPC in raw-token mode; SMG owns the
chat template + tokenization and **owns output parsing**
(`crates/tool_parser`, `crates/reasoning_parser`). That parsing layer is the
"frontend" under test. The BFCL nightly already verifies it on single-turn FC.
τ²-bench adds the multi-turn dimension that BFCL's `multi_turn` categories only
weakly cover: a real **LLM user simulator**, **stateful tools** mutating a
domain database, and a **reward = did the final DB state + required
communication match the goal**. This stresses the parser across long
tool-calling trajectories where a single mis-parsed call derails the whole task.

**Premise check (holds):** τ²-bench runs the agent in native function-calling
mode — it sends the OpenAI `tools` param and reads `message.tool_calls` from the
server response. So SMG's parser is squarely on the critical path, exactly as in
BFCL FC mode. The user simulator produces *plain text only* (it never calls
tools), so the user-sim's tool-parsing is irrelevant — which is why the user-sim
can be a different model from the agent.

## 2. The A/B — two self-hosted arms + a fixed external user-sim

| | Arm A (baseline) | Arm B (candidate) | User-simulator |
|---|---|---|---|
| stack | pure vLLM OpenAI | SMG → vLLM gRPC | OpenAI API |
| model | Qwen3.6-27B | Qwen3.6-27B | **gpt-5.2** (pinned) |
| owns template/tokenize/parse | vLLM | **SMG (under test)** | n/a (text only) |
| role | agent | agent | user, **identical for both arms** |
| sampling | temp 0 (fixed) | temp 0 (fixed) | temp 0 (fixed) |

The user-sim is one fixed GPT-5.2 endpoint that **both** arms' `--user-llm` point
at, so the *only* difference between an Arm-A run and an Arm-B run is the agent's
frontend.

### Why GPT-5.2 as the user-sim

- The user-sim only needs to be **pinned + identical across arms**; it is part of
  the fixed environment, not the thing under test.
- A strong, stable instruction-follower at temp 0 minimizes conversation drift →
  tighter parser Δ, fewer trials needed. A self-hosted 27B is a weaker, noisier
  simulator.
- The public τ²-bench leaderboard now recommends **gpt-5.2** as the user-sim
  (the original paper used `gpt-4.1-2025-04-14`). Matching it keeps our absolute
  numbers **comparable to the public board**, not just internally consistent.
- Using an external API for the user-sim **frees all GPUs for the two agent
  arms** (back to BFCL's exact layout) — no third serving process.

### GPU layout

- **Validation (8×H100 `moirai-exp-2`):** Arm A → GPUs 0,1; Arm B worker →
  GPUs 2,3 (TP=2 each). Plenty of headroom (4–7 spare; can bump TP for speed).
- **CI (`4-gpu-h100`, same runner class as BFCL):** Arm A TP=2 (0,1), Arm B
  worker TP=2 (2,3). TP is a workflow input.

## 3. How τ²-bench routes agent vs user to different endpoints (LiteLLM)

τ²-bench uses **LiteLLM**. The agent and the user-sim both speak the OpenAI
protocol but live at **different base URLs** (agent → local arm; user →
`api.openai.com`). A single global `OPENAI_API_BASE` cannot express that, so we
need **per-model `base_url` routing**. Plan, in order of preference:

1. **LiteLLM `model_list` config** (a `model_list:` YAML) with one entry per
   model, each carrying its own `litellm_params.api_base` + `api_key`:
   ```yaml
   model_list:
     - model_name: agent-arm          # passed to --agent-llm
       litellm_params:
         model: openai/Qwen/Qwen3.6-27B
         api_base: http://127.0.0.1:<ARM_PORT>/v1
         api_key: dummy               # vLLM/SMG ignore it
     - model_name: user-sim           # passed to --user-llm
       litellm_params:
         model: gpt-5.2
         api_key: os.environ/OPENAI_API_KEY
   ```
   Then `tau2 run --agent-llm agent-arm --user-llm user-sim ...`.
2. If τ²-bench does not consume a LiteLLM config directly, a **small shim** that
   calls `litellm.register_model` / sets per-model `api_base` before invoking
   τ²-bench, or τ²-bench's own model-registration mechanism if it exposes
   `base_url`.

**This is the #1 thing validation must prove.** Documented fallback if per-model
`base_url` proves infeasible: degrade the user-sim to *per-arm self-contained*
(each arm serves its own user-sim) — cheaper isolation, recorded in the README as
a known caveat. (Resolving this is why we validate before committing scripts.)

## 4. Files (mirror `scripts/bfcl/`)

`scripts/tau2/`:

| file | what |
|---|---|
| `launch_arms.sh` | bring up `a` (pure vLLM) / `b` (vLLM-gRPC + SMG) / `stop`; pidfile-tracked, env-parameterized, free-port by default. Same shape as `scripts/bfcl/launch_arm.sh` (no user-sim process — that's the OpenAI API). |
| `run_ab.py` | for each arm, run `tau2 run` over the configured domains × `--num-trials`, agent→arm, user→gpt-5.2; parse τ²-bench results (reward / pass^k per domain); emit a markdown + JSON comparison table (candidate − baseline) + a tolerance-based regression annotation (informational). Mirrors `scripts/bfcl/run_ab.py`. |
| `litellm_tau2.yaml` | the agent/user → base_url + api_key mapping from §3 (or a shim if a config file isn't honored). |
| `README.md` | quick start, per-model parser flags, gotchas, and a **Validation status** table with the Qwen3.6 numbers produced during bring-up (mirrors the BFCL README). |

A `register_tau2_model.py` is likely **not** needed (LiteLLM accepts arbitrary
`openai/<name>` without a pinned model list, unlike `bfcl-eval`) — confirm in
validation.

## 5. `.github/workflows/nightly-tau2.yml`

Modeled on `nightly-bfcl.yml`:

- **`build-wheel`** job (cached smg wheel) → **`tau2-ab`** job on `4-gpu-h100`.
- **Schedule:** cron `23 10 * * *` UTC — off-round and distinct from existing
  nightlies (bfcl `37 9`, benchmark `0 8`, mlx `17 9`).
- **PR trigger:** only on `scripts/tau2/**` + the workflow file → quick sanity
  subset (`--domain retail --num-trials 1 --num-tasks <small>`) for pipeline
  confidence, not a statistical result.
- **`workflow_dispatch` inputs:** `model`, `domains`, `num_trials`, `num_tasks`,
  `tp`, `agent_tool_parser` (vLLM + SMG), `reasoning_parser`, `user_sim_model`.
- **Defaults (nightly):** domains = `retail,airline,telecom`, `num_trials=2`
  (pass^2), tolerance `0.02` (informational), user-sim = `gpt-5.2`.
- **Secrets:** `OPENAI_API_KEY` (user-sim). No Anthropic key needed.
- **Artifacts:** `tau2_ab.{md,json}` + per-arm τ²-bench transcripts/results
  (for debugging, like BFCL captures both arms' transcripts).
- **Timeout:** generous (multi-turn is slow). Default plan: run full task sets at
  360 min; if validation throughput shows that overruns, cap `num_tasks` per
  domain to a documented default and expand via dispatch. **The concrete cap is
  finalized from validation timing.**

## 6. Metric

τ²-bench grades each task by **reward** (final DB-state hash match + required
communicate/action checks) and reports **pass^k** (all k trials pass) per domain.

- Report per domain: **pass^1** (avg reward) and **pass^2**.
- **Overall** = unweighted mean across domains (matches BFCL's reporting).
- **Δ** = candidate − baseline per domain and overall.
- Flag a "regression" only if candidate drops **> tolerance** below baseline —
  an annotation, never a merge gate.

## 7. Validation sequence on the H100 (first deliverable)

1. `~/tau`: install `uv`; `git clone` τ²-bench; `uv sync` (Python 3.12 ✓ on box).
2. Bring up Arm A + Arm B with Qwen3.6-27B (already in the box's HF cache).
3. **Solve §3** — per-model `base_url` routing (agent→arm, user→gpt-5.2).
4. Smoke each arm: `tau2 run --domain retail --agent-llm agent-arm --user-llm
   user-sim --num-trials 1 --num-tasks 5`; confirm server-side `tool_calls` (FC)
   and `<think>` reasoning parse on both arms.
5. Small A/B (retail, k=1, ~20 tasks) → confirm parity; capture README numbers;
   measure OpenAI user-sim token spend + per-task wall-clock (sets the nightly
   `num_tasks`/timeout).
6. Write the scripts + workflow + docs to match the validated commands.

## 8. Per-model parser flags (Qwen3.6, from the BFCL bring-up)

| | pure-vLLM (Arm A) | SMG (Arm B) |
|---|---|---|
| `--tool-call-parser` | `qwen3_xml` | `qwen_xml` |
| `--reasoning-parser` | `qwen3` | `qwen3` |

SMG's model→parser auto-map may lag `Qwen3.6*` (falls back to JSON `qwen`,
wrong for the XML format) → pass `--tool-call-parser qwen_xml` explicitly. Adding
a `Qwen3.6*`→`qwen_xml` mapping to `crates/tool_parser` is a good follow-up.

## 9. Open risks (resolved during validation)

- **LiteLLM per-model `base_url`** (agent vs user) — §3; fallback documented.
- **τ²-bench results file layout** → exact `run_ab.py` parse logic.
- **Runtime vs timeout** for 3 domains × k=2 → may cap `num_tasks`; measure first.
- **External dependencies:** confirm `retail`/`airline`/`telecom` run fully
  locally (no external services beyond the user-sim API). We exclude
  `banking_knowledge` (needs a RAG pipeline).
- **OpenAI user-sim spend** per nightly — measure in step 5 and report; revisit
  the model if it's heavier than expected.
- **CI runner GPU count** — target `4-gpu-h100` (known to exist from BFCL),
  2 arms × TP=2.
- **Reproducibility** — pin the exact `gpt-5.2` snapshot string LiteLLM resolves
  to; pin Qwen3.6 checkpoint; temp 0 on all three.

## 10. Decisions recorded

- A/B isolates SMG's parser; **informational, never a merge gate** (Track B).
- **User-sim = GPT-5.2**, pinned, temp 0, shared/identical across arms
  (public-leaderboard comparable). Trade-off accepted: re-introduces per-model
  `base_url` routing (§3) and adds recurring OpenAI spend.
- **Nightly scope = retail + airline + telecom, k=2.**
- Two self-hosted arms only; user-sim is the external API (no third GPU process).
- Validate on the H100 with Qwen3.6 first; commit scripts/workflow/docs after.
