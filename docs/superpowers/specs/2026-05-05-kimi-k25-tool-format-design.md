# Kimi-K2.5 Tool-Format Fix — Design

**Status:** Approved
**Date:** 2026-05-05
**Branch:** `keyang/kimi_tool_parser`

## Problem

SMG's tokenizer crate renders Kimi-K2.5 chat templates with the wrong tool format. The model was trained on a TypeScript-namespace serialization of tool definitions; SMG emits raw JSON instead. This costs ~1.7 pp on BFCL, concentrated in `simple_python` where the tool definition dominates the prompt.

### Root cause

Kimi-K2.5 ships a custom Python tokenizer (`tokenization_kimi.py`) declared via `tokenizer_config.json::auto_map.AutoTokenizer`. That class overrides `apply_chat_template` to:

1. Call `encode_tools_to_typescript_style(tools)` (in `tool_declaration_ts.py`, 475 lines) → a TypeScript namespace string.
2. Inject the result as `tools_ts_str` into the template render context.

The model's `chat_template.jinja` branches on this variable:

```jinja
{%- if tools_ts_str -%}
  {{ tools_ts_str }}
{%- else -%}
  {{ tools | tojson }}
{%- endif -%}
```

Python frameworks (vLLM, SGLang, TokenSpeed-HTTP) load the tokenizer through `transformers.AutoTokenizer.apply_chat_template(..., trust_remote_code=True)` so the Python override fires and `tools_ts_str` is set. SMG renders the template directly via Rust + minijinja (`crates/tokenizer/src/chat_template.rs:611`) and never executes the Python preprocessor, so `tools_ts_str` is undefined and the template falls into the JSON branch.

### Verified facts

- `crates/tokenizer/src/chat_template.rs:644-653` builds the minijinja context with `messages`, `add_generation_prompt`, `tools`, `documents`, `bos_token`, `eos_token`, `unk_token`, `pad_token` — no `tools_ts_str`.
- `crates/tokenizer/src/huggingface.rs:242-289` reads `tokenizer_config.json` for the chat template and special tokens but never inspects `auto_map`.
- `crates/tokenizer/Cargo.toml` has no `pyo3` dependency.
- Existing model-specific dispatch lives in `huggingface.rs::Renderer` (lines 19-24): `Jinja`, `DeepseekV32`, `DeepseekV4`. Detection is via `config.json::architectures` string match (`huggingface.rs:452-484`). DeepSeek variants replace the entire renderer; minijinja is bypassed.
- `crates/tokenizer/src/tiktoken.rs` has no `Renderer` enum today; its `apply_chat_template` (lines 515-529) is a passthrough to `ChatTemplateState::apply`.
- Only **Kimi-K2.5** is affected: `Kimi-K2-Instruct` and `Kimi-K2-Thinking` ship a 349-line `tokenization_kimi.py` with no TS encoder, no `tool_declaration_ts.py`, and their `chat_template.jinja` does not reference `tools_ts_str`. K2.5 ships a 352-line variant with the TS encoder and 2 template references.
- Kimi-K2.5's `config.json::architectures = ["KimiK25ForConditionalGeneration"]`, `model_type = "kimi_k25"`, `tokenizer_class = "TikTokenTokenizer"` — so the model loads via SMG's Tiktoken path, not the HuggingFace path.

## Goals

1. When SMG serves a Kimi-K2.5 model, render `tools_ts_str` correctly so the model receives the trained TypeScript format.
2. Other models — including Kimi-K2-Instruct and Kimi-K2-Thinking — are unaffected.
3. Add the fix as an extensible scaffold: a future model needing similar context enrichment should require adding one variant to a `Renderer` enum and one encoder file, not redesigning the tokenizer crate.

## Non-goals

- Generic embedded-Python fallback (PyO3) for arbitrary `trust_remote_code` models. We hand-port the one encoder we need.
- Generic `ChatTemplateEnricher` trait abstracted across both tokenizer types. Per the chosen approach, each tokenizer file owns its own `Renderer` enum dispatch table; we accept that the `Renderer` concept exists in two places (tiktoken.rs and huggingface.rs) and that a future enricher-style model needs a parallel variant in each. This is the same pattern DeepSeek uses today.
- Auto-discovery of arbitrary `auto_map`-declared tokenizers. Detection is an explicit architecture-string allowlist.
- Fixing K2-Instruct or K2-Thinking — they don't have this bug.

## Approach

Approach A from brainstorming: extend the existing `Renderer` enum pattern. Add a new variant `Renderer::KimiK25Tools` to `huggingface.rs` and introduce the same enum to `tiktoken.rs` (which has no `Renderer` today). The variant dispatches to a thin renderer wrapper that:

1. Computes `tools_ts_str` via a hand-ported encoder.
2. Merges it into `template_kwargs`.
3. Delegates the actual render to the existing minijinja-backed `ChatTemplateState::apply`.

Unlike DeepSeek (which replaces the whole renderer with hand-written Rust), Kimi-K2.5 only needs context enrichment — minijinja still renders the template, just with one extra variable.

## Architecture

### Code layout

```
crates/tokenizer/src/
  encoders/
    kimi_k25_tools.rs           NEW — TS-namespace encoder + apply_kimi_k25_tools wrapper
    mod.rs                      add `pub(crate) mod kimi_k25_tools;`
  tiktoken.rs                   ADD: enum Renderer { Jinja, KimiK25Tools }, renderer field,
                                     detect_renderer_from_config(dir), match in apply_chat_template
  huggingface.rs                EXTEND: add Renderer::KimiK25Tools to existing enum,
                                     extend detect_renderer_from_config with one branch (defensive)

crates/tokenizer/tests/
  kimi_k25_tools_encoder.rs     NEW — golden tests for the TS encoder
  kimi_k25_chat_template.rs     NEW — end-to-end render test
  fixtures/kimi_k25/
    schemas/*.json              NEW — schema fixtures
    expected/*.txt              NEW — Python-generated reference outputs
    regenerate.py               NEW — fixture regeneration script (not run in CI)
```

### Renderer wrapper (the layer-2 entry point)

```rust
// crates/tokenizer/src/encoders/kimi_k25_tools.rs

pub fn apply_kimi_k25_tools(
    chat_template: &ChatTemplateState,
    messages: &[serde_json::Value],
    params: ChatTemplateParams,
) -> Result<String> {
    let ts_str = params.tools.and_then(encode_tools_to_typescript);

    let merged_kwargs = match (params.template_kwargs, ts_str.as_deref()) {
        (Some(existing), Some(ts)) => {
            let mut m = existing.clone();
            m.insert("tools_ts_str".to_string(), Value::String(ts.to_owned()));
            Some(m)
        }
        (None, Some(ts)) => Some(HashMap::from([
            ("tools_ts_str".to_string(), Value::String(ts.to_owned())),
        ])),
        _ => params.template_kwargs.cloned(),  // tools empty/None → leave undefined
    };

    let new_params = ChatTemplateParams {
        template_kwargs: merged_kwargs.as_ref(),
        ..params
    };
    chat_template.apply(messages, new_params)
}
```

### Encoder API

```rust
/// Encode an array of OpenAI-style tool definitions into Kimi-K2.5's
/// TypeScript-namespace format. Output matches the Python reference
/// `encode_tools_to_typescript_style` byte-for-byte.
///
/// Returns None if `tools` is empty — callers leave `tools_ts_str`
/// undefined so the template branches into the no-tools path.
pub fn encode_tools_to_typescript(tools: &[serde_json::Value]) -> Option<String>;
```

Internally split into private helpers (`encode_tool`, `encode_schema`, `encode_object`, `encode_array`, `encode_union`, `encode_primitive`, `render_description`, `resolve_ref`, `deep_sort_object_keys`). A small `Ctx` struct carries the schema root for `$ref` resolution, current indent depth, and a recursion budget (32 levels; on overflow emit `any` and continue). The encoder never panics — malformed schemas degrade to `any`, matching the Python reference's permissive behavior.

### Dispatch

```rust
// tiktoken.rs (and parallel structure in huggingface.rs)
match self.renderer {
    Renderer::Jinja => self.chat_template.apply(messages, params_with_tokens),
    Renderer::KimiK25Tools => apply_kimi_k25_tools(&self.chat_template, messages, params_with_tokens),
}
```

## Detection

Both `tiktoken.rs` and `huggingface.rs` consult `config.json::architectures`. If the array contains `"KimiK25ForConditionalGeneration"`, the tokenizer is constructed with `Renderer::KimiK25Tools`; otherwise `Renderer::Jinja`.

```rust
fn detect_renderer_from_config(dir: &Path) -> Renderer {
    // read config.json, parse, extract architectures
    // missing/unreadable/malformed → Renderer::Jinja with debug log
    if arch_strs.contains(&"KimiK25ForConditionalGeneration") {
        return Renderer::KimiK25Tools;
    }
    Renderer::Jinja
}
```

The branch is added to `huggingface.rs` defensively even though Kimi-K2.5 ships a Tiktoken tokenizer in practice — if a future build wires it via HF tokenizer.json, the same enricher kicks in.

## Error handling and observability

- Encoder returns `Option<String>`, never `Result`, never panics. Schema-level malformation → emit `any` and continue. A slightly imperfect TS rendering is strictly better than a 500.
- Detection fall-throughs (missing/malformed `config.json`, unknown architecture) silently return `Renderer::Jinja` with a `debug!` log. Matches DeepSeek's behavior.
- Three log points, all `debug!` (no `error!`/`warn!` for encoder behavior):
  - At detection: `"selected KimiK25Tools renderer"`.
  - At each render call when enricher fires: `tool_count`, `ts_bytes`.
  - On individual schema fallback: `reason`, `schema`.
- No new metrics in this change. The existing chat-template-render histogram captures end-to-end latency.

### Edge cases (each gets a fixture)

| Input | Behavior |
|---|---|
| `tools = []` or absent | Return `None`. |
| Tool missing `function` field | Skip, continue. |
| Tool with empty `parameters` | `type name = (_: {}) => any;` |
| Schema missing `type` | `any` |
| Unknown `type` | `any` |
| `enum` field | String-literal union |
| `oneOf` / `anyOf` | Recursive union |
| `allOf` | Best-effort merge, last-wins on conflict |
| `$ref` to local `$defs` | Resolve and inline |
| `$ref` to unknown target | `any`, debug log |
| Circular `$ref` | Recursion budget catches it; `any` |
| Deeply nested object (>32 levels) | `any` at the limit |
| Unicode in description / property names | Pass through |
| Property named `class`, `function`, etc. | Pass through unquoted (matches Python) |

## Performance

- Encoder is `O(total schema nodes)`, single recursive descent. Single `String` buffer with `write!` macros.
- Per render, not memoized in v1. Typical 10-tool BFCL prompt produces ~3 KB; expected runtime well under 100 µs. If profiling later shows it matters, key a small LRU on `Arc<Vec<Value>>` identity.
- Encoder is pure, `Send + Sync`, no locks.

## Backward compatibility

- Models without `KimiK25ForConditionalGeneration` architecture: zero behavior change.
- Kimi-K2.5 models currently served by SMG: their tool-prompt output changes from JSON to TypeScript. **This is the bug fix and is the point.**
- DeepSeek V3.2 / V4: untouched (different `Renderer` variants).

## Test plan

Three layers. **The matrix below is the upper bound used to validate the port end-to-end. Before merge, prune aggressively to the essential cases — keep only what catches a class of regression that no other test catches.**

### Layer 1 — Encoder golden tests

`crates/tokenizer/tests/kimi_k25_tools_encoder.rs` walks `fixtures/kimi_k25/schemas/*.json`, runs the Rust encoder, asserts byte-equality against `expected/*.txt`. Failure prints a unified diff.

`fixtures/kimi_k25/regenerate.py` is the source-of-truth capture mechanism: imports the actual Python `tool_declaration_ts` and writes byte-exact output. Checked in but **not run in CI**. Re-run by hand if the Python reference ever changes.

Validation matrix (~25 fixtures during development; **prune to ~5-8 for merge** — at minimum: simple object, nested object, optional field, union, `$ref`, edge case):

- Primitives × 4
- Optional vs required mix × 2
- Object shapes (flat / nested / deep) × 3
- Arrays (of primitives / objects / unions) × 3
- Unions (`enum` / `oneOf` / `anyOf`) × 3
- `$ref` (local / nested / missing) × 3
- Descriptions (single-line / multi-line / unicode) × 3
- Edge cases (empty / missing type / unknown type / circular) × 4

### Layer 2 — Renderer-wrapper integration

`crates/tokenizer/tests/kimi_k25_chat_template.rs`:

1. Load Kimi-K2.5's `chat_template.jinja` (vendored fixture), build a chat with messages + tools, render through `apply_kimi_k25_tools`, assert the prompt contains the TS namespace block and not the JSON fallback marker.
2. No-tools path — same template, empty `tools` — produces output with no tool block.
3. Caller-supplied `template_kwargs` round-trip alongside the injected `tools_ts_str`.

**Prune to merge**: keep #1 (the headline case) and #2 (the no-tools regression guard). #3 is nice-to-have.

### Layer 3 — Detection

Extend `crates/tokenizer/tests/deepseek_renderer_detection.rs` (or split into a new file) with cases for `Renderer::KimiK25Tools`. **Prune to merge**: at minimum one positive (architecture matches → `KimiK25Tools`) and one negative (missing `config.json` → `Jinja`, no panic). The remaining variants (malformed JSON, missing `architectures` array, unknown architecture) all share the same fall-through code path; a single representative test covers them.

### What's NOT tested in CI

- Live Python comparison — the fixtures are the captured contract, regenerated manually.
- BFCL accuracy regression — manual eval, attached to the PR description.
- Load-time benchmark — detection cost is identical to existing DeepSeek detection.

### Pre-merge verification

1. `cargo test -p tokenizer` green.
2. Run `regenerate.py`, confirm zero diff against committed fixtures.
3. Manual smoke against a Kimi-K2.5 deployment with `RUST_LOG=tokenizer=debug`, eyeball the rendered prompt for the TS block.
4. BFCL eval, expect ~+1.7 pp recovery on `simple_python`; numbers attached to the PR.

## What we're explicitly not doing

- No generic `ChatTemplateEnricher` trait. The Renderer enum is the dispatch mechanism; the wrapper is one specific renderer.
- No PyO3 / embedded Python.
- No public exposure of the encoder outside the tokenizer crate.
- No cross-file `$ref` resolution. Tool schemas are self-contained.
- No fix for K2-Instruct / K2-Thinking. They don't have the bug.
- No retroactive fix for the existing tool-id-formatting hack at `model_gateway/src/routers/grpc/utils/chat_utils.rs:526-545`. That's a separate concern (output side, not input side).
