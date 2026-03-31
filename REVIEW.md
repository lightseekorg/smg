# Code Review Guidelines

Use the `/smg:review-pr` skill for all PR reviews. It contains the full subsystem-aware checklist, file-to-section mapping, and anti-patterns.

This file provides additional context the skill doesn't cover.

## Always flag
- Silent fallbacks to `None`/default when config validation should fail loudly
- `clone()` in gRPC streaming hot paths (per-token response processing)
- `serde(rename)` that doesn't match OpenAI/Anthropic API spec field names
- Worker registry mutations without proper locking (DashMap vs bare HashMap)

## Domain knowledge
- Config changes are the #1 source of bugs — they touch CLI args, types.rs, main.rs (two conversion paths), Python bindings, and Go SDK
- The gRPC pipeline has 8 constructors in pipeline.rs — changes to shared stages must work for all of them
- HTTP and gRPC routers are two separate code paths that must handle the same API contract
- PD disaggregation adds dual-dispatch complexity on top of regular routing

## Skip
- Generated files under `crates/grpc_client/src/` (proto-generated)
- Formatting-only changes
- Documentation-only PRs (`docs/**`)
- Dependency version bumps with no code changes
