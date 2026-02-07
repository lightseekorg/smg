---
title: Extension API
---

# Extension API Reference

This page documents non-OpenAI extension endpoints exposed by SMG, aligned to route registration in `model_gateway/src/server.rs`.

---

## Auth Model

SMG endpoint auth is route-group based:

| Route group | Auth behavior |
|---|---|
| Public routes | No auth middleware (`/health`, `/readiness`, `/liveness`, `/v1/models`, etc.) |
| Protected routes | Standard API auth middleware (`/v1/tokenize`, `/v1/detokenize`, `/generate`, etc.) |
| Control-plane routes | Control-plane auth middleware when configured; otherwise standard API auth |

When control-plane auth is enabled, control-plane endpoints require admin role.

---

## Public Extension Endpoints

These endpoints are available without the protected-route middleware:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Overall gateway health |
| `GET` | `/liveness` | Process liveness probe |
| `GET` | `/readiness` | Traffic readiness probe |
| `GET` | `/health_generate` | Generation health check |
| `GET` | `/engine_metrics` | Engine-level metrics snapshot |
| `GET` | `/v1/models` | List models |
| `GET` | `/get_model_info` | Model metadata |
| `GET` | `/get_server_info` | Server metadata |

---

## Protected Utility Endpoints

These run behind protected-route auth middleware:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/tokenize` | Convert text to token IDs |
| `POST` | `/v1/detokenize` | Convert token IDs to text |
| `POST` | `/generate` | Native generate endpoint |
| `POST` | `/rerank` | Native rerank endpoint |
| `POST` | `/v1/rerank` | OpenAI-style rerank endpoint |
| `POST` | `/v1/messages` | Messages endpoint |
| `POST` | `/v1/classify` | Classification endpoint |

For OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`), see:

- [OpenAI Compatible API](openai.md)
- [Responses API](responses.md)

---

## Control-Plane Endpoints

These endpoints are for gateway operations and administration.

### Worker Management

| Method | Path |
|---|---|
| `GET`, `POST` | `/workers` |
| `GET`, `PUT`, `DELETE` | `/workers/{worker_id}` |

### Tokenizer Management

| Method | Path |
|---|---|
| `GET`, `POST` | `/v1/tokenizers` |
| `GET`, `DELETE` | `/v1/tokenizers/{tokenizer_id}` |
| `GET` | `/v1/tokenizers/{tokenizer_id}/status` |

### Parser Utilities

| Method | Path |
|---|---|
| `POST` | `/parse/function_call` |
| `POST` | `/parse/reasoning` |

### WASM Management

| Method | Path |
|---|---|
| `GET`, `POST` | `/wasm` |
| `DELETE` | `/wasm/{module_uuid}` |

### Cache and Load Utilities

| Method | Path |
|---|---|
| `POST` | `/flush_cache` |
| `GET` | `/get_loads` |

---

## HA / Mesh Management Endpoints

SMG also exposes mesh control routes under `/ha/*`:

| Method | Path |
|---|---|
| `GET` | `/ha/status` |
| `GET` | `/ha/health` |
| `GET` | `/ha/workers` |
| `GET` | `/ha/workers/{worker_id}` |
| `GET` | `/ha/policies` |
| `GET` | `/ha/policies/{model_id}` |
| `GET` | `/ha/config/{key}` |
| `POST` | `/ha/config` |
| `GET`, `POST` | `/ha/rate-limit` |
| `GET` | `/ha/rate-limit/stats` |
| `POST` | `/ha/shutdown` |

---

## Quick Examples

Tokenize:

```bash
curl -X POST http://localhost:30000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"hello"}'
```

List workers (with admin token when control-plane auth is enabled):

```bash
curl http://localhost:30000/workers \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

List tokenizers:

```bash
curl http://localhost:30000/v1/tokenizers \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

---

## Related Pages

- [Admin API Reference](admin.md)
- [Configuration Reference](../configuration.md#control-plane-authentication)
- [Getting Started: Control Plane Operations](../../getting-started/control-plane-operations.md)
