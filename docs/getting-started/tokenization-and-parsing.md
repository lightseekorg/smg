---
title: Tokenization and Parsing APIs
---

# Tokenization and Parsing APIs

SMG exposes utility endpoints for tokenization, detokenization, function-call parsing, and reasoning separation.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- For `/v1/tokenize` and `/v1/detokenize`, ensure at least one tokenizer is available (`--model-path` and/or tokenizer registration)

</div>

---

## Tokenize

`POST /v1/tokenize`

Single input:

```bash
curl http://localhost:30000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello world"
  }'
```

Batch input:

```bash
curl http://localhost:30000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": ["Hello", "World"]
  }'
```

---

## Detokenize

`POST /v1/detokenize`

```bash
curl http://localhost:30000/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "tokens": [15496, 995],
    "skip_special_tokens": true
  }'
```

---

## Parse Function Calls

`POST /parse/function_call`

If control-plane auth is configured, include an admin bearer token.

```bash
curl http://localhost:30000/parse/function_call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "text": "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}",
    "tool_call_parser": "json",
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

---

## Parse Reasoning

`POST /parse/reasoning`

If control-plane auth is configured, include an admin bearer token.

```bash
curl http://localhost:30000/parse/reasoning \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "text": "<think>internal reasoning</think>Final answer",
    "reasoning_parser": "deepseek_r1"
  }'
```

---

## Auth Notes

- `tokenize` / `detokenize` are in protected routes and follow API-key middleware when configured.
- `parse/function_call` and `parse/reasoning` are control-plane admin routes when control-plane auth is configured.

---

## Next Steps

- [Gateway Extensions API](../reference/api/extensions.md#tokenize)
- [Gateway Extensions API](../reference/api/extensions.md#parse-function-calls)
- [Admin API Reference](../reference/api/admin.md)
