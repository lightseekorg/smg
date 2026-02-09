---
title: Control Plane Auth
---

# Control Plane Auth

Control plane endpoints are used to manage workers, tokenizers, and WASM modules. Configure admin authentication with JWT/OIDC and/or control-plane API keys.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Decide how admins authenticate (JWT, API key, or both)

</div>

---

## Protected Control Plane Endpoints

These routes are guarded by control-plane auth middleware when configured:

- Worker management: `/workers`, `/workers/{worker_id}`
- Tokenizer management: `/v1/tokenizers`, `/v1/tokenizers/{tokenizer_id}`
- Parser admin endpoints: `/parse/function_call`, `/parse/reasoning`
- WASM management: `/wasm`, `/wasm/{module_uuid}`
- Cache and load endpoints: `/flush_cache`, `/get_loads`

Control-plane middleware requires **admin role**; non-admin principals receive `403`.

---

## Option A: API keys

```bash
smg \
  --worker-urls http://worker:8000 \
  --control-plane-api-keys 'admin1:PlatformAdmin:admin:super-secret-key'
```

Use the key in `Authorization` header:

```bash
curl -H "Authorization: Bearer super-secret-key" \
  http://localhost:30000/v1/tokenizers
```

Format: `id:name:role:key` where role is `admin` or `user`.

---

## Option B: JWT / OIDC

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer https://login.example.com \
  --jwt-audience api://smg-control-plane \
  --jwt-role-claim roles \
  --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'
```

Optional explicit JWKS URI:

```bash
--jwt-jwks-uri https://login.example.com/.well-known/jwks.json
```

JWTs are validated first when configured. If a JWT-shaped token fails validation, SMG does not silently fall back to API key validation.

---

## Option C: JWT + API keys together

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer https://login.example.com \
  --jwt-audience api://smg-control-plane \
  --control-plane-api-keys 'admin1:PlatformAdmin:admin:super-secret-key'
```

This lets human admins use OIDC while service automation uses API keys.

---

## Audit logging

Control-plane auth emits audit logs by default. Disable only if needed:

```bash
smg \
  --worker-urls http://worker:8000 \
  --control-plane-api-keys 'admin1:PlatformAdmin:admin:super-secret-key' \
  --disable-audit-logging
```

---

## Next Steps

- [Admin API Reference](../reference/api/admin.md)
- [Configuration Reference](../reference/configuration.md#control-plane-authentication)
