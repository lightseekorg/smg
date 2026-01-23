---
title: Authentication
---

# Authentication

SMG supports multiple authentication methods for securing access to inference APIs and the control plane, including JWT/OIDC integration, API keys, and role-based access control.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-key-chain: Multiple Auth Methods

Support for JWT/OIDC, API keys, and worker authentication to fit your deployment model.

</div>

<div class="card" markdown>

### :material-shield-account: Role-Based Access

Admin and user roles control access to control plane vs. data plane APIs.

</div>

<div class="card" markdown>

### :material-office-building: Enterprise SSO

Integrate with Keycloak, Auth0, Azure AD, Okta, and other OIDC providers.

</div>

<div class="card" markdown>

### :material-clipboard-text: Audit Logging

Track all control plane operations for security monitoring and compliance.

</div>

</div>

---

## Authentication Methods

| Method | Use Case | Configuration |
|--------|----------|---------------|
| **JWT/OIDC** | Enterprise SSO integration with identity providers | `--jwt-issuer`, `--jwt-audience` |
| **API Keys** | Service accounts and programmatic access | `--control-plane-api-keys` |
| **Worker API Key** | Gateway-to-worker authentication | `--api-key` |

### When to Use Each Method

- **JWT/OIDC**: Use for enterprise deployments with existing identity providers (Keycloak, Auth0, Azure AD, Okta). Provides centralized user management and SSO.
- **API Keys**: Use for service-to-service communication, CI/CD pipelines, and automated tooling. Simpler to set up but requires manual key management.
- **Worker API Key**: Use when workers require authentication for data parallel (DP) aware scheduling or when workers are deployed with API key protection.

---

## JWT/OIDC Authentication

JWT (JSON Web Token) authentication allows integration with OIDC-compliant identity providers for enterprise single sign-on.

### Configuration Options

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| `--jwt-issuer` | `JWT_ISSUER` | OIDC issuer URL (required for JWT auth) |
| `--jwt-audience` | `JWT_AUDIENCE` | Expected audience claim (required for JWT auth) |
| `--jwt-jwks-uri` | `JWT_JWKS_URI` | Explicit JWKS URI (auto-discovered if not set) |
| `--jwt-role-claim` | - | Claim name containing roles (default: `roles`) |
| `--jwt-role-mapping` | - | Map IDP roles to gateway roles |

### Basic Setup

Enable JWT authentication by providing the issuer and audience:

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://auth.example.com/realms/myrealm" \
  --jwt-audience "smg-gateway"
```

### JWKS Discovery

By default, SMG discovers the JWKS (JSON Web Key Set) endpoint automatically via OIDC discovery (`/.well-known/openid-configuration`). You can override this with an explicit JWKS URI:

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://auth.example.com" \
  --jwt-audience "smg-gateway" \
  --jwt-jwks-uri "https://auth.example.com/.well-known/jwks.json"
```

### Role Mapping

Map identity provider roles to SMG gateway roles:

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://auth.example.com" \
  --jwt-audience "smg-gateway" \
  --jwt-role-mapping "Gateway.Admin=admin" \
  --jwt-role-mapping "Gateway.User=user"
```

**Role Mapping Format**: `idp_role=gateway_role`

| Gateway Role | Permissions |
|--------------|-------------|
| `admin` | Full access to all control plane APIs (workers, WASM modules, tokenizers) |
| `user` | Access to inference/data plane APIs only |

### Supported Claims

SMG extracts roles from the following claims (in order of precedence):

1. Configured `--jwt-role-claim` (default: `roles`)
2. `role` claim
3. `roles` claim
4. `groups` claim
5. `group` claim

If no role is found, the user defaults to the `user` role.

### Supported Algorithms

SMG supports the following JWT signing algorithms:

- **RSA**: RS256, RS384, RS512
- **ECDSA**: ES256, ES384

---

## Identity Provider Setup

### Keycloak

<div class="grid" markdown>

<div class="card" markdown>

**1. Create a Client**

- Navigate to Clients > Create
- Client ID: `smg-gateway`
- Client Protocol: `openid-connect`
- Access Type: `confidential` or `public`

</div>

<div class="card" markdown>

**2. Configure Mappers**

- Add a mapper of type "User Realm Role"
- Token Claim Name: `roles`
- Add to ID token: Yes
- Add to access token: Yes

</div>

</div>

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://keycloak.example.com/realms/myrealm" \
  --jwt-audience "smg-gateway" \
  --jwt-role-mapping "admin=admin" \
  --jwt-role-mapping "user=user"
```

### Auth0

<div class="grid" markdown>

<div class="card" markdown>

**1. Create an API**

- Navigate to Applications > APIs > Create API
- Name: `SMG Gateway`
- Identifier: `https://smg.example.com/api`

</div>

<div class="card" markdown>

**2. Add Roles Action**

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://smg.example.com';
  if (event.authorization) {
    api.accessToken.setCustomClaim(
      `${namespace}/roles`,
      event.authorization.roles
    );
  }
};
```

</div>

</div>

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://your-tenant.auth0.com/" \
  --jwt-audience "https://smg.example.com/api" \
  --jwt-role-claim "https://smg.example.com/roles" \
  --jwt-role-mapping "smg-admin=admin" \
  --jwt-role-mapping "smg-user=user"
```

### Azure AD / Entra ID

<div class="grid" markdown>

<div class="card" markdown>

**1. Register an Application**

- Navigate to Azure Portal > App registrations
- Name: `SMG Gateway`
- Configure app roles: `Gateway.Admin`, `Gateway.User`

</div>

<div class="card" markdown>

**2. Expose an API**

- Navigate to Expose an API
- Set Application ID URI: `api://smg-gateway`

</div>

</div>

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://smg-gateway" \
  --jwt-role-mapping "Gateway.Admin=admin" \
  --jwt-role-mapping "Gateway.User=user"
```

### Okta

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://your-org.okta.com/oauth2/default" \
  --jwt-audience "api://smg" \
  --jwt-role-mapping "smg_admins=admin" \
  --jwt-role-mapping "smg_users=user"
```

---

## API Key Authentication

API keys provide a simpler authentication method for service accounts and programmatic access.

### Control Plane API Keys

Configure API keys for control plane access:

```bash
smg \
  --worker-urls http://worker:8000 \
  --control-plane-api-keys "key1:Service Account:admin:sk-your-secret-key-here"
```

**Format**: `id:name:role:key`

| Component | Description |
|-----------|-------------|
| `id` | Unique identifier for the key |
| `name` | Human-readable name/description |
| `role` | Gateway role (`admin` or `user`) |
| `key` | The secret API key value |

### Multiple API Keys

```bash
smg \
  --worker-urls http://worker:8000 \
  --control-plane-api-keys "admin1:Admin Service:admin:sk-admin-key-12345" \
  --control-plane-api-keys "user1:Read Only Service:user:sk-readonly-key-67890"
```

### Environment Variable Configuration

For security, pass API keys via environment variable:

```bash
export CONTROL_PLANE_API_KEYS="admin1:Admin Service:admin:sk-admin-key-12345"
smg --worker-urls http://worker:8000
```

### Using API Keys

Clients authenticate by including the API key in the Authorization header:

```bash
curl -H "Authorization: Bearer sk-admin-key-12345" \
  https://smg.example.com/workers
```

### Security Features

- **Hashed Storage**: Keys are SHA-256 hashed immediately; plaintext keys are never stored in memory
- **Constant-Time Comparison**: Key verification uses constant-time comparison to prevent timing attacks
- **Role-Based Access**: Each key is assigned a specific role limiting its permissions

---

## Worker API Key Authentication

The `--api-key` option configures an API key for gateway-to-worker communication:

```bash
smg \
  --worker-urls http://worker:8000 \
  --api-key "worker-secret-key"
```

This is useful when:

- Workers require authentication (e.g., deployed with API key protection)
- Using DP-aware scheduling that requires authenticated worker queries
- Workers are behind an authentication proxy

The gateway automatically adds `Authorization: Bearer <api-key>` to requests sent to workers.

---

## Role-Based Access Control

SMG implements role-based access control (RBAC) with two primary roles:

### Admin Role

Full access to all control plane APIs:

- Worker management (`/workers`, `/workers/{id}`)
- WASM module management (`/wasm/*`)
- Tokenizer configuration
- System administration

### User Role

Access to inference/data plane APIs only:

- Chat completions (`/v1/chat/completions`)
- Completions (`/v1/completions`)
- Embeddings (`/v1/embeddings`)
- Model listing (`/v1/models`)

### Role Assignment

Roles are assigned through:

1. **JWT Claims**: Via `--jwt-role-mapping` configuration
2. **API Key Configuration**: Via the role component in `--control-plane-api-keys`

If no role can be determined, the user defaults to `user` role for safety.

---

## Audit Logging

SMG provides audit logging for control plane operations to support security monitoring and compliance.

### Configuration

Audit logging is **enabled by default** when authentication is configured. To disable:

```bash
smg \
  --worker-urls http://worker:8000 \
  --jwt-issuer "https://auth.example.com" \
  --jwt-audience "smg-gateway" \
  --disable-audit-logging
```

### Audit Log Format

Audit events are logged with structured fields:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "principal": "user@example.com",
  "auth_method": "jwt",
  "role": "admin",
  "method": "POST",
  "path": "/workers",
  "resource": "worker-123",
  "outcome": "success",
  "request_id": "req-abc-123"
}
```

### Audit Event Fields

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp of the event |
| `principal` | User ID, email, or API key ID |
| `auth_method` | Authentication method (`jwt`, `api_key`) |
| `role` | Role of the principal (`admin`, `user`) |
| `method` | HTTP method (GET, POST, DELETE, etc.) |
| `path` | Request path |
| `resource` | Resource being accessed (if applicable) |
| `outcome` | Result (`success`, `denied`) |
| `request_id` | Correlation ID for request tracing |

### Viewing Audit Logs

```bash
# Filter for audit logs
RUST_LOG=smg::audit=info smg ...

# Or view in combined logs
kubectl logs -n inference -l app=smg | grep "control_plane_audit"
```

---

## Production Configuration

A production setup combining JWT and API key authentication:

```bash
smg \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --host 0.0.0.0 \
  --port 443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --jwt-issuer "https://auth.example.com/realms/production" \
  --jwt-audience "smg-gateway" \
  --jwt-role-mapping "Gateway.Admin=admin" \
  --jwt-role-mapping "Gateway.User=user" \
  --control-plane-api-keys "ci-cd:CI/CD Pipeline:admin:${CI_CD_API_KEY}" \
  --control-plane-api-keys "monitoring:Prometheus:user:${MONITORING_API_KEY}"
```

---

## Kubernetes Deployment

### Secret for API Keys

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: smg-auth
  namespace: inference
type: Opaque
stringData:
  CONTROL_PLANE_API_KEYS: "admin1:Admin:admin:sk-secret-key"
```

### ConfigMap for JWT Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: smg-config
  namespace: inference
data:
  JWT_ISSUER: "https://auth.example.com/realms/production"
  JWT_AUDIENCE: "smg-gateway"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smg
  namespace: inference
spec:
  template:
    spec:
      containers:
        - name: smg
          image: smg:latest
          envFrom:
            - configMapRef:
                name: smg-config
            - secretRef:
                name: smg-auth
          args:
            - --service-discovery
            - --selector
            - app=sglang-worker
            - --jwt-role-mapping
            - "Gateway.Admin=admin"
            - --jwt-role-mapping
            - "Gateway.User=user"
```

---

## Troubleshooting

### JWT Validation Failures

**Symptom**: `Invalid JWT` or `Token validation failed` errors

**Solutions**:

1. Verify issuer URL matches exactly (including trailing slash):
   ```bash
   curl https://auth.example.com/.well-known/openid-configuration
   ```

2. Verify audience claim matches your configuration:
   ```bash
   echo "YOUR_JWT" | cut -d. -f2 | base64 -d | jq .
   ```

3. Check clock synchronization (JWT validation uses time-based claims)

4. Verify JWKS endpoint is accessible from the SMG pod

### API Key Not Working

**Symptom**: `Invalid authentication token` errors

**Solutions**:

1. Verify key format is correct: `id:name:role:key`
2. Check for special characters that may need escaping
3. Ensure the Authorization header format is correct: `Bearer <key>`

### Role Mapping Issues

**Symptom**: Users getting wrong permissions

**Solutions**:

1. Check which claim contains roles in your JWT
2. Verify role mapping syntax: `idp_role=gateway_role`
3. Check if role claim name needs to be specified with `--jwt-role-claim`

---

## Security Best Practices

<div class="grid" markdown>

<div class="card" markdown>

### :material-shield-lock: Use HTTPS

Always enable TLS for the gateway in production.

</div>

<div class="card" markdown>

### :material-key-change: Rotate Keys

Regularly rotate API keys and use short-lived JWT tokens.

</div>

<div class="card" markdown>

### :material-account-lock: Least Privilege

Assign `user` role by default, `admin` only when needed.

</div>

<div class="card" markdown>

### :material-clipboard-text: Enable Auditing

Keep audit logs for security monitoring and compliance.

</div>

</div>

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-traffic-light: Rate Limiting

Protect against overload and abuse.

[Rate Limiting →](../reliability/rate-limiting.md)

</div>

<div class="card" markdown>

### :material-shield-check: High Availability

Deploy SMG in a highly available configuration.

[High Availability →](../architecture/high-availability.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Monitor authentication metrics.

[Metrics Reference →](../../reference/metrics.md)

</div>

</div>
