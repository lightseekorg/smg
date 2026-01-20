---
title: Authentication Configuration
---

# Authentication Configuration

This guide covers authentication and authorization configuration for SMG, including JWT/OIDC integration, API key authentication, role-based access control, and audit logging.

---

## Overview

SMG supports multiple authentication methods for securing access to the control plane APIs:

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

Map identity provider roles to SMG gateway roles using `--jwt-role-mapping`:

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

## Setting Up OIDC with Common Providers

### Keycloak

1. **Create a Client**:
   - Navigate to Clients > Create
   - Client ID: `smg-gateway`
   - Client Protocol: `openid-connect`
   - Access Type: `confidential` or `public` (depending on your use case)

2. **Configure Mappers** (for role claims):
   - Add a mapper of type "User Realm Role" or "User Client Role"
   - Token Claim Name: `roles`
   - Add to ID token: Yes
   - Add to access token: Yes

3. **Get Configuration Values**:
   ```
   Issuer: https://keycloak.example.com/realms/myrealm
   Audience: smg-gateway
   ```

4. **Configure SMG**:
   ```bash
   smg \
     --worker-urls http://worker:8000 \
     --jwt-issuer "https://keycloak.example.com/realms/myrealm" \
     --jwt-audience "smg-gateway" \
     --jwt-role-mapping "admin=admin" \
     --jwt-role-mapping "user=user"
   ```

### Auth0

1. **Create an API**:
   - Navigate to Applications > APIs > Create API
   - Name: `SMG Gateway`
   - Identifier: `https://smg.example.com/api`

2. **Create Roles**:
   - Navigate to User Management > Roles
   - Create `smg-admin` and `smg-user` roles

3. **Add Roles to Access Token** (via Auth0 Action):
   ```javascript
   exports.onExecutePostLogin = async (event, api) => {
     const namespace = 'https://smg.example.com';
     if (event.authorization) {
       api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
     }
   };
   ```

4. **Configure SMG**:
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

1. **Register an Application**:
   - Navigate to Azure Portal > App registrations > New registration
   - Name: `SMG Gateway`
   - Supported account types: Select based on your requirements

2. **Configure App Roles**:
   - Navigate to App roles > Create app role
   - Create `Gateway.Admin` and `Gateway.User` roles

3. **Expose an API**:
   - Navigate to Expose an API
   - Set Application ID URI: `api://smg-gateway`

4. **Get Configuration Values**:
   ```
   Issuer: https://login.microsoftonline.com/{tenant-id}/v2.0
   Audience: api://smg-gateway (or your client ID)
   ```

5. **Configure SMG**:
   ```bash
   smg \
     --worker-urls http://worker:8000 \
     --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
     --jwt-audience "api://smg-gateway" \
     --jwt-role-mapping "Gateway.Admin=admin" \
     --jwt-role-mapping "Gateway.User=user"
   ```

### Okta

1. **Create an Authorization Server** (or use the default):
   - Navigate to Security > API > Authorization Servers

2. **Create Scopes and Claims**:
   - Add a custom claim for roles
   - Claim name: `roles`
   - Value type: Groups
   - Filter: Matches regex `.*` (or specific groups)

3. **Configure SMG**:
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

Configure API keys for control plane access using `--control-plane-api-keys`:

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

You can configure multiple API keys:

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

API keys in SMG include several security measures:

- **Hashed Storage**: Keys are SHA-256 hashed immediately upon loading; plaintext keys are never stored in memory
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
| `details` | Additional context or error message |

### Security Features

Audit logging includes:

- **Input Sanitization**: Log injection attacks are prevented by escaping control characters
- **Truncation**: Long inputs are truncated to prevent log flooding
- **Structured Format**: Machine-readable format for SIEM integration

### Viewing Audit Logs

Audit events are logged to the `smg::audit` target:

```bash
# Filter for audit logs
RUST_LOG=smg::audit=info smg ...

# Or view in combined logs
kubectl logs -n inference -l app=smg | grep "control_plane_audit"
```

---

## Combined Configuration Example

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
   # Check your IDP's discovery endpoint
   curl https://auth.example.com/.well-known/openid-configuration
   ```

2. Verify audience claim matches your configuration:
   ```bash
   # Decode your JWT to inspect claims
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

1. Check which claim contains roles in your JWT:
   ```bash
   echo "YOUR_JWT" | cut -d. -f2 | base64 -d | jq .
   ```

2. Verify role mapping syntax: `idp_role=gateway_role`
3. Check if role claim name needs to be specified with `--jwt-role-claim`

### JWKS Fetch Errors

**Symptom**: `Failed to fetch JWKS` or SSRF protection errors

**Solutions**:

1. Ensure JWKS endpoint uses HTTPS (required for production)
2. Check network connectivity to the OIDC provider
3. Verify the JWKS URI is not pointing to internal/private addresses

---

## Security Best Practices

1. **Use HTTPS**: Always enable TLS for the gateway in production
2. **Rotate API Keys**: Regularly rotate API keys and use short-lived JWT tokens
3. **Principle of Least Privilege**: Assign `user` role by default, `admin` only when needed
4. **Enable Audit Logging**: Keep audit logs for security monitoring and compliance
5. **Secure Secrets**: Use Kubernetes secrets or a secret manager for API keys
6. **Network Segmentation**: Restrict network access to the control plane APIs

---

## What's Next?

- [Configure TLS](../tasks/deployment/tls.md) - Secure communications with TLS/mTLS
- [Monitoring](../tasks/operations/monitoring.md) - Set up observability
- [Rate Limiting](../concepts/reliability/rate-limiting.md) - Protect against overload
