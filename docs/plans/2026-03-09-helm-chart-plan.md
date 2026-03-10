# SMG Helm Chart Implementation Plan

> **Note:** This plan is partially superseded. The v0.1.0 chart implements
> **router-only mode**. Worker and PD tasks were not implemented and are
> deferred to a future release.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a production-ready Helm chart for SMG. The v0.1.0 release supports router-only mode. Worker and PD modes are planned for future releases.

**Architecture:** Single Helm chart at `deploy/helm/smg/` with a top-level `mode` value controlling conditional template rendering. Router deployment is always created; worker and PD templates only render in their respective modes. CLI args are assembled in `_helpers.tpl` from structured values.

**Tech Stack:** Helm v3 (apiVersion: v2), Go templates, JSON Schema for validation

**Design doc:** `docs/plans/2026-03-09-helm-chart-design.md`

---

### Task 1: Chart Scaffolding

**Files:**
- Create: `deploy/helm/smg/Chart.yaml`
- Create: `deploy/helm/smg/.helmignore`

**Step 1: Create the directory structure**

```bash
mkdir -p deploy/helm/smg/templates/worker deploy/helm/smg/templates/pd deploy/helm/smg/examples deploy/helm/smg/tests
```

**Step 2: Create Chart.yaml**

Create `deploy/helm/smg/Chart.yaml`:

```yaml
apiVersion: v2
name: smg
description: Shepherd Model Gateway — high-performance inference router for LLM deployments
type: application
version: 0.1.0
appVersion: "0.7.0"
home: https://github.com/lightseekorg/smg
sources:
  - https://github.com/lightseekorg/smg
maintainers:
  - name: LightSeek
    url: https://github.com/lightseekorg
keywords:
  - llm
  - inference
  - gateway
  - router
  - sglang
  - vllm
dependencies:
  - name: postgresql
    version: "~16"
    repository: https://charts.bitnami.com/bitnami
    condition: history.postgres.deploy
  - name: redis
    version: "~20"
    repository: https://charts.bitnami.com/bitnami
    condition: history.redis.deploy
```

**Step 3: Create .helmignore**

Create `deploy/helm/smg/.helmignore`:

```
.git
.gitignore
*.md
!README.md
examples/
```

**Step 4: Verify chart is parseable**

Run: `helm lint deploy/helm/smg/ 2>&1 | head -20`
Expected: May warn about missing templates/values but should not error on Chart.yaml parsing.

**Step 5: Commit**

```bash
git add deploy/helm/smg/Chart.yaml deploy/helm/smg/.helmignore
git commit -s -m "feat(helm): scaffold chart with Chart.yaml and helmignore"
```

---

### Task 2: values.yaml

**Files:**
- Create: `deploy/helm/smg/values.yaml`

**Step 1: Create values.yaml**

Create `deploy/helm/smg/values.yaml` with the full values structure from the design doc. This is the largest single file. Key sections:

```yaml
# =============================================================================
# SMG Helm Chart - values.yaml
# =============================================================================
# Deployment mode: "router" | "router-worker" | "router-pd"
#   router        - Router only, connect to existing workers
#   router-worker - Router + inference worker pods
#   router-pd     - Router + prefill/decode disaggregated workers
mode: router

# -- Global settings shared across all components
global:
  image:
    registry: docker.io
    repository: lightseekorg/smg
    tag: ""  # defaults to Chart.appVersion
    pullPolicy: IfNotPresent
  imagePullSecrets: []

# =============================================================================
# Router Configuration
# =============================================================================
router:
  replicas: 1

  # -- Routing policy: cache_aware, round_robin, power_of_two,
  #    consistent_hashing, prefix_hash, manual, random, bucket
  policy: cache_aware

  # -- Explicit worker URLs (used when mode=router or as additional workers)
  workerUrls: []
  #   - http://worker-1:8000
  #   - grpc://worker-2:8001

  # -- Kubernetes service discovery (auto-discover worker pods by label)
  serviceDiscovery:
    enabled: false
    namespace: ""          # defaults to release namespace
    selector: ""           # e.g. "app=sglang-worker"
    port: 80
    # -- PD-specific selectors (used when mode=router-pd)
    prefillSelector: ""    # e.g. "component=prefill"
    decodeSelector: ""     # e.g. "component=decode"
    # -- Extract model ID from pod metadata
    modelIdFrom: ""        # "namespace", "label:key", or "annotation:key"

  # -- Model / tokenizer
  model: ""                # HuggingFace model ID or local path
  tokenizerPath: ""        # override tokenizer (defaults to model)
  chatTemplate: ""         # custom chat template path

  # -- Cache-aware routing tuning
  cacheThreshold: 0.3
  balanceAbsThreshold: 64
  balanceRelThreshold: 1.5
  evictionInterval: 120
  maxTreeSize: 67108864
  blockSize: 16

  # -- Request handling
  maxPayloadSize: 536870912    # 512MB in bytes
  requestTimeoutSecs: 1800
  maxConcurrentRequests: -1    # -1 = unlimited
  queueSize: 100
  queueTimeoutSecs: 60

  # -- Resilience: retry
  retry:
    enabled: true
    maxRetries: 5
    initialBackoffMs: 50
    maxBackoffMs: 30000
    backoffMultiplier: 1.5
    jitterFactor: 0.2

  # -- Resilience: circuit breaker
  circuitBreaker:
    enabled: true
    failureThreshold: 10
    successThreshold: 3
    timeoutDurationSecs: 60
    windowDurationSecs: 120

  # -- Resilience: health checks on workers
  healthCheck:
    enabled: true
    failureThreshold: 3
    successThreshold: 2
    timeoutSecs: 5
    intervalSecs: 60
    endpoint: /health

  # -- Networking
  port: 30000
  service:
    type: ClusterIP
    port: 80
    annotations: {}
  ingress:
    enabled: false
    className: ""
    annotations: {}
    hosts: []
    #   - host: smg.example.com
    #     paths:
    #       - path: /
    #         pathType: Prefix
    tls: []

  # -- Observability: Prometheus metrics
  metrics:
    port: 29000
    serviceMonitor:
      enabled: false
      interval: 30s
      labels: {}

  # -- Observability: OpenTelemetry tracing
  tracing:
    enabled: false
    otlpEndpoint: ""

  # -- Observability: logging
  logging:
    level: info
    json: false
    dir: ""

  # -- Pod resources
  resources: {}
  #   requests:
  #     cpu: "2"
  #     memory: 4Gi
  #   limits:
  #     cpu: "4"
  #     memory: 8Gi

  nodeSelector: {}
  tolerations: []
  affinity: {}
  podAnnotations: {}
  podLabels: {}

  # -- Extra CLI args appended to the smg binary invocation
  extraArgs: []

  # -- Extra environment variables
  extraEnv: []

  # -- Autoscaling (HPA)
  autoscaling:
    enabled: false
    minReplicas: 1
    maxReplicas: 5
    targetCPUUtilizationPercentage: 80

  # -- Pod disruption budget
  podDisruptionBudget:
    enabled: false
    minAvailable: 1

  # -- Optional features
  wasm:
    enabled: false
    path: ""
  mcp:
    enabled: false
    configPath: ""
  reasoningParser: ""
  toolCallParser: ""

# =============================================================================
# Auth Configuration
# =============================================================================
auth:
  # -- API key authentication
  apiKey: ""
  apiKeySecret: ""           # name of an existing Secret
  apiKeySecretKey: "api-key" # key within the Secret

  # -- OIDC
  oidc:
    enabled: false
    issuer: ""
    audience: ""

  # -- Rate limiting
  rateLimitTokensPerSecond: -1  # -1 = disabled

# =============================================================================
# History / Storage Backend
# =============================================================================
history:
  # -- Backend type: none, memory, postgres, redis, oracle
  backend: memory

  postgres:
    url: ""                    # "postgresql://user:pass@host:5432/db"
    existingSecret: ""         # Secret name containing key "postgres-url"
    poolMax: 10
    retentionDays: 30
    # -- Deploy a PostgreSQL instance via sub-chart
    deploy: false

  redis:
    url: ""                    # "redis://localhost:6379"
    existingSecret: ""         # Secret name containing key "redis-url"
    poolMax: 10
    # -- Deploy a Redis instance via sub-chart
    deploy: false

  oracle:
    dsn: ""
    user: ""
    password: ""
    existingSecret: ""         # Secret containing oracle-dsn, oracle-user, oracle-password
    poolMax: 10

# =============================================================================
# Worker Configuration (mode: router-worker)
# =============================================================================
worker:
  replicas: 1

  # -- Inference backend: sglang, vllm, trtllm
  backend: sglang

  image:
    # -- Backend-specific image (e.g. lmsysorg/sglang:latest, vllm/vllm-openai:latest)
    repository: ""
    tag: ""
    pullPolicy: IfNotPresent

  # -- Model to serve
  model: ""                    # e.g. "meta-llama/Llama-3-70b"

  # -- Parallelism
  tensorParallelSize: 1
  dataParallelSize: 1

  # -- Worker port
  port: 8000

  # -- GPU resources
  resources:
    limits:
      nvidia.com/gpu: 1

  # -- Model storage volume
  modelVolume:
    # -- Volume type: "emptyDir" (download at start), "hostPath", or "pvc"
    type: emptyDir
    hostPath: ""
    pvcName: ""                # name of an existing PVC
    pvcSize: 100Gi
    pvcStorageClass: ""

  # -- Shared memory size for /dev/shm
  shmSize: "16Gi"

  # -- Extra args passed to the backend CLI (sglang/vllm/trtllm)
  extraArgs: []

  # -- Extra environment variables
  extraEnv: []

  nodeSelector: {}
  tolerations: []
  affinity: {}
  podAnnotations: {}
  podLabels: {}

# =============================================================================
# PD Disaggregation Configuration (mode: router-pd)
# =============================================================================
pd:
  prefill:
    replicas: 1
    policy: cache_aware
    port: 8000
    bootstrapPort: 9001

    image:
      repository: ""
      tag: ""
      pullPolicy: IfNotPresent
    model: ""
    backend: sglang
    tensorParallelSize: 1

    resources:
      limits:
        nvidia.com/gpu: 1

    modelVolume:
      type: emptyDir
      hostPath: ""
      pvcName: ""
      pvcSize: 100Gi
      pvcStorageClass: ""

    shmSize: "16Gi"
    extraArgs: []
    extraEnv: []
    nodeSelector: {}
    tolerations: []
    affinity: {}
    podAnnotations: {}
    podLabels: {}

  decode:
    replicas: 1
    policy: power_of_two
    port: 8000

    image:
      repository: ""
      tag: ""
      pullPolicy: IfNotPresent
    model: ""
    backend: sglang
    tensorParallelSize: 1

    resources:
      limits:
        nvidia.com/gpu: 1

    modelVolume:
      type: emptyDir
      hostPath: ""
      pvcName: ""
      pvcSize: 100Gi
      pvcStorageClass: ""

    shmSize: "16Gi"
    extraArgs: []
    extraEnv: []
    nodeSelector: {}
    tolerations: []
    affinity: {}
    podAnnotations: {}
    podLabels: {}

# =============================================================================
# Service Account & RBAC
# =============================================================================
serviceAccount:
  create: true
  name: ""
  annotations: {}

rbac:
  # -- Create Role/RoleBinding for K8s service discovery (pods list/watch).
  #    Automatically enabled when router.serviceDiscovery.enabled=true.
  create: false

# =============================================================================
# Grafana Dashboard
# =============================================================================
grafana:
  dashboard:
    enabled: false
    labels:
      grafana_dashboard: "1"

# =============================================================================
# Dependency Sub-Charts (overrides go here)
# =============================================================================
postgresql:
  enabled: false

redis:
  enabled: false
```

**Step 2: Lint the chart**

Run: `helm lint deploy/helm/smg/`
Expected: PASS (may warn about missing templates)

**Step 3: Commit**

```bash
git add deploy/helm/smg/values.yaml
git commit -s -m "feat(helm): add values.yaml with all configuration sections"
```

---

### Task 3: Template Helpers

**Files:**
- Create: `deploy/helm/smg/templates/_helpers.tpl`

**Step 1: Create _helpers.tpl**

This file defines reusable template functions. Key helpers:

```gotemplate
{{/*
Expand the name of the chart.
*/}}
{{- define "smg.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "smg.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "smg.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "smg.labels" -}}
helm.sh/chart: {{ include "smg.chart" . }}
{{ include "smg.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.global.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "smg.selectorLabels" -}}
app.kubernetes.io/name: {{ include "smg.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Router image
*/}}
{{- define "smg.routerImage" -}}
{{- $registry := .Values.global.image.registry -}}
{{- $repository := .Values.global.image.repository -}}
{{- $tag := .Values.global.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Worker image — uses worker.image if set, otherwise falls back to a default
based on the backend.
*/}}
{{- define "smg.workerImage" -}}
{{- $img := .image -}}
{{- if and $img.repository $img.tag -}}
  {{- printf "%s:%s" $img.repository $img.tag -}}
{{- else if $img.repository -}}
  {{- $img.repository -}}
{{- else -}}
  {{- if eq .backend "sglang" -}}lmsysorg/sglang:latest
  {{- else if eq .backend "vllm" -}}vllm/vllm-openai:latest
  {{- else -}}{{ fail "worker.image.repository is required for trtllm backend" }}
  {{- end -}}
{{- end -}}
{{- end }}

{{/*
Service account name
*/}}
{{- define "smg.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "smg.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Router CLI arguments — builds the full args list from values.
Called from the router Deployment template.
*/}}
{{- define "smg.routerArgs" -}}
- "--host"
- "0.0.0.0"
- "--port"
- {{ .Values.router.port | quote }}
- "--policy"
- {{ .Values.router.policy | quote }}
{{- /* Worker URLs — combine explicit URLs with auto-wired worker service */}}
{{- if or .Values.router.workerUrls (eq .Values.mode "router-worker") }}
{{- if ne .Values.mode "router-pd" }}
- "--worker-urls"
{{- end }}
{{- range .Values.router.workerUrls }}
- {{ . | quote }}
{{- end }}
{{- if eq .Values.mode "router-worker" }}
- {{ printf "http://%s-worker:%d" (include "smg.fullname" $) (int .Values.worker.port) | quote }}
{{- end }}
{{- end }}
{{- /* Service discovery */}}
{{- if .Values.router.serviceDiscovery.enabled }}
- "--service-discovery"
{{- if .Values.router.serviceDiscovery.selector }}
- "--selector"
- {{ .Values.router.serviceDiscovery.selector | quote }}
{{- end }}
- "--service-discovery-port"
- {{ .Values.router.serviceDiscovery.port | quote }}
{{- if .Values.router.serviceDiscovery.namespace }}
- "--service-discovery-namespace"
- {{ .Values.router.serviceDiscovery.namespace | quote }}
{{- end }}
{{- if .Values.router.serviceDiscovery.modelIdFrom }}
- "--model-id-from"
- {{ .Values.router.serviceDiscovery.modelIdFrom | quote }}
{{- end }}
{{- end }}
{{- /* PD disaggregation */}}
{{- if eq .Values.mode "router-pd" }}
- "--pd-disaggregation"
{{- /* Auto-wire prefill service */}}
- "--prefill"
- {{ printf "http://%s-prefill:%d" (include "smg.fullname" $) (int .Values.pd.prefill.port) | quote }}
{{- if .Values.pd.prefill.bootstrapPort }}
- {{ .Values.pd.prefill.bootstrapPort | quote }}
{{- end }}
{{- /* Auto-wire decode service */}}
- "--decode"
- {{ printf "http://%s-decode:%d" (include "smg.fullname" $) (int .Values.pd.decode.port) | quote }}
- "--prefill-policy"
- {{ .Values.pd.prefill.policy | quote }}
- "--decode-policy"
- {{ .Values.pd.decode.policy | quote }}
{{- /* PD service discovery selectors */}}
{{- if .Values.router.serviceDiscovery.prefillSelector }}
- "--prefill-selector"
- {{ .Values.router.serviceDiscovery.prefillSelector | quote }}
{{- end }}
{{- if .Values.router.serviceDiscovery.decodeSelector }}
- "--decode-selector"
- {{ .Values.router.serviceDiscovery.decodeSelector | quote }}
{{- end }}
{{- end }}
{{- /* Model / tokenizer */}}
{{- if .Values.router.model }}
- "--model-path"
- {{ .Values.router.model | quote }}
{{- end }}
{{- if .Values.router.tokenizerPath }}
- "--tokenizer-path"
- {{ .Values.router.tokenizerPath | quote }}
{{- end }}
{{- if .Values.router.chatTemplate }}
- "--chat-template"
- {{ .Values.router.chatTemplate | quote }}
{{- end }}
{{- /* Cache tuning */}}
- "--cache-threshold"
- {{ .Values.router.cacheThreshold | quote }}
- "--balance-abs-threshold"
- {{ .Values.router.balanceAbsThreshold | quote }}
- "--balance-rel-threshold"
- {{ .Values.router.balanceRelThreshold | quote }}
- "--eviction-interval"
- {{ .Values.router.evictionInterval | quote }}
- "--max-tree-size"
- {{ .Values.router.maxTreeSize | quote }}
- "--block-size"
- {{ .Values.router.blockSize | quote }}
{{- /* Request handling */}}
- "--max-payload-size"
- {{ .Values.router.maxPayloadSize | quote }}
- "--request-timeout-secs"
- {{ .Values.router.requestTimeoutSecs | quote }}
- "--max-concurrent-requests"
- {{ .Values.router.maxConcurrentRequests | quote }}
- "--queue-size"
- {{ .Values.router.queueSize | quote }}
- "--queue-timeout-secs"
- {{ .Values.router.queueTimeoutSecs | quote }}
{{- /* Retry */}}
{{- if not .Values.router.retry.enabled }}
- "--disable-retries"
{{- else }}
- "--retry-max-retries"
- {{ .Values.router.retry.maxRetries | quote }}
- "--retry-initial-backoff-ms"
- {{ .Values.router.retry.initialBackoffMs | quote }}
- "--retry-max-backoff-ms"
- {{ .Values.router.retry.maxBackoffMs | quote }}
- "--retry-backoff-multiplier"
- {{ .Values.router.retry.backoffMultiplier | quote }}
- "--retry-jitter-factor"
- {{ .Values.router.retry.jitterFactor | quote }}
{{- end }}
{{- /* Circuit breaker */}}
{{- if not .Values.router.circuitBreaker.enabled }}
- "--disable-circuit-breaker"
{{- else }}
- "--cb-failure-threshold"
- {{ .Values.router.circuitBreaker.failureThreshold | quote }}
- "--cb-success-threshold"
- {{ .Values.router.circuitBreaker.successThreshold | quote }}
- "--cb-timeout-duration-secs"
- {{ .Values.router.circuitBreaker.timeoutDurationSecs | quote }}
- "--cb-window-duration-secs"
- {{ .Values.router.circuitBreaker.windowDurationSecs | quote }}
{{- end }}
{{- /* Health checks */}}
{{- if not .Values.router.healthCheck.enabled }}
- "--disable-health-check"
{{- else }}
- "--health-failure-threshold"
- {{ .Values.router.healthCheck.failureThreshold | quote }}
- "--health-success-threshold"
- {{ .Values.router.healthCheck.successThreshold | quote }}
- "--health-check-timeout-secs"
- {{ .Values.router.healthCheck.timeoutSecs | quote }}
- "--health-check-interval-secs"
- {{ .Values.router.healthCheck.intervalSecs | quote }}
- "--health-check-endpoint"
- {{ .Values.router.healthCheck.endpoint | quote }}
{{- end }}
{{- /* Observability */}}
- "--prometheus-port"
- {{ .Values.router.metrics.port | quote }}
- "--log-level"
- {{ .Values.router.logging.level | quote }}
{{- if .Values.router.logging.json }}
- "--log-json"
{{- end }}
{{- if .Values.router.logging.dir }}
- "--log-dir"
- {{ .Values.router.logging.dir | quote }}
{{- end }}
{{- if .Values.router.tracing.enabled }}
- "--otlp-traces-endpoint"
- {{ .Values.router.tracing.otlpEndpoint | quote }}
{{- end }}
{{- /* History backend */}}
{{- if and (ne .Values.history.backend "memory") (ne .Values.history.backend "none") }}
- "--history-backend"
- {{ .Values.history.backend | quote }}
{{- end }}
{{- if eq .Values.history.backend "postgres" }}
{{- if .Values.history.postgres.url }}
- "--postgres-url"
- {{ .Values.history.postgres.url | quote }}
{{- end }}
- "--postgres-pool-max"
- {{ .Values.history.postgres.poolMax | quote }}
- "--postgres-retention-days"
- {{ .Values.history.postgres.retentionDays | quote }}
{{- end }}
{{- if eq .Values.history.backend "redis" }}
{{- if .Values.history.redis.url }}
- "--redis-url"
- {{ .Values.history.redis.url | quote }}
{{- end }}
- "--redis-pool-max"
- {{ .Values.history.redis.poolMax | quote }}
{{- end }}
{{- if eq .Values.history.backend "oracle" }}
{{- if .Values.history.oracle.dsn }}
- "--oracle-dsn"
- {{ .Values.history.oracle.dsn | quote }}
{{- end }}
- "--oracle-pool-max"
- {{ .Values.history.oracle.poolMax | quote }}
{{- end }}
{{- /* Rate limiting */}}
{{- if gt (int .Values.auth.rateLimitTokensPerSecond) 0 }}
- "--rate-limit-tokens-per-second"
- {{ .Values.auth.rateLimitTokensPerSecond | quote }}
{{- end }}
{{- /* WASM */}}
{{- if .Values.router.wasm.enabled }}
- "--enable-wasm"
{{- if .Values.router.wasm.path }}
- "--storage-hook-wasm-path"
- {{ .Values.router.wasm.path | quote }}
{{- end }}
{{- end }}
{{- /* MCP */}}
{{- if .Values.router.mcp.enabled }}
- "--mcp-config-path"
- {{ .Values.router.mcp.configPath | quote }}
{{- end }}
{{- /* Parsers */}}
{{- if .Values.router.reasoningParser }}
- "--reasoning-parser"
- {{ .Values.router.reasoningParser | quote }}
{{- end }}
{{- if .Values.router.toolCallParser }}
- "--tool-call-parser"
- {{ .Values.router.toolCallParser | quote }}
{{- end }}
{{- /* Extra args (escape hatch) */}}
{{- range .Values.router.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}
```

**Step 2: Lint**

Run: `helm lint deploy/helm/smg/`
Expected: PASS

**Step 3: Commit**

```bash
git add deploy/helm/smg/templates/_helpers.tpl
git commit -s -m "feat(helm): add template helpers with router args builder"
```

---

### Task 4: Router Templates (core — always deployed)

**Files:**
- Create: `deploy/helm/smg/templates/deployment-router.yaml`
- Create: `deploy/helm/smg/templates/service-router.yaml`
- Create: `deploy/helm/smg/templates/serviceaccount.yaml`
- Create: `deploy/helm/smg/templates/secret.yaml`
- Create: `deploy/helm/smg/templates/NOTES.txt`

**Step 1: Create deployment-router.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: router
spec:
  {{- if not .Values.router.autoscaling.enabled }}
  replicas: {{ .Values.router.replicas }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: router
  template:
    metadata:
      annotations:
        {{- with .Values.router.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "smg.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: router
        {{- with .Values.router.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.global.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "smg.serviceAccountName" . }}
      containers:
        - name: router
          image: {{ include "smg.routerImage" . }}
          imagePullPolicy: {{ .Values.global.image.pullPolicy }}
          args:
            {{- include "smg.routerArgs" . | nindent 12 }}
          ports:
            - name: http
              containerPort: {{ .Values.router.port }}
              protocol: TCP
            - name: metrics
              containerPort: {{ .Values.router.metrics.port }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          {{- with .Values.router.resources }}
          resources:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          env:
            {{- if .Values.auth.apiKey }}
            - name: SMG_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "smg.fullname" . }}-secrets
                  key: api-key
            {{- else if .Values.auth.apiKeySecret }}
            - name: SMG_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.auth.apiKeySecret }}
                  key: {{ .Values.auth.apiKeySecretKey }}
            {{- end }}
            {{- if and (eq .Values.history.backend "postgres") .Values.history.postgres.existingSecret }}
            - name: POSTGRES_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.history.postgres.existingSecret }}
                  key: postgres-url
            {{- end }}
            {{- if and (eq .Values.history.backend "redis") .Values.history.redis.existingSecret }}
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.history.redis.existingSecret }}
                  key: redis-url
            {{- end }}
            {{- with .Values.router.extraEnv }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
      {{- with .Values.router.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.router.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.router.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

**Step 2: Create service-router.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: router
  {{- with .Values.router.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.router.service.type }}
  ports:
    - name: http
      port: {{ .Values.router.service.port }}
      targetPort: http
      protocol: TCP
    - name: metrics
      port: {{ .Values.router.metrics.port }}
      targetPort: metrics
      protocol: TCP
  selector:
    {{- include "smg.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: router
```

**Step 3: Create serviceaccount.yaml**

```yaml
{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "smg.serviceAccountName" . }}
  labels:
    {{- include "smg.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}
```

**Step 4: Create secret.yaml**

```yaml
{{- if or .Values.auth.apiKey (and (eq .Values.history.backend "postgres") .Values.history.postgres.url (not .Values.history.postgres.existingSecret)) (and (eq .Values.history.backend "redis") .Values.history.redis.url (not .Values.history.redis.existingSecret)) (and (eq .Values.history.backend "oracle") .Values.history.oracle.dsn (not .Values.history.oracle.existingSecret)) }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "smg.fullname" . }}-secrets
  labels:
    {{- include "smg.labels" . | nindent 4 }}
type: Opaque
stringData:
  {{- if .Values.auth.apiKey }}
  api-key: {{ .Values.auth.apiKey | quote }}
  {{- end }}
  {{- if and (eq .Values.history.backend "postgres") .Values.history.postgres.url (not .Values.history.postgres.existingSecret) }}
  postgres-url: {{ .Values.history.postgres.url | quote }}
  {{- end }}
  {{- if and (eq .Values.history.backend "redis") .Values.history.redis.url (not .Values.history.redis.existingSecret) }}
  redis-url: {{ .Values.history.redis.url | quote }}
  {{- end }}
  {{- if and (eq .Values.history.backend "oracle") .Values.history.oracle.dsn (not .Values.history.oracle.existingSecret) }}
  oracle-dsn: {{ .Values.history.oracle.dsn | quote }}
  oracle-user: {{ .Values.history.oracle.user | quote }}
  oracle-password: {{ .Values.history.oracle.password | quote }}
  {{- end }}
{{- end }}
```

**Step 5: Create NOTES.txt**

```
{{- $fullname := include "smg.fullname" . -}}

SMG deployed in "{{ .Values.mode }}" mode.

Router:
  Endpoint: http://{{ $fullname }}-router:{{ .Values.router.service.port }}
  Metrics:  http://{{ $fullname }}-router:{{ .Values.router.metrics.port }}/metrics
  Policy:   {{ .Values.router.policy }}

{{- if eq .Values.mode "router-worker" }}

Workers:
  Backend:  {{ .Values.worker.backend }}
  Replicas: {{ .Values.worker.replicas }}
  Model:    {{ .Values.worker.model }}
{{- end }}

{{- if eq .Values.mode "router-pd" }}

PD Disaggregation:
  Prefill:  {{ .Values.pd.prefill.replicas }} replica(s), policy={{ .Values.pd.prefill.policy }}
  Decode:   {{ .Values.pd.decode.replicas }} replica(s), policy={{ .Values.pd.decode.policy }}
{{- end }}

{{- if .Values.router.ingress.enabled }}

Ingress:
{{- range .Values.router.ingress.hosts }}
  http{{ if $.Values.router.ingress.tls }}s{{ end }}://{{ .host }}
{{- end }}
{{- end }}

To test the deployment:
  helm test {{ .Release.Name }}
```

**Step 6: Lint and template**

Run: `helm lint deploy/helm/smg/ && helm template test deploy/helm/smg/ --set router.workerUrls[0]=http://w:8000 | head -80`
Expected: Lint PASS, rendered YAML output with Deployment, Service, ServiceAccount.

**Step 7: Commit**

```bash
git add deploy/helm/smg/templates/deployment-router.yaml deploy/helm/smg/templates/service-router.yaml deploy/helm/smg/templates/serviceaccount.yaml deploy/helm/smg/templates/secret.yaml deploy/helm/smg/templates/NOTES.txt
git commit -s -m "feat(helm): add router deployment, service, serviceaccount, and secret templates"
```

---

### Task 5: RBAC Templates

**Files:**
- Create: `deploy/helm/smg/templates/role.yaml`
- Create: `deploy/helm/smg/templates/rolebinding.yaml`

**Step 1: Create role.yaml**

```yaml
{{- if or .Values.rbac.create .Values.router.serviceDiscovery.enabled }}
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {{ include "smg.fullname" . }}
  labels:
    {{- include "smg.labels" . | nindent 4 }}
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["list", "watch"]
{{- end }}
```

**Step 2: Create rolebinding.yaml**

```yaml
{{- if or .Values.rbac.create .Values.router.serviceDiscovery.enabled }}
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {{ include "smg.fullname" . }}
  labels:
    {{- include "smg.labels" . | nindent 4 }}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: {{ include "smg.fullname" . }}
subjects:
  - kind: ServiceAccount
    name: {{ include "smg.serviceAccountName" . }}
    namespace: {{ .Release.Namespace }}
{{- end }}
```

**Step 3: Test with service discovery**

Run: `helm template test deploy/helm/smg/ --set router.workerUrls[0]=http://w:8000 --set router.serviceDiscovery.enabled=true | grep "kind: Role"`
Expected: Shows both `Role` and `RoleBinding`.

Run: `helm template test deploy/helm/smg/ --set router.workerUrls[0]=http://w:8000 | grep "kind: Role"`
Expected: No output (RBAC not rendered when discovery disabled).

**Step 4: Commit**

```bash
git add deploy/helm/smg/templates/role.yaml deploy/helm/smg/templates/rolebinding.yaml
git commit -s -m "feat(helm): add RBAC templates for service discovery"
```

---

### Task 6: Optional Router Templates (Ingress, HPA, PDB, ServiceMonitor, Grafana)

**Files:**
- Create: `deploy/helm/smg/templates/ingress.yaml`
- Create: `deploy/helm/smg/templates/hpa-router.yaml`
- Create: `deploy/helm/smg/templates/pdb.yaml`
- Create: `deploy/helm/smg/templates/servicemonitor.yaml`
- Create: `deploy/helm/smg/templates/grafana-dashboard.yaml`

**Step 1: Create ingress.yaml**

```yaml
{{- if .Values.router.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
  {{- with .Values.router.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.router.ingress.className }}
  ingressClassName: {{ .Values.router.ingress.className }}
  {{- end }}
  {{- if .Values.router.ingress.tls }}
  tls:
    {{- toYaml .Values.router.ingress.tls | nindent 4 }}
  {{- end }}
  rules:
    {{- range .Values.router.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "smg.fullname" $ }}-router
                port:
                  name: http
          {{- end }}
    {{- end }}
{{- end }}
```

**Step 2: Create hpa-router.yaml**

```yaml
{{- if .Values.router.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "smg.fullname" . }}-router
  minReplicas: {{ .Values.router.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.router.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.router.autoscaling.targetCPUUtilizationPercentage }}
{{- end }}
```

**Step 3: Create pdb.yaml**

```yaml
{{- if .Values.router.podDisruptionBudget.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
spec:
  minAvailable: {{ .Values.router.podDisruptionBudget.minAvailable }}
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: router
{{- end }}
```

**Step 4: Create servicemonitor.yaml**

```yaml
{{- if .Values.router.metrics.serviceMonitor.enabled }}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "smg.fullname" . }}-router
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    {{- with .Values.router.metrics.serviceMonitor.labels }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
spec:
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: router
  endpoints:
    - port: metrics
      interval: {{ .Values.router.metrics.serviceMonitor.interval }}
{{- end }}
```

**Step 5: Create grafana-dashboard.yaml**

```yaml
{{- if .Values.grafana.dashboard.enabled }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "smg.fullname" . }}-grafana-dashboard
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    {{- with .Values.grafana.dashboard.labels }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
data:
  smg-dashboard.json: |
    {
      "annotations": { "list": [] },
      "editable": true,
      "fiscalYearStartMonth": 0,
      "graphTooltip": 0,
      "links": [],
      "panels": [
        {
          "title": "Request Rate",
          "type": "timeseries",
          "datasource": "Prometheus",
          "targets": [{ "expr": "rate(smg_requests_total[5m])", "legendFormat": "{{method}} {{status}}" }],
          "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
        },
        {
          "title": "Request Latency (p99)",
          "type": "timeseries",
          "datasource": "Prometheus",
          "targets": [{ "expr": "histogram_quantile(0.99, rate(smg_request_duration_seconds_bucket[5m]))", "legendFormat": "p99" }],
          "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 }
        },
        {
          "title": "Active Workers",
          "type": "stat",
          "datasource": "Prometheus",
          "targets": [{ "expr": "smg_active_workers", "legendFormat": "workers" }],
          "gridPos": { "h": 4, "w": 6, "x": 0, "y": 8 }
        },
        {
          "title": "Queue Depth",
          "type": "timeseries",
          "datasource": "Prometheus",
          "targets": [{ "expr": "smg_queue_depth", "legendFormat": "queue" }],
          "gridPos": { "h": 8, "w": 12, "x": 0, "y": 12 }
        }
      ],
      "schemaVersion": 39,
      "tags": ["smg", "llm", "inference"],
      "templating": { "list": [] },
      "time": { "from": "now-1h", "to": "now" },
      "title": "SMG Overview",
      "uid": "smg-overview"
    }
{{- end }}
```

**Step 6: Lint**

Run: `helm lint deploy/helm/smg/`
Expected: PASS

**Step 7: Commit**

```bash
git add deploy/helm/smg/templates/ingress.yaml deploy/helm/smg/templates/hpa-router.yaml deploy/helm/smg/templates/pdb.yaml deploy/helm/smg/templates/servicemonitor.yaml deploy/helm/smg/templates/grafana-dashboard.yaml
git commit -s -m "feat(helm): add ingress, HPA, PDB, ServiceMonitor, and Grafana dashboard templates"
```

---

### Task 7: Worker Templates (mode: router-worker)

**Files:**
- Create: `deploy/helm/smg/templates/worker/deployment.yaml`
- Create: `deploy/helm/smg/templates/worker/service.yaml`

**Step 1: Create worker/deployment.yaml**

```yaml
{{- if eq .Values.mode "router-worker" }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "smg.fullname" . }}-worker
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: worker
spec:
  replicas: {{ .Values.worker.replicas }}
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: worker
  template:
    metadata:
      {{- with .Values.worker.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "smg.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: worker
        {{- with .Values.worker.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.global.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
        - name: worker
          image: {{ include "smg.workerImage" (dict "image" .Values.worker.image "backend" .Values.worker.backend) }}
          imagePullPolicy: {{ .Values.worker.image.pullPolicy }}
          args:
            - "--model"
            - {{ .Values.worker.model | quote }}
            - "--port"
            - {{ .Values.worker.port | quote }}
            - "--host"
            - "0.0.0.0"
            {{- if gt (int .Values.worker.tensorParallelSize) 1 }}
            - "--tensor-parallel-size"
            - {{ .Values.worker.tensorParallelSize | quote }}
            {{- end }}
            {{- if gt (int .Values.worker.dataParallelSize) 1 }}
            - "--data-parallel-size"
            - {{ .Values.worker.dataParallelSize | quote }}
            {{- end }}
            {{- range .Values.worker.extraArgs }}
            - {{ . | quote }}
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.worker.port }}
              protocol: TCP
          resources:
            {{- toYaml .Values.worker.resources | nindent 12 }}
          env:
            {{- with .Values.worker.extraEnv }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            {{- if ne .Values.worker.modelVolume.type "emptyDir" }}
            - name: models
              mountPath: /models
            {{- end }}
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: {{ .Values.worker.shmSize }}
        {{- if eq .Values.worker.modelVolume.type "hostPath" }}
        - name: models
          hostPath:
            path: {{ .Values.worker.modelVolume.hostPath }}
            type: Directory
        {{- else if eq .Values.worker.modelVolume.type "pvc" }}
        - name: models
          persistentVolumeClaim:
            claimName: {{ .Values.worker.modelVolume.pvcName }}
        {{- end }}
      {{- with .Values.worker.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.worker.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.worker.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
{{- end }}
```

**Step 2: Create worker/service.yaml**

```yaml
{{- if eq .Values.mode "router-worker" }}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "smg.fullname" . }}-worker
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: worker
spec:
  type: ClusterIP
  ports:
    - name: http
      port: {{ .Values.worker.port }}
      targetPort: http
      protocol: TCP
  selector:
    {{- include "smg.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: worker
{{- end }}
```

**Step 3: Test rendering**

Run: `helm template test deploy/helm/smg/ --set mode=router-worker --set worker.model=meta-llama/Llama-3-70b | grep "kind:" | sort | uniq -c`
Expected: Shows Deployment (x2: router + worker), Service (x2), ServiceAccount.

Run: `helm template test deploy/helm/smg/ --set router.workerUrls[0]=http://w:8000 | grep "worker" | head -5`
Expected: No worker Deployment (mode=router by default).

**Step 4: Commit**

```bash
git add deploy/helm/smg/templates/worker/
git commit -s -m "feat(helm): add worker deployment and service templates"
```

---

### Task 8: PD Templates (mode: router-pd)

**Files:**
- Create: `deploy/helm/smg/templates/pd/deployment-prefill.yaml`
- Create: `deploy/helm/smg/templates/pd/deployment-decode.yaml`
- Create: `deploy/helm/smg/templates/pd/service-prefill.yaml`
- Create: `deploy/helm/smg/templates/pd/service-decode.yaml`

**Step 1: Create pd/deployment-prefill.yaml**

Same structure as worker deployment, but uses `pd.prefill.*` values and adds `bootstrapPort` as an extra container port. Component label: `prefill`.

```yaml
{{- if eq .Values.mode "router-pd" }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "smg.fullname" . }}-prefill
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: prefill
spec:
  replicas: {{ .Values.pd.prefill.replicas }}
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: prefill
  template:
    metadata:
      {{- with .Values.pd.prefill.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "smg.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: prefill
        {{- with .Values.pd.prefill.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.global.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
        - name: prefill
          image: {{ include "smg.workerImage" (dict "image" .Values.pd.prefill.image "backend" .Values.pd.prefill.backend) }}
          imagePullPolicy: {{ .Values.pd.prefill.image.pullPolicy }}
          args:
            - "--model"
            - {{ .Values.pd.prefill.model | quote }}
            - "--port"
            - {{ .Values.pd.prefill.port | quote }}
            - "--host"
            - "0.0.0.0"
            {{- if gt (int .Values.pd.prefill.tensorParallelSize) 1 }}
            - "--tensor-parallel-size"
            - {{ .Values.pd.prefill.tensorParallelSize | quote }}
            {{- end }}
            {{- range .Values.pd.prefill.extraArgs }}
            - {{ . | quote }}
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.pd.prefill.port }}
              protocol: TCP
            {{- if .Values.pd.prefill.bootstrapPort }}
            - name: bootstrap
              containerPort: {{ .Values.pd.prefill.bootstrapPort }}
              protocol: TCP
            {{- end }}
          resources:
            {{- toYaml .Values.pd.prefill.resources | nindent 12 }}
          env:
            {{- with .Values.pd.prefill.extraEnv }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            {{- if ne .Values.pd.prefill.modelVolume.type "emptyDir" }}
            - name: models
              mountPath: /models
            {{- end }}
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: {{ .Values.pd.prefill.shmSize }}
        {{- if eq .Values.pd.prefill.modelVolume.type "hostPath" }}
        - name: models
          hostPath:
            path: {{ .Values.pd.prefill.modelVolume.hostPath }}
            type: Directory
        {{- else if eq .Values.pd.prefill.modelVolume.type "pvc" }}
        - name: models
          persistentVolumeClaim:
            claimName: {{ .Values.pd.prefill.modelVolume.pvcName }}
        {{- end }}
      {{- with .Values.pd.prefill.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.pd.prefill.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.pd.prefill.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
{{- end }}
```

**Step 2: Create pd/deployment-decode.yaml**

Same as prefill but uses `pd.decode.*` values, component label `decode`, no bootstrap port.

```yaml
{{- if eq .Values.mode "router-pd" }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "smg.fullname" . }}-decode
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: decode
spec:
  replicas: {{ .Values.pd.decode.replicas }}
  selector:
    matchLabels:
      {{- include "smg.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: decode
  template:
    metadata:
      {{- with .Values.pd.decode.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "smg.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: decode
        {{- with .Values.pd.decode.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.global.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
        - name: decode
          image: {{ include "smg.workerImage" (dict "image" .Values.pd.decode.image "backend" .Values.pd.decode.backend) }}
          imagePullPolicy: {{ .Values.pd.decode.image.pullPolicy }}
          args:
            - "--model"
            - {{ .Values.pd.decode.model | quote }}
            - "--port"
            - {{ .Values.pd.decode.port | quote }}
            - "--host"
            - "0.0.0.0"
            {{- if gt (int .Values.pd.decode.tensorParallelSize) 1 }}
            - "--tensor-parallel-size"
            - {{ .Values.pd.decode.tensorParallelSize | quote }}
            {{- end }}
            {{- range .Values.pd.decode.extraArgs }}
            - {{ . | quote }}
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.pd.decode.port }}
              protocol: TCP
          resources:
            {{- toYaml .Values.pd.decode.resources | nindent 12 }}
          env:
            {{- with .Values.pd.decode.extraEnv }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            {{- if ne .Values.pd.decode.modelVolume.type "emptyDir" }}
            - name: models
              mountPath: /models
            {{- end }}
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: {{ .Values.pd.decode.shmSize }}
        {{- if eq .Values.pd.decode.modelVolume.type "hostPath" }}
        - name: models
          hostPath:
            path: {{ .Values.pd.decode.modelVolume.hostPath }}
            type: Directory
        {{- else if eq .Values.pd.decode.modelVolume.type "pvc" }}
        - name: models
          persistentVolumeClaim:
            claimName: {{ .Values.pd.decode.modelVolume.pvcName }}
        {{- end }}
      {{- with .Values.pd.decode.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.pd.decode.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.pd.decode.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
{{- end }}
```

**Step 3: Create pd/service-prefill.yaml**

```yaml
{{- if eq .Values.mode "router-pd" }}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "smg.fullname" . }}-prefill
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: prefill
spec:
  type: ClusterIP
  ports:
    - name: http
      port: {{ .Values.pd.prefill.port }}
      targetPort: http
      protocol: TCP
    {{- if .Values.pd.prefill.bootstrapPort }}
    - name: bootstrap
      port: {{ .Values.pd.prefill.bootstrapPort }}
      targetPort: bootstrap
      protocol: TCP
    {{- end }}
  selector:
    {{- include "smg.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: prefill
{{- end }}
```

**Step 4: Create pd/service-decode.yaml**

```yaml
{{- if eq .Values.mode "router-pd" }}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "smg.fullname" . }}-decode
  labels:
    {{- include "smg.labels" . | nindent 4 }}
    app.kubernetes.io/component: decode
spec:
  type: ClusterIP
  ports:
    - name: http
      port: {{ .Values.pd.decode.port }}
      targetPort: http
      protocol: TCP
  selector:
    {{- include "smg.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: decode
{{- end }}
```

**Step 5: Test rendering**

Run: `helm template test deploy/helm/smg/ --set mode=router-pd --set pd.prefill.model=m --set pd.decode.model=m | grep "kind:" | sort | uniq -c`
Expected: Deployment (x3: router + prefill + decode), Service (x3), ServiceAccount.

Run: `helm template test deploy/helm/smg/ --set mode=router-pd --set pd.prefill.model=m --set pd.decode.model=m | grep "pd-disaggregation"`
Expected: Shows `--pd-disaggregation` in router args.

**Step 6: Commit**

```bash
git add deploy/helm/smg/templates/pd/
git commit -s -m "feat(helm): add PD disaggregation prefill and decode templates"
```

---

### Task 9: Helm Test

**Files:**
- Create: `deploy/helm/smg/tests/test-connection.yaml`

**Step 1: Create test-connection.yaml**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "smg.fullname" . }}-test-connection"
  labels:
    {{- include "smg.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: curl
      image: curlimages/curl:latest
      command:
        - curl
        - -sf
        - http://{{ include "smg.fullname" . }}-router:{{ .Values.router.service.port }}/health
  restartPolicy: Never
```

**Step 2: Lint**

Run: `helm lint deploy/helm/smg/`
Expected: PASS with no errors.

**Step 3: Commit**

```bash
git add deploy/helm/smg/tests/test-connection.yaml
git commit -s -m "feat(helm): add helm test for router health check"
```

---

### Task 10: JSON Schema Validation

**Files:**
- Create: `deploy/helm/smg/values.schema.json`

**Step 1: Create values.schema.json**

Validates the critical enums and required fields:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "mode": {
      "type": "string",
      "enum": ["router", "router-worker", "router-pd"],
      "description": "Deployment mode"
    },
    "router": {
      "type": "object",
      "properties": {
        "policy": {
          "type": "string",
          "enum": ["cache_aware", "round_robin", "power_of_two", "consistent_hashing", "prefix_hash", "manual", "random", "bucket"]
        },
        "port": { "type": "integer", "minimum": 1, "maximum": 65535 },
        "replicas": { "type": "integer", "minimum": 1 }
      }
    },
    "worker": {
      "type": "object",
      "properties": {
        "backend": {
          "type": "string",
          "enum": ["sglang", "vllm", "trtllm"]
        },
        "replicas": { "type": "integer", "minimum": 1 },
        "port": { "type": "integer", "minimum": 1, "maximum": 65535 }
      }
    },
    "history": {
      "type": "object",
      "properties": {
        "backend": {
          "type": "string",
          "enum": ["none", "memory", "postgres", "redis", "oracle"]
        }
      }
    }
  },
  "required": ["mode"]
}
```

**Step 2: Test schema validation**

Run: `helm lint deploy/helm/smg/ --set mode=invalid 2>&1`
Expected: Error about invalid mode value.

Run: `helm lint deploy/helm/smg/ --set mode=router --set router.workerUrls[0]=http://w:8000`
Expected: PASS

**Step 3: Commit**

```bash
git add deploy/helm/smg/values.schema.json
git commit -s -m "feat(helm): add JSON Schema validation for values"
```

---

### Task 11: Example Value Files

**Files:**
- Create: `deploy/helm/smg/examples/router-only.yaml`
- Create: `deploy/helm/smg/examples/router-worker-sglang.yaml`
- Create: `deploy/helm/smg/examples/router-worker-vllm.yaml`
- Create: `deploy/helm/smg/examples/router-worker-trtllm.yaml`
- Create: `deploy/helm/smg/examples/router-pd.yaml`
- Create: `deploy/helm/smg/examples/with-postgres.yaml`
- Create: `deploy/helm/smg/examples/with-service-discovery.yaml`
- Create: `deploy/helm/smg/examples/with-ingress.yaml`
- Create: `deploy/helm/smg/examples/with-monitoring.yaml`
- Create: `deploy/helm/smg/examples/production.yaml`

**Step 1: Create all example files**

Each file should be a minimal, commented values override showing one scenario. For example:

`router-only.yaml`:
```yaml
# Minimal router-only deployment pointing at existing workers.
# Usage: helm install smg deploy/helm/smg -f examples/router-only.yaml
mode: router
router:
  workerUrls:
    - http://worker-1.default.svc:8000
    - http://worker-2.default.svc:8000
  policy: cache_aware
```

`router-worker-sglang.yaml`:
```yaml
# Router with 2 sglang workers, each using 4 GPUs.
# Usage: helm install smg deploy/helm/smg -f examples/router-worker-sglang.yaml
mode: router-worker
router:
  policy: cache_aware
worker:
  replicas: 2
  backend: sglang
  model: meta-llama/Llama-3-70b
  tensorParallelSize: 4
  resources:
    limits:
      nvidia.com/gpu: 4
    requests:
      cpu: "8"
      memory: 64Gi
  modelVolume:
    type: hostPath
    hostPath: /models
```

`router-pd.yaml`:
```yaml
# PD disaggregation: 2 prefill workers + 4 decode workers.
# Usage: helm install smg deploy/helm/smg -f examples/router-pd.yaml
mode: router-pd
pd:
  prefill:
    replicas: 2
    policy: cache_aware
    model: meta-llama/Llama-3-70b
    tensorParallelSize: 4
    resources:
      limits:
        nvidia.com/gpu: 4
  decode:
    replicas: 4
    policy: power_of_two
    model: meta-llama/Llama-3-70b
    tensorParallelSize: 2
    resources:
      limits:
        nvidia.com/gpu: 2
```

`production.yaml`:
```yaml
# Production-hardened deployment with HPA, PDB, monitoring, and resource limits.
# Usage: helm install smg deploy/helm/smg -f examples/production.yaml
mode: router-worker
router:
  replicas: 3
  policy: cache_aware
  resources:
    requests:
      cpu: "4"
      memory: 8Gi
    limits:
      cpu: "8"
      memory: 16Gi
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  podDisruptionBudget:
    enabled: true
    minAvailable: 2
  metrics:
    serviceMonitor:
      enabled: true
      interval: 15s
  logging:
    level: info
    json: true
  ingress:
    enabled: true
    className: nginx
    annotations:
      nginx.ingress.kubernetes.io/proxy-read-timeout: "1800"
      nginx.ingress.kubernetes.io/proxy-send-timeout: "1800"
    hosts:
      - host: smg.example.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: smg-tls
        hosts:
          - smg.example.com
worker:
  replicas: 4
  backend: sglang
  model: meta-llama/Llama-3-70b
  tensorParallelSize: 4
  resources:
    limits:
      nvidia.com/gpu: 4
    requests:
      cpu: "16"
      memory: 128Gi
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  modelVolume:
    type: pvc
    pvcName: model-storage
grafana:
  dashboard:
    enabled: true
```

Create the remaining examples (with-postgres, with-service-discovery, with-ingress, with-monitoring, router-worker-vllm, router-worker-trtllm) following the same pattern — minimal, commented, one concern per file.

**Step 2: Validate each example renders**

Run: `for f in deploy/helm/smg/examples/*.yaml; do echo "=== $f ===" && helm template test deploy/helm/smg/ -f "$f" > /dev/null && echo "OK" || echo "FAIL"; done`
Expected: All OK.

**Step 3: Commit**

```bash
git add deploy/helm/smg/examples/
git commit -s -m "docs(helm): add example values for all deployment scenarios"
```

---

### Task 12: README

**Files:**
- Create: `deploy/helm/smg/README.md`

**Step 1: Write README.md**

Structure:
1. Title + badges
2. Overview (1 paragraph)
3. Prerequisites (K8s >=1.26, Helm >=3.12, NVIDIA GPU Operator for worker modes)
4. Quick Start — three code blocks, one per mode
5. Configuration — link to values.yaml, document key sections
6. Examples — table linking to each example file with description
7. Upgrading — placeholder for future versions
8. Troubleshooting — GPU scheduling, worker discovery, common errors

**Step 2: Commit**

```bash
git add deploy/helm/smg/README.md
git commit -s -m "docs(helm): add comprehensive README with quickstart and examples"
```

---

### Task 13: Final Validation

**Step 1: Full lint**

Run: `helm lint deploy/helm/smg/`
Expected: PASS, 0 errors, 0 warnings.

**Step 2: Template all three modes**

```bash
helm template test deploy/helm/smg/ --set router.workerUrls[0]=http://w:8000 > /dev/null && echo "router: OK"
helm template test deploy/helm/smg/ --set mode=router-worker --set worker.model=m > /dev/null && echo "router-worker: OK"
helm template test deploy/helm/smg/ --set mode=router-pd --set pd.prefill.model=m --set pd.decode.model=m > /dev/null && echo "router-pd: OK"
```
Expected: All three OK.

**Step 3: Template all examples**

```bash
for f in deploy/helm/smg/examples/*.yaml; do
  helm template test deploy/helm/smg/ -f "$f" > /dev/null 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```
Expected: All OK.

**Step 4: Schema validation negative test**

Run: `helm lint deploy/helm/smg/ --set mode=invalid 2>&1`
Expected: Error mentioning enum validation.

**Step 5: Verify no files are left unstaged**

Run: `git status deploy/helm/`
Expected: Clean working tree (all committed).
