---
title: Run with Docker
---

# Run SMG with Docker

This task shows you how to deploy SMG using Docker.

<div class="prerequisites" markdown>

#### Before you begin

- Docker 20.10 or later installed
- Network access to your inference workers

</div>

---

## Basic Deployment

### Step 1: Pull the image

```bash
docker pull lightseekorg/smg:latest
```

### Step 2: Run the container

```bash
docker run -d \
  --name smg \
  -p 30000:30000 \
  -p 29000:29000 \
  lightseekorg/smg:latest \
  --worker-urls http://host.docker.internal:8000 \
  --policy cache_aware \
  --prometheus-port 29000
```

!!! note "Networking"
    Use `host.docker.internal` to reach services on your host machine. For production, use actual hostnames or IPs.

### Step 3: Verify the deployment

```bash
# Check container is running
docker ps | grep smg

# Check health
curl http://localhost:30000/health

# Check workers
curl http://localhost:30000/workers
```

---

## Docker Compose

For more complex deployments, use Docker Compose.

### Step 1: Create docker-compose.yml

```yaml title="docker-compose.yml"
version: '3.8'

services:
  smg:
    image: lightseekorg/smg:latest
    container_name: smg
    ports:
      - "30000:30000"  # API
      - "29000:29000"  # Metrics
    command:
      - --worker-urls
      - http://worker:8000
      - --policy
      - cache_aware
      - --prometheus-port
      - "29000"
      - --log-level
      - info
    depends_on:
      - worker
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:30000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  worker:
    image: lightseekorg/smg:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command:
      - python
      - -m
      - sglang.launch_server
      - --model-path
      - meta-llama/Llama-3.1-8B-Instruct
      - --port
      - "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
```

### Step 2: Start the services

```bash
docker compose up -d
```

### Step 3: Check logs

```bash
# SMG logs
docker compose logs smg

# Worker logs
docker compose logs worker

# Follow logs
docker compose logs -f
```

---

## Environment Variables

Configure SMG using environment variables:

```yaml title="docker-compose.yml"
services:
  smg:
    image: lightseekorg/smg:latest
    environment:
      - RUST_LOG=info
      - HF_TOKEN=${HF_TOKEN}  # For private models
    # ...
```

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level (debug, info, warn, error) |
| `HF_TOKEN` | HuggingFace token for private models |

---

## Volume Mounts

Mount configuration files and certificates:

```yaml title="docker-compose.yml"
services:
  smg:
    volumes:
      - ./config:/etc/smg:ro
      - ./certs:/etc/certs:ro
      - ./logs:/var/log/smg
    command:
      - --tls-cert-path
      - /etc/certs/server.crt
      - --tls-key-path
      - /etc/certs/server.key
      # ...
```

---

## Multiple Workers

Scale to multiple workers:

```yaml title="docker-compose.yml"
services:
  smg:
    command:
      - --worker-urls
      - http://worker-1:8000
      - http://worker-2:8000
      - --policy
      - cache_aware

  worker-1:
    image: lightseekorg/smg:latest
    # ...

  worker-2:
    image: lightseekorg/smg:latest
    # ...
```

---

## Verification

Confirm your deployment is working:

```bash
# Health check
curl http://localhost:30000/health
# Expected: {"status": "ok"}

# List workers
curl http://localhost:30000/workers
# Expected: {"workers": [...], "total": 1}

# Test inference
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10
  }'
```

---

## Troubleshooting

??? question "Container exits immediately"

    Check logs for errors:
    ```bash
    docker logs smg
    ```

    Common causes:

    - Invalid worker URL
    - Port already in use
    - Missing required arguments

??? question "Cannot connect to worker"

    Verify network connectivity:
    ```bash
    # From inside the container
    docker exec smg curl http://worker:8000/health
    ```

    Ensure containers are on the same network:
    ```bash
    docker network ls
    docker network inspect <network_name>
    ```

??? question "Out of memory"

    Set resource limits:
    ```yaml
    services:
      smg:
        deploy:
          resources:
            limits:
              memory: 4G
    ```

---

## What's Next?

- [Deploy to Kubernetes](kubernetes.md) — Production deployment
- [Configure TLS](tls.md) — Secure communications
- [Monitor with Prometheus](../operations/monitoring.md) — Set up observability
