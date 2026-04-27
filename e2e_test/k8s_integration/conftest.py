"""Pytest configuration for K8s integration tests.

These tests require:
  - A kind cluster named 'smg-test'
  - The smg-gateway:test image loaded into kind
  - kubectl configured to use the kind-smg-test context

Setup: ./e2e_test/k8s_integration/setup.sh
Teardown: ./e2e_test/k8s_integration/setup.sh teardown
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

NAMESPACE = "smg-test"
MANIFESTS_DIR = Path(__file__).parent / "manifests"
FAKE_WORKER_SCRIPT = Path(__file__).parent / "fake_worker.py"
KUBECTL_CONTEXT = "kind-smg-test"

# Reconciliation interval: 60s when using --service-discovery CLI flag
# (see model_gateway/src/main.rs:1275). Note: DiscoveryConfig::default()
# uses 120s; the 60s value is CLI-specific.
RECONCILIATION_INTERVAL_SECS = 60
RECONCILIATION_WAIT_SECS = RECONCILIATION_INTERVAL_SECS + 30


def _kubectl(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "--context", KUBECTL_CONTEXT, *args]
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def _kubectl_json(*args: str) -> dict:
    result = _kubectl(*args, "-o", "json")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse kubectl JSON output for args={args!r}. "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        ) from e


def _wait_for_pod_ready(name: str, namespace: str = NAMESPACE, timeout: int = 120):
    """Wait until a pod is Ready."""
    logger.info("Waiting for pod %s to be ready (timeout=%ds)", name, timeout)
    _kubectl(
        "wait", "--for=condition=Ready", f"pod/{name}", "-n", namespace, f"--timeout={timeout}s"
    )


def _wait_for_deployment_ready(name: str, namespace: str = NAMESPACE, timeout: int = 180):
    """Wait until a deployment has all replicas available."""
    logger.info("Waiting for deployment %s to be ready (timeout=%ds)", name, timeout)
    _kubectl("rollout", "status", f"deployment/{name}", "-n", namespace, f"--timeout={timeout}s")


def _get_gateway_url() -> str:
    """Return the gateway URL, assuming port-forward is active on localhost:30000."""
    return "http://127.0.0.1:30000"


def _get_metrics_url() -> str:
    """Return the metrics URL, assuming port-forward is active on localhost:29000."""
    return "http://127.0.0.1:29000"


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: int = 15):
    """Poll until a TCP connection to localhost:port succeeds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"port-forward process exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


def _port_forward_start(
    namespace: str, service: str, local_port: int, remote_port: int
) -> subprocess.Popen:
    """Start kubectl port-forward and verify the port is reachable."""
    cmd = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "port-forward",
        f"svc/{service}",
        f"{local_port}:{remote_port}",
        "-n",
        namespace,
    ]
    logger.info("Starting port-forward: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _wait_for_port(local_port, proc)
    return proc


@pytest.fixture(scope="session")
def k8s_cluster():
    """Ensure the kind cluster exists and is reachable."""
    result = subprocess.run(
        ["kind", "get", "clusters"],
        capture_output=True,
        text=True,
        check=True,
    )
    if "smg-test" not in result.stdout:
        pytest.skip("kind cluster 'smg-test' not found — run setup first")

    # Verify kubectl connectivity
    _kubectl("cluster-info")
    return True


@pytest.fixture(scope="session")
def deploy_base(k8s_cluster):
    """Ensure namespace, RBAC, configmap, and gateway are deployed.

    Does NOT tear down at session end — use setup.sh teardown for that.
    This allows running pytest multiple times without full re-setup.
    """
    # Create namespace (apply is idempotent — succeeds if already exists)
    _kubectl("apply", "-f", str(MANIFESTS_DIR / "namespace.yaml"))

    # Create/update the fake-worker script as a ConfigMap
    cm_result = _kubectl(
        "create",
        "configmap",
        "fake-worker-script",
        f"--from-file=fake_worker.py={FAKE_WORKER_SCRIPT}",
        "-n",
        NAMESPACE,
        "--dry-run=client",
        "-o",
        "yaml",
    )
    _apply_from_stdin(cm_result.stdout)

    # Apply RBAC
    _kubectl("apply", "-f", str(MANIFESTS_DIR / "rbac.yaml"))

    # Apply gateway deployment
    _kubectl("apply", "-f", str(MANIFESTS_DIR / "gateway.yaml"))

    # Wait for gateway to be ready
    _wait_for_deployment_ready("smg-gateway")

    # Clean up any residual test pods from previous runs
    result = _kubectl(
        "get",
        "pods",
        "-n",
        NAMESPACE,
        "-l",
        "app=fake-worker",
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        for pod_name in result.stdout.strip().split():
            _kubectl(
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "--force",
                "--grace-period=0",
                "--ignore-not-found",
            )
        # Wait a bit for cleanup
        time.sleep(5)

    yield


def _apply_from_stdin(yaml_content: str):
    """Apply a YAML manifest from stdin."""
    proc = subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=yaml_content,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc


def _cleanup_port_forward(name: str, pf: subprocess.Popen):
    """Terminate a port-forward process and log any errors."""
    try:
        pf.terminate()
        pf.wait(timeout=10)
    except Exception as e:
        logger.warning("Error cleaning up %s port-forward: %s", name, e)
    if pf.returncode and pf.returncode != -15:  # -15 = SIGTERM
        stderr = pf.stderr.read().decode() if pf.stderr else ""
        if stderr:
            logger.warning("Port-forward %s exited with code %d: %s", name, pf.returncode, stderr)


@pytest.fixture(scope="session")
def gateway_port_forward(deploy_base):
    """Set up port-forwarding to the gateway service."""
    pf_http = _port_forward_start(NAMESPACE, "smg-gateway", 30000, 30000)
    try:
        pf_metrics = _port_forward_start(NAMESPACE, "smg-gateway", 29000, 29000)
    except Exception:
        pf_http.terminate()
        pf_http.wait()
        raise
    yield _get_gateway_url(), _get_metrics_url()
    _cleanup_port_forward("http", pf_http)
    _cleanup_port_forward("metrics", pf_metrics)
