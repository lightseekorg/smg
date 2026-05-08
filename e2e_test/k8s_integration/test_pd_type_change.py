"""Integration test for PD mode pod type change during hostNetwork rollout.

Scenario: With hostNetwork, the pod IP = node IP. During a rolling update,
an old prefill pod is deleted and a new decode pod comes up on the same node
with the same IP but a new UID. The gateway must:

1. Remove the stale prefill worker (via watcher delete event or reconciliation)
2. Register the new decode worker
3. End up with the correct worker_type=decode, not the old prefill

This covers the UID-based eviction path in handle_pod_event (same name,
different UID) and the reconciliation diff (stale uid-A, missing uid-B).

Run with:
    cd e2e_test/k8s_integration
    source .venv/bin/activate
    pytest test_pd_type_change.py -v -s
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from .conftest import (
    KUBECTL_CONTEXT,
    NAMESPACE,
    RECONCILIATION_WAIT_SECS,
    _kubectl,
    _wait_for_deployment_ready,
    _wait_for_pod_ready,
)

logger = logging.getLogger(__name__)

MANIFESTS_DIR = Path(__file__).parent / "manifests"
_TRANSIENT_ERRORS = (httpx.HTTPError, ConnectionError, OSError)

PD_GATEWAY_HTTP_PORT = 30001


def _get_workers(gateway_url: str) -> dict:
    resp = httpx.get(f"{gateway_url}/workers", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "total" not in data:
        raise ValueError(f"/workers unexpected structure: {json.dumps(data)[:200]}")
    return data


def _get_worker_count(gateway_url: str) -> int:
    return _get_workers(gateway_url)["total"]


def _get_workers_by_type(gateway_url: str) -> dict[str, list[dict]]:
    """Return workers grouped by worker_type."""
    data = _get_workers(gateway_url)
    result: dict[str, list[dict]] = {}
    for w in data.get("workers", []):
        wtype = w.get("worker_type", "unknown")
        result.setdefault(wtype, []).append(w)
    return result


def _poll_until(predicate, description: str, timeout: int, interval: int = 5) -> bool:
    """Poll until predicate returns True, or raise TimeoutError."""
    deadline = time.time() + timeout
    last_error = None
    attempts = 0
    while time.time() < deadline:
        try:
            attempts += 1
            if predicate():
                logger.info("Condition met: %s (after %d attempts)", description, attempts)
                return True
        except _TRANSIENT_ERRORS as e:
            last_error = e
            logger.debug("Transient error on attempt %d: %s", attempts, e)
        time.sleep(interval)
    msg = f"Timeout waiting for: {description} (after {timeout}s, {attempts} attempts)"
    if last_error:
        msg += f" — last error: {last_error}"
    raise TimeoutError(msg)


def _deploy_pd_worker(name: str, role: str):
    """Deploy a fake worker pod with a role label for PD mode."""
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"role": role},
        },
        "spec": {
            "containers": [
                {
                    "name": "worker",
                    "image": "python:3.12-slim",
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["python3", "/app/fake_worker.py"],
                    "ports": [{"containerPort": 8000}],
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": 8000},
                        "initialDelaySeconds": 2,
                        "periodSeconds": 3,
                    },
                    "volumeMounts": [{"name": "app", "mountPath": "/app"}],
                }
            ],
            "volumes": [{"name": "app", "configMap": {"name": "fake-worker-script"}}],
        },
    }
    subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=json.dumps(pod_manifest),
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Deployed PD pod %s with role=%s", name, role)


def _safe_delete_pod(name: str):
    try:
        _kubectl(
            "delete",
            "pod",
            name,
            "-n",
            NAMESPACE,
            "--ignore-not-found",
            "--force",
            "--grace-period=0",
        )
    except Exception as e:
        logger.warning("Cleanup failed for pod %s: %s", name, e)


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: int = 15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"port-forward exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


@pytest.fixture(scope="module")
def pd_gateway():
    """Deploy the PD-mode gateway and set up port-forwarding."""
    _kubectl("apply", "-f", str(MANIFESTS_DIR / "gateway-pd.yaml"))
    _wait_for_deployment_ready("smg-gateway-pd")

    cmd = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "port-forward",
        "svc/smg-gateway-pd",
        f"{PD_GATEWAY_HTTP_PORT}:{PD_GATEWAY_HTTP_PORT}",
        "-n",
        NAMESPACE,
    ]
    pf = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _wait_for_port(PD_GATEWAY_HTTP_PORT, pf)

    yield f"http://127.0.0.1:{PD_GATEWAY_HTTP_PORT}"

    pf.terminate()
    pf.wait()
    _kubectl(
        "delete",
        "-f",
        str(MANIFESTS_DIR / "gateway-pd.yaml"),
        "--ignore-not-found",
        check=False,
    )


class TestPDRolloutTypeChange:
    """Test that the gateway correctly transitions worker type when a pod
    is deleted and recreated with a different role (prefill -> decode).

    This simulates a hostNetwork rolling update where a node changes role.
    The new pod has the same name and IP but a different UID and labels.
    """

    def test_prefill_discovered_as_prefill(self, pd_gateway):
        """Baseline: a prefill pod is correctly discovered as prefill type."""
        pod_name = "test-pd-baseline"

        try:
            _deploy_pd_worker(pod_name, role="prefill")
            _wait_for_pod_ready(pod_name)

            _poll_until(
                lambda: _get_worker_count(pd_gateway) >= 1,
                "prefill worker discovered",
                timeout=30,
                interval=3,
            )

            by_type = _get_workers_by_type(pd_gateway)
            logger.info("Workers by type: %s", json.dumps(by_type, indent=2))
            assert "prefill" in by_type, f"Expected prefill, got: {list(by_type.keys())}"

        finally:
            _safe_delete_pod(pod_name)
            _poll_until(
                lambda: _get_worker_count(pd_gateway) == 0,
                "cleanup: worker count back to 0",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

    def test_delete_prefill_recreate_as_decode(self, pd_gateway):
        """Delete a prefill pod and recreate with the same name as decode.

        This is the realistic rollout path: the old pod is deleted (new UID),
        and a new pod with a different role comes up. The gateway should
        transition the worker from prefill to decode.
        """
        pod_name = "test-pd-rollout"

        try:
            # Step 1: Deploy as prefill
            _deploy_pd_worker(pod_name, role="prefill")
            _wait_for_pod_ready(pod_name)

            _poll_until(
                lambda: _get_worker_count(pd_gateway) >= 1,
                "prefill worker discovered",
                timeout=30,
                interval=3,
            )

            by_type = _get_workers_by_type(pd_gateway)
            logger.info("Before rollout: %s", json.dumps(by_type, indent=2))
            assert "prefill" in by_type

            # Capture the prefill worker's URL for later comparison
            prefill_url = by_type["prefill"][0]["url"]
            logger.info("Prefill worker URL: %s", prefill_url)

            # Step 2: Delete the prefill pod (simulates rollout termination)
            _kubectl(
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "--force",
                "--grace-period=0",
            )

            # Step 3: Wait for the gateway to remove the stale prefill worker
            _poll_until(
                lambda: _get_worker_count(pd_gateway) == 0,
                "prefill worker removed",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            # Step 4: Recreate with same name as decode (new UID!)
            _deploy_pd_worker(pod_name, role="decode")
            _wait_for_pod_ready(pod_name)

            # Step 5: Verify the gateway discovers it as decode
            _poll_until(
                lambda: _get_worker_count(pd_gateway) >= 1,
                "decode worker discovered after rollout",
                timeout=30,
                interval=3,
            )

            by_type = _get_workers_by_type(pd_gateway)
            logger.info("After rollout: %s", json.dumps(by_type, indent=2))

            assert "decode" in by_type, (
                f"Expected decode worker after rollout, got: {list(by_type.keys())}"
            )
            assert "prefill" not in by_type, "Stale prefill worker persists after rollout"

        finally:
            _safe_delete_pod(pod_name)

    def test_simultaneous_prefill_and_decode(self, pd_gateway):
        """Both prefill and decode pods exist at the same time.

        During a rolling update there may be a brief overlap where both
        old and new pods are running. The gateway should track both.
        """
        prefill_pod = "test-pd-both-p"
        decode_pod = "test-pd-both-d"

        try:
            _deploy_pd_worker(prefill_pod, role="prefill")
            _deploy_pd_worker(decode_pod, role="decode")
            _wait_for_pod_ready(prefill_pod)
            _wait_for_pod_ready(decode_pod)

            _poll_until(
                lambda: _get_worker_count(pd_gateway) >= 2,
                "both prefill and decode workers discovered",
                timeout=30,
                interval=3,
            )

            by_type = _get_workers_by_type(pd_gateway)
            logger.info("Both pods: %s", json.dumps(by_type, indent=2))

            assert "prefill" in by_type, f"Missing prefill, got: {list(by_type.keys())}"
            assert "decode" in by_type, f"Missing decode, got: {list(by_type.keys())}"

            # Now remove prefill, only decode should remain
            _safe_delete_pod(prefill_pod)

            _poll_until(
                lambda: _get_worker_count(pd_gateway) == 1,
                "only decode worker remains",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            by_type = _get_workers_by_type(pd_gateway)
            logger.info("After prefill removed: %s", json.dumps(by_type, indent=2))

            assert "decode" in by_type, "Decode worker should still exist"
            assert "prefill" not in by_type, "Prefill worker should be gone"

        finally:
            _safe_delete_pod(prefill_pod)
            _safe_delete_pod(decode_pod)
