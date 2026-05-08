"""Integration tests for multi-pool label-selector service discovery.

Covers the feature that lets one gateway watch engines carrying different
label sets in the same namespace — for example, a plain Deployment labeled
``engine=sglang, role=worker`` alongside an LWS-deployed engine labeled
``engine=deepseek-v4, leaderworkerset.sigs.k8s.io/name=deepseek-v4`` — which
a single apiserver ``labelSelector`` cannot express.

The gateway under test (``manifests/gateway-multi-selector.yaml``) is launched
with two ``--selector-pool`` flags. A pod is included if it matches *any*
pool, and each pool still AND's its own labels. These tests assert:

* Pool A pods are discovered.
* Pool B pods (with both AND'd labels) are discovered.
* Pods that only partially match a pool are rejected.
* Pods unrelated to any pool are rejected.
* Flipping an unmatched pod's labels in place (via ``kubectl label``) causes
  the watcher to pick it up without restarting the gateway — proving the
  client-side ``should_include`` filter re-evaluates on label updates.

Run with::

    cd e2e_test/k8s_integration
    source .venv/bin/activate
    pytest test_multi_selector_discovery.py -v -s
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
    _kubectl,
    _wait_for_deployment_ready,
    _wait_for_pod_ready,
)

logger = logging.getLogger(__name__)

MANIFESTS_DIR = Path(__file__).parent / "manifests"
_TRANSIENT_ERRORS = (httpx.HTTPError, ConnectionError, OSError)

MULTI_GATEWAY_HTTP_PORT = 30002

# Pool A in gateway-multi-selector.yaml
POOL_A_LABELS = {"engine": "sglang", "role": "worker"}
# Pool B in gateway-multi-selector.yaml — both labels are required (AND).
POOL_B_LABELS = {
    "engine": "deepseek-v4",
    "leaderworkerset.sigs.k8s.io/name": "deepseek-v4",
}


def _get_workers(gateway_url: str) -> dict:
    resp = httpx.get(f"{gateway_url}/workers", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "total" not in data:
        raise ValueError(f"/workers unexpected structure: {json.dumps(data)[:200]}")
    return data


def _get_worker_count(gateway_url: str) -> int:
    return _get_workers(gateway_url)["total"]


def _get_worker_ips(gateway_url: str) -> set[str]:
    """Return the set of pod IPs currently registered as workers.

    Worker URLs are of the form ``http://<pod-ip>:8000``; extracting the IP
    lets tests match workers back to the pods they deployed.
    """
    ips: set[str] = set()
    for w in _get_workers(gateway_url).get("workers", []):
        url = w.get("url", "")
        # Strip scheme and port
        if "://" in url:
            url = url.split("://", 1)[1]
        host = url.split(":", 1)[0]
        if host:
            ips.add(host)
    return ips


def _poll_until(predicate, description: str, timeout: int, interval: float = 2.0) -> bool:
    """Poll until ``predicate`` returns truthy, or raise ``TimeoutError``.

    Transient HTTP/network errors are retried. Other exceptions propagate.
    """
    deadline = time.time() + timeout
    last_error: Exception | None = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
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


def _deploy_labeled_worker(name: str, labels: dict[str, str]) -> str:
    """Deploy a fake worker pod with arbitrary labels. Returns the pod IP."""
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": labels,
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
    _wait_for_pod_ready(name)
    pod_json = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pod",
            name,
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pod = json.loads(pod_json.stdout)
    pod_ip = pod.get("status", {}).get("podIP")
    if not pod_ip:
        raise RuntimeError(f"Pod {name} has no podIP after becoming Ready")
    logger.info("Deployed pod %s with labels %s (IP %s)", name, labels, pod_ip)
    return pod_ip


def _set_pod_labels(name: str, labels: dict[str, str]) -> None:
    """Overwrite the pod's label set via `kubectl label --overwrite`.

    This triggers an UPDATE event on the watcher without recreating the pod,
    exercising the client-side ``should_include`` filter on label changes.
    """
    # Remove any existing labels we'd conflict with; then set the new ones.
    args = [
        "label",
        "pod",
        name,
        "-n",
        NAMESPACE,
        "--overwrite",
    ]
    for k, v in labels.items():
        args.append(f"{k}={v}")
    _kubectl(*args)


def _safe_delete_pod(name: str) -> None:
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


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: int = 15) -> None:
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
def multi_gateway(deploy_base):
    """Deploy the multi-selector gateway and set up port-forwarding.

    Mirrors the ``pd_gateway`` fixture in ``test_pd_type_change.py``: deploy
    the manifest, wait for rollout, port-forward the HTTP service, then tear
    everything down on fixture teardown.
    """
    manifest = MANIFESTS_DIR / "gateway-multi-selector.yaml"
    _kubectl("apply", "-f", str(manifest))
    _wait_for_deployment_ready("smg-gateway-multi")

    cmd = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "port-forward",
        "svc/smg-gateway-multi",
        f"{MULTI_GATEWAY_HTTP_PORT}:{MULTI_GATEWAY_HTTP_PORT}",
        "-n",
        NAMESPACE,
    ]
    pf = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _wait_for_port(MULTI_GATEWAY_HTTP_PORT, pf)

    yield f"http://127.0.0.1:{MULTI_GATEWAY_HTTP_PORT}"

    pf.terminate()
    pf.wait()
    _kubectl("delete", "-f", str(manifest), "--ignore-not-found", check=False)


class TestMultiSelectorDiscovery:
    """Verify that two disjoint ``--selector-pool`` values discover the union."""

    def test_pool_a_pod_is_discovered(self, multi_gateway):
        """A pod matching pool A (``engine=sglang,role=worker``) is picked up."""
        pod_name = "multi-pool-a"
        try:
            pod_ip = _deploy_labeled_worker(pod_name, POOL_A_LABELS)
            _poll_until(
                lambda: pod_ip in _get_worker_ips(multi_gateway),
                f"pool-A pod {pod_name} ({pod_ip}) registered as worker",
                timeout=30,
            )
        finally:
            _safe_delete_pod(pod_name)

    def test_pool_b_pod_with_both_labels_is_discovered(self, multi_gateway):
        """A pod matching pool B (both LWS-style labels) is picked up."""
        pod_name = "multi-pool-b"
        try:
            pod_ip = _deploy_labeled_worker(pod_name, POOL_B_LABELS)
            _poll_until(
                lambda: pod_ip in _get_worker_ips(multi_gateway),
                f"pool-B pod {pod_name} ({pod_ip}) registered as worker",
                timeout=30,
            )
        finally:
            _safe_delete_pod(pod_name)

    def test_partial_pool_b_match_is_rejected(self, multi_gateway):
        """A pod with only one of pool B's labels must NOT be discovered.

        Each pool AND's its keys, so dropping ``leaderworkerset.sigs.k8s.io/name``
        leaves only ``engine=deepseek-v4``, which matches neither pool A nor
        pool B.
        """
        pod_name = "multi-partial-b"
        try:
            pod_ip = _deploy_labeled_worker(pod_name, {"engine": "deepseek-v4"})
            # Give the watcher time to observe the pod and decide.
            time.sleep(10)
            ips = _get_worker_ips(multi_gateway)
            assert pod_ip not in ips, (
                f"Pod {pod_name} ({pod_ip}) was discovered despite only "
                f"partially matching pool B. Current worker IPs: {sorted(ips)}"
            )
        finally:
            _safe_delete_pod(pod_name)

    def test_unrelated_pod_is_rejected(self, multi_gateway):
        """A pod whose labels match neither pool must NOT be discovered."""
        pod_name = "multi-unrelated"
        try:
            pod_ip = _deploy_labeled_worker(pod_name, {"engine": "unrelated", "role": "worker"})
            time.sleep(10)
            ips = _get_worker_ips(multi_gateway)
            assert pod_ip not in ips, (
                f"Pod {pod_name} ({pod_ip}) was discovered despite matching "
                f"no pool. Current worker IPs: {sorted(ips)}"
            )
        finally:
            _safe_delete_pod(pod_name)

    def test_union_of_pools_yields_both_pods(self, multi_gateway):
        """Deploy one pod per pool; both should appear in ``/workers``."""
        pod_a = "multi-union-a"
        pod_b = "multi-union-b"
        try:
            ip_a = _deploy_labeled_worker(pod_a, POOL_A_LABELS)
            ip_b = _deploy_labeled_worker(pod_b, POOL_B_LABELS)
            _poll_until(
                lambda: {ip_a, ip_b}.issubset(_get_worker_ips(multi_gateway)),
                f"both pods registered: pool-A ({ip_a}) + pool-B ({ip_b})",
                timeout=30,
            )
        finally:
            _safe_delete_pod(pod_a)
            _safe_delete_pod(pod_b)

    def test_label_flip_triggers_inclusion(self, multi_gateway):
        """Relabeling an unmatched pod into pool A should make it discovered.

        Proves the watcher re-evaluates ``should_include`` on pod UPDATE
        events, not just on ADD. This is the code path that matters when an
        engine's label set is adjusted in place (e.g. rollout annotation
        changes) rather than recreated.
        """
        pod_name = "multi-flip"
        try:
            pod_ip = _deploy_labeled_worker(pod_name, {"engine": "standby", "role": "worker"})
            # Not in any pool yet.
            time.sleep(5)
            assert pod_ip not in _get_worker_ips(multi_gateway)

            # Flip into pool A.
            _set_pod_labels(pod_name, POOL_A_LABELS)
            _poll_until(
                lambda: pod_ip in _get_worker_ips(multi_gateway),
                f"relabeled pod {pod_name} ({pod_ip}) becomes a worker",
                timeout=30,
            )
        finally:
            _safe_delete_pod(pod_name)
