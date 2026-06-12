"""Process management utilities for E2E tests."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Port reservation utilities
# ---------------------------------------------------------------------------

# Port reservation to prevent the OS from returning the same port
# for sequential get_open_port() calls before the port is actually bound.
_reserved_ports: set[int] = set()


def get_open_port(max_attempts: int = 10) -> int:
    """Get an available port with reservation tracking.

    Finds an available port from the kernel and reserves it in our tracking set
    to prevent the OS from returning the same port on subsequent calls.

    Args:
        max_attempts: Maximum attempts to find an unreserved port.

    Returns:
        An available port number that is reserved until release_port() is called.

    Raises:
        RuntimeError: If unable to find an available port after max_attempts.
    """
    for attempt in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        if port not in _reserved_ports:
            _reserved_ports.add(port)
            logger.debug("Reserved port %d (attempt %d)", port, attempt + 1)
            return port

        logger.debug(
            "Port %d already reserved, retrying (attempt %d/%d)",
            port,
            attempt + 1,
            max_attempts,
        )

    raise RuntimeError(f"Failed to find available port after {max_attempts} attempts")


def release_port(port: int) -> None:
    """Release a reserved port back to the available pool.

    Should be called when the process using the port has terminated.

    Args:
        port: The port number to release.
    """
    _reserved_ports.discard(port)
    logger.debug("Released port %d", port)


def kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Kill a process and all its children.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default: SIGTERM)
    """
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        parent.send_signal(sig)
    except ImportError:
        # Fallback if psutil not available
        os.kill(pid, sig)
    except Exception as e:
        logger.warning("Failed to kill process tree for PID %d: %s", pid, e)


def terminate_process(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Gracefully terminate a process, kill if needed.

    Args:
        proc: Process to terminate
        timeout: Seconds to wait before force-killing
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def wait_for_health(
    url: str,
    timeout: float = 60,
    api_key: str | None = None,
    check_interval: float = 1.0,
) -> None:
    """Wait for a server's /health endpoint to return 200.

    Args:
        url: Base URL of the server
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
        check_interval: Seconds between health checks
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{url}/health", headers=headers, timeout=5)
                if resp.status_code == 200:
                    logger.info("Service healthy at %s", url)
                    return
            except requests.RequestException:
                pass
            time.sleep(check_interval)

    raise TimeoutError(f"Server at {url} did not become healthy within {timeout}s")


def wait_for_workers_ready(
    router_url: str,
    expected_workers: int,
    timeout: float = 300,
    api_key: str | None = None,
) -> None:
    """Wait for router to have all workers connected.

    Args:
        router_url: Base URL of the router
        expected_workers: Number of workers to wait for
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{router_url}/workers", headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    total = data.get("total", len(data.get("workers", [])))
                    if total >= expected_workers:
                        logger.info(
                            "All %d workers connected after %.1fs",
                            expected_workers,
                            time.perf_counter() - start,
                        )
                        return
            except requests.RequestException:
                pass
            time.sleep(2)

    raise TimeoutError(
        f"Router at {router_url} did not get {expected_workers} workers within {timeout}s"
    )


# RoCE GIDs are programmed per network namespace from the bound netdev's IPs.
# A pod with its own netns (no hostNetwork / SR-IOV) sees the shared host mlx5
# devices as PORT_ACTIVE but with an all-zero GID table, so ibv_query_gid()
# returns ENODATA and NIXL(UCX)/Mooncake RDMA init dies during worker startup.
# Treat such a device as unusable so PD falls back to TCP: Mooncake auto-falls
# back when no --disaggregation-ib-device is forced, and NIXL is pinned off
# RDMA via UCX_TLS (see Worker._build_env).
_IB_SYSFS_ROOT = "/sys/class/infiniband"
_ZERO_GID = "0000:0000:0000:0000:0000:0000:0000:0000"


def _port_has_usable_gid(port_dir: str) -> bool:
    """True if an RDMA port exposes at least one non-zero GID in this netns."""
    gids_dir = os.path.join(port_dir, "gids")
    try:
        names = os.listdir(gids_dir)
    except OSError:
        return False
    for name in names:
        try:
            with open(os.path.join(gids_dir, name), encoding="ascii") as fh:
                gid = fh.read().strip()
        except OSError:
            continue
        if gid and gid != _ZERO_GID:
            return True
    return False


def detect_ib_device() -> str | None:
    """Detect an RDMA device usable for PD KV transfer in this netns.

    Returns a device name (e.g. "mlx5_0") only when it has an ACTIVE port whose
    GID table is populated. Returns None when no device is usable — e.g. on CI
    runner pods where the shared NICs are ACTIVE but expose an empty GID table
    inside the pod network namespace — so callers skip RDMA and let PD fall
    back to TCP/cuda_ipc instead of dying in transfer-engine init.
    """
    try:
        devices = sorted(os.listdir(_IB_SYSFS_ROOT))
    except OSError:
        return None

    for dev in devices:
        ports_dir = os.path.join(_IB_SYSFS_ROOT, dev, "ports")
        try:
            ports = sorted(os.listdir(ports_dir))
        except OSError:
            continue
        for port in ports:
            port_dir = os.path.join(ports_dir, port)
            try:
                with open(os.path.join(port_dir, "state"), encoding="ascii") as fh:
                    state = fh.read()
            except OSError:
                continue
            if "ACTIVE" not in state:
                continue
            if _port_has_usable_gid(port_dir):
                logger.info("Detected usable RDMA device: %s (port %s)", dev, port)
                return dev

    logger.info(
        "No RDMA device with a usable GID in this netns; PD KV transfer will "
        "fall back to TCP/cuda_ipc instead of RDMA"
    )
    return None
