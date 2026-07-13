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


def detect_ib_device() -> str | None:
    """Detect the first RDMA device with an active port (e.g., "mlx5_0").

    Reads ``/sys/class/infiniband`` directly so it works for both InfiniBand and
    RoCE and does NOT depend on the ``ibv_devinfo`` CLI, which isn't installed on
    every GPU runner (the tokenspeed image ships libibverbs but not the utils).
    Without a device the mooncake transfer engine enumerates every NIC on the
    node and hangs, so returning None here must be a genuine "no RDMA" signal,
    not just "the CLI is missing". Falls back to ``ibv_devinfo`` if sysfs is
    absent.

    Returns:
        Device name if found (e.g., "mlx5_0"), None otherwise.
    """
    ib_root = "/sys/class/infiniband"
    if os.path.isdir(ib_root):

        def _dev_key(name: str) -> tuple[str, int]:
            # Numeric sort so mlx5_2 precedes mlx5_10 (lexical order would not).
            head, _, tail = name.rpartition("_")
            return (head, int(tail)) if tail.isdigit() else (name, 0)

        for dev in sorted(os.listdir(ib_root), key=_dev_key):
            ports = os.path.join(ib_root, dev, "ports")
            if not os.path.isdir(ports):
                continue
            for port in sorted(os.listdir(ports)):
                try:
                    with open(os.path.join(ports, port, "state")) as f:
                        # e.g. "4: ACTIVE"
                        if "ACTIVE" in f.read():
                            logger.info("Detected IB device: %s (port %s)", dev, port)
                            return dev
                except OSError:
                    continue

    # Fallback: the ibv_devinfo CLI, if the sysfs tree wasn't available.
    try:
        subprocess.run(
            ["ibv_devinfo", "-l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("detect_ib_device: no active RDMA device (sysfs empty, ibv_devinfo absent)")
        return None

    for i in range(12):
        dev = f"mlx5_{i}"
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and "state:" in res.stdout:
                for line in res.stdout.splitlines():
                    if "state:" in line and "PORT_ACTIVE" in line:
                        logger.info("Detected IB device: %s", dev)
                        return dev
        except Exception:
            pass
    logger.warning("detect_ib_device: no active RDMA device found")
    return None
