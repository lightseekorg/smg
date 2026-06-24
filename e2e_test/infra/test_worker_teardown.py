"""Regression test for Worker.stop() reaping orphaned children.

sglang forks a GPU-holding scheduler subprocess that shares the worker's
process group. If the parent crashes first, stop() must still kill the group
— otherwise the child keeps the GPU pinned and the next launch OOMs.
POSIX-only (process groups).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from e2e_test.infra.worker import Worker

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_reaps_orphaned_child_after_parent_exits() -> None:
    # Leader (own session) spawns a long-lived child, then exits immediately —
    # the exact orphan scenario behind the e2e-1gpu-responses OOM.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, subprocess;"
            "c = subprocess.Popen(['sleep', '300']);"
            "print(c.pid, flush=True);"
            "os._exit(0)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    child_pid = int(proc.stdout.readline())
    worker = Worker(model_id="test/dummy", engine="sglang", port=0, gpu_ids=[0])
    worker.process = proc
    worker._pgid = proc.pid  # start_new_session ⇒ pgid == pid

    try:
        while proc.poll() is None:
            time.sleep(0.02)
        assert _alive(child_pid), "child should outlive the parent (orphan setup)"

        worker.stop()

        deadline = time.perf_counter() + 5
        while _alive(child_pid) and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert not _alive(child_pid), "orphaned child survived stop() — GPU would stay pinned"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def test_stop_is_safe_when_never_started() -> None:
    Worker(model_id="test/dummy", engine="sglang", port=0, gpu_ids=[0]).stop()
