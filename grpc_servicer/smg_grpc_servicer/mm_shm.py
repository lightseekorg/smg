"""Shared multimodal ``/dev/shm`` tensor-transport helpers for engine servicers.

The gateway writes large multimodal tensor payloads to ``/dev/shm`` and sends
only a ``ShmHandle`` (name + offset + nbytes) in the proto; servicers read the
bytes back here. Engine-agnostic: no vLLM/TokenSpeed imports, so it stays unit
testable. The handle argument is duck-typed (``.name``/``.offset``/``.nbytes``).
"""

from __future__ import annotations

import os

DEFAULT_SHM_DIR = "/dev/shm"


def _unlink_after_read() -> bool:
    # Unlink each payload after reading so gateway-produced segments are
    # reclaimed once consumed. ``SMG_UNLINK_MM_SHM_AFTER_READ`` is the current
    # knob; ``TOKENSPEED_UNLINK_MM_SHM_AFTER_READ`` is honored for back-compat.
    for name in ("SMG_UNLINK_MM_SHM_AFTER_READ", "TOKENSPEED_UNLINK_MM_SHM_AFTER_READ"):
        value = os.getenv(name)
        if value is not None:
            return value.lower() not in ("0", "false", "no")
    return True


UNLINK_MM_SHM_AFTER_READ = _unlink_after_read()

# Names the gateway is allowed to produce. Restricting to these prevents a
# malformed/compromised request from reading or unlinking an arbitrary
# /dev/shm entry. `smg-tokenspeed-` is the legacy prefix, kept for compat.
_ALLOWED_SHM_PREFIXES = ("smg-mm-", "smg-tokenspeed-")


def validated_shm_name(name: str) -> str:
    """Reject path traversal / absolute / out-of-namespace names before touching the filesystem."""
    name = name.lstrip("/")
    if not name or "/" in name or name in (".", "..") or "\x00" in name:
        raise ValueError(f"Invalid shm tensor name: {name!r}")
    if not name.startswith(_ALLOWED_SHM_PREFIXES):
        raise ValueError(f"shm tensor name outside allowed namespace: {name!r}")
    return name


def tensor_payload_bytes_from_shm(shm_handle, shm_dir: str = DEFAULT_SHM_DIR) -> bytes:
    """Read a tensor payload the gateway wrote to ``shm_dir`` for ``shm_handle``."""
    name = validated_shm_name(shm_handle.name)
    path = os.path.join(shm_dir, name)
    fd = None
    try:
        # O_NOFOLLOW: /dev/shm is world-writable; refuse to follow a symlink planted
        # at the validated name (would otherwise read/unlink an arbitrary file).
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        raw = os.pread(fd, int(shm_handle.nbytes), int(shm_handle.offset))
    finally:
        if fd is not None:
            os.close(fd)
            if UNLINK_MM_SHM_AFTER_READ:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
    if len(raw) != int(shm_handle.nbytes):
        raise ValueError(
            f"shm tensor byte length mismatch for name={shm_handle.name!r}: "
            f"expected {int(shm_handle.nbytes)}, got {len(raw)}"
        )
    return raw


def shm_namespace_id() -> str:
    """Identity of this process's ``/dev/shm`` tmpfs: ``<boot_id>:<st_dev>``.

    ``boot_id`` pins the host; ``st_dev`` is the tmpfs superblock device backing
    ``/dev/shm``. Two processes share ``/dev/shm`` iff both match. The router
    compares this to its own to decide the SHM tensor transport under ``auto``.
    Empty string if it can't be determined.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
            boot_id = f.read().strip()
        shm_dev = os.stat(DEFAULT_SHM_DIR).st_dev
        return f"{boot_id}:{shm_dev}"
    except OSError:
        return ""
