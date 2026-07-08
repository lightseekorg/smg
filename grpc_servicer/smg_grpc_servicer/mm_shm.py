"""Shared multimodal SHM tensor-transport helpers (engine-neutral).

Reads a ``TensorData`` payload that carries its raw little-endian bytes either
inline in the gRPC message or via a same-host ``/dev/shm`` handle, and reports
this process's ``/dev/shm`` namespace identity. Used by the vLLM and TokenSpeed
gRPC servicers; their ``TensorData``/``ShmHandle`` messages share the ``payload``
oneof shape (``inline`` | ``shm`` | ``remote``) via ``common.proto``.
"""

import os

# Unlink each /dev/shm segment right after the worker reads it (default on) so
# same-host SHM tensors don't accumulate. Disable with
# TOKENSPEED_UNLINK_MM_SHM_AFTER_READ=0 (e.g. for debugging).
UNLINK_MM_SHM_AFTER_READ = os.getenv("TOKENSPEED_UNLINK_MM_SHM_AFTER_READ", "1").lower() not in (
    "0",
    "false",
    "no",
)


def tensor_payload_bytes(tensor_data) -> bytes:
    """Raw bytes of a ``TensorData``, from whichever payload transport it carries."""
    payload = tensor_data.WhichOneof("payload")
    if payload == "inline":
        return bytes(tensor_data.inline)
    if payload == "shm":
        return tensor_payload_bytes_from_shm(tensor_data.shm)
    if payload == "remote":
        raise ValueError("TensorData.remote payload is not implemented yet")
    raise ValueError("TensorData payload is required")


def tensor_payload_bytes_from_shm(shm_handle) -> bytes:
    """Read ``nbytes`` from ``/dev/shm/<name>`` at ``offset`` (unlinking after read
    when enabled)."""
    name = validated_shm_name(shm_handle.name)

    path = os.path.join("/dev/shm", name)
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        raw = os.pread(fd, int(shm_handle.nbytes), int(shm_handle.offset))
    finally:
        if fd is not None:
            os.close(fd)
        if fd is not None and UNLINK_MM_SHM_AFTER_READ:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    if len(raw) != int(shm_handle.nbytes):
        raise ValueError(
            f"TensorData.shm byte length mismatch for name={shm_handle.name!r}: "
            f"expected {int(shm_handle.nbytes)}, got {len(raw)}"
        )
    return raw


def validated_shm_name(name: str) -> str:
    """Reject path-traversal / absolute / empty SHM names before opening."""
    name = name.lstrip("/")
    if not name or "/" in name or name in (".", "..") or "\x00" in name:
        raise ValueError(f"Invalid TensorData.shm name: {name!r}")
    return name


def shm_namespace_id() -> str:
    """Identity of this process's ``/dev/shm`` tmpfs: ``<boot_id>:<st_dev>``.

    ``boot_id`` (``/proc/sys/kernel/random/boot_id``) is not namespaced, so it
    pins the host; ``st_dev`` is the tmpfs superblock device backing
    ``/dev/shm``. Two processes share ``/dev/shm`` iff both match -- including
    separate containers sharing it via ``--ipc``/bind-mount, where mount
    namespaces differ but the underlying superblock (``st_dev``) is the same. The
    router compares this token to its own to decide the SHM tensor transport.
    Empty string if it can't be determined.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
            boot_id = f.read().strip()
        shm_dev = os.stat("/dev/shm").st_dev
        return f"{boot_id}:{shm_dev}"
    except OSError:
        return ""
