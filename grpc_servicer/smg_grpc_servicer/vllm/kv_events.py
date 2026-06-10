"""vLLM KV-cache-event → proto conversion and ZMQ streaming.

Engine-free: imports only stdlib + the generated proto. vLLM event objects are
dispatched by class name (BlockStored / BlockRemoved / AllBlocksCleared) so this
module needs no vLLM import and is unit-testable without a vLLM install.

Mirrors grpc_servicer/smg_grpc_servicer/sglang/servicer.py:652-779.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)

_U64_MASK = 0xFFFFFFFFFFFFFFFF
_I64_SIGN_BIT = 0x8000000000000000
_U64_MODULUS = 0x10000000000000000


def to_int64(value: int) -> int:
    """Reduce an arbitrary-width int (e.g. a vLLM block hash) to a signed int64.

    The proto block_hash field is int64; vLLM hashes can be wider. SMG uses the
    hash only as a node identity and re-derives a content hash from token_ids
    (kv_event_monitor.rs:592-597), so a deterministic 64-bit reduction is safe.
    """
    masked = value & _U64_MASK
    if masked >= _I64_SIGN_BIT:
        masked -= _U64_MODULUS
    return masked


def endpoint_for_rank(endpoint: str, dp_rank: int) -> str:
    """Resolve a vLLM KV-events PUB endpoint to a connectable SUB address.

    The publisher binds to e.g. ``tcp://*:5557``; subscribers connect to
    ``127.0.0.1``. For data-parallel deployments each rank publishes on
    ``base_port + dp_rank``. Non-tcp endpoints (ipc://, inproc://) are returned
    with the bind wildcard substituted but no port arithmetic.
    """
    resolved = endpoint.replace("*", "127.0.0.1")
    if resolved.startswith("tcp://") and dp_rank:
        host, _, port = resolved.rpartition(":")
        return f"{host}:{int(port) + dp_rank}"
    return resolved


def resolve_kv_events_config(engine: object):
    """Return the vLLM KVEventsConfig iff ZMQ event publishing is enabled, else None.

    Reads ``engine.vllm_config.kv_events_config`` (same access pattern as
    servicer.py:416 for kv_transfer_config). Uses getattr throughout so it works
    against any vLLM version and is testable with a fake engine.
    """
    vllm_config = getattr(engine, "vllm_config", None)
    cfg = getattr(vllm_config, "kv_events_config", None)
    if cfg is None:
        return None
    if not getattr(cfg, "enable_kv_cache_events", False):
        logger.info("vLLM KV cache events not enabled; SubscribeKvEvents disabled")
        return None
    if getattr(cfg, "publisher", None) != "zmq":
        logger.info(
            "vLLM KV events publisher is %r, not 'zmq'; SubscribeKvEvents disabled",
            getattr(cfg, "publisher", None),
        )
        return None
    logger.info("vLLM KV events enabled: endpoint=%s", getattr(cfg, "endpoint", "?"))
    return cfg


def convert_event(event: object, event_id: int) -> common_pb2.KvCacheEvent | None:
    """Convert one decoded vLLM event to a proto KvCacheEvent (or None if unknown)."""
    name = type(event).__name__

    if name == "BlockStored":
        block_size = int(event.block_size)
        blocks = []
        for i, block_hash in enumerate(event.block_hashes):
            start = i * block_size
            end = start + block_size
            block = common_pb2.KvBlock(
                block_hash=to_int64(block_hash),
                token_ids=list(event.token_ids[start:end]),
                block_size=block_size,
            )
            lora_id = getattr(event, "lora_id", None)
            if lora_id is not None:
                block.lora_id = to_int64(lora_id)
            blocks.append(block)
        stored = common_pb2.KvBlocksStored(blocks=blocks)
        parent = getattr(event, "parent_block_hash", None)
        if parent is not None:
            stored.parent_block_hash = to_int64(parent)
        return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

    if name == "BlockRemoved":
        return common_pb2.KvCacheEvent(
            event_id=event_id,
            removed=common_pb2.KvBlocksRemoved(
                block_hashes=[to_int64(h) for h in event.block_hashes]
            ),
        )

    if name == "AllBlocksCleared":
        return common_pb2.KvCacheEvent(event_id=event_id, cleared=common_pb2.KvCacheCleared())

    logger.debug("Unknown vLLM KV event type %r, skipping", name)
    return None


def convert_batch(
    raw_batch: object, seq_num: int, event_id_start: int
) -> tuple[common_pb2.KvEventBatch, int]:
    """Convert a decoded vLLM KVEventBatch to a proto KvEventBatch.

    Returns the proto batch and the new event-id counter. The counter advances
    once per input event (even if unconvertible) so ids stay monotonic.
    """
    proto = common_pb2.KvEventBatch(sequence_number=seq_num, timestamp=raw_batch.ts)
    dp_rank = getattr(raw_batch, "data_parallel_rank", None)
    if dp_rank is not None:
        proto.dp_rank = dp_rank

    event_id = event_id_start
    for event in raw_batch.events:
        event_id += 1
        proto_event = convert_event(event, event_id)
        if proto_event is not None:
            proto.events.append(proto_event)
    return proto, event_id


async def stream_kv_events(
    sub_socket: object,
    decode: Callable[[bytes], object],
    send_initial_metadata: Callable[[], Awaitable[None]],
    is_cancelled: Callable[[], bool],
    *,
    recv_timeout: float = 1.0,
) -> AsyncIterator[common_pb2.KvEventBatch]:
    """Core ZMQ→proto streaming loop, decoupled from vLLM and gRPC types.

    Args:
        sub_socket: a connected ``zmq.asyncio`` SUB socket (duck-typed; only
            ``recv_multipart()`` is used).
        decode: bytes → decoded vLLM batch (e.g. ``msgspec.msgpack.Decoder(KVEventBatch).decode``).
        send_initial_metadata: awaitable called once before the first recv so the
            gRPC client's ``subscribe_kv_events().await`` resolves promptly.
        is_cancelled: returns True when the RPC is cancelled; loop then exits.
        recv_timeout: per-recv timeout so cancellation is observed even when idle.

    Yields proto KvEventBatch using the ZMQ publisher's native sequence numbers.
    """
    await send_initial_metadata()
    event_id = 0
    while not is_cancelled():
        try:
            frames = await asyncio.wait_for(sub_socket.recv_multipart(), timeout=recv_timeout)
        except TimeoutError:
            continue

        # ZMQ multipart: [topic, 8-byte big-endian seq, msgpack payload].
        if len(frames) < 3:
            continue
        zmq_seq = int.from_bytes(frames[1], "big")
        try:
            raw_batch = decode(frames[2])
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the stream
            logger.warning("Failed to decode vLLM KV event batch: %s", e)
            continue

        proto_batch, event_id = convert_batch(raw_batch, zmq_seq, event_id)
        yield proto_batch
