"""
Bridge vLLM's internal KV cache event publisher onto the SMG gRPC proto.
"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator

import msgspec.msgpack
import zmq
import zmq.asyncio
from smg_grpc_proto.generated import common_pb2
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch as VllmKvEventBatch,
    ZmqEventPublisher,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_loopback_ip

logger = init_logger(__name__)

_REPLAY_TIMEOUT_MS = 100


class UnsupportedKvEventLayoutError(RuntimeError):
    """Raised when a vLLM KV event cannot be translated losslessly."""


def _connect_endpoint(endpoint: str | None, rank: int) -> str | None:
    """Convert a publisher bind endpoint into a subscriber connect endpoint."""
    offset = ZmqEventPublisher.offset_endpoint_port(endpoint, rank)
    if offset is None:
        return None
    if offset.startswith("tcp://*"):
        return offset.replace("*", get_loopback_ip(), 1)
    return offset


def _chunk_token_ids(token_ids: list[int], block_size: int, num_blocks: int) -> list[list[int]]:
    if num_blocks == 0:
        if token_ids:
            raise UnsupportedKvEventLayoutError(
                "Received BlockStored with tokens but no block hashes"
            )
        return []
    if block_size <= 0:
        raise UnsupportedKvEventLayoutError(
            f"Received BlockStored with invalid block_size={block_size}"
        )

    expected_tokens = block_size * num_blocks
    if len(token_ids) != expected_tokens:
        raise UnsupportedKvEventLayoutError(
            "Unsupported BlockStored layout from vLLM: "
            f"expected {expected_tokens} token ids for {num_blocks} blocks "
            f"of size {block_size}, got {len(token_ids)}. "
            "This usually indicates null blocks in the cached range, which "
            "cannot be translated losslessly without per-block token ranges."
        )

    chunks = []
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        chunks.append(token_ids[start:end])
    return chunks


def _translate_event(event) -> common_pb2.KvCacheEvent:
    if isinstance(event, BlockStored):
        blocks = []
        token_chunks = _chunk_token_ids(
            event.token_ids,
            event.block_size,
            len(event.block_hashes),
        )
        for block_hash, token_ids in zip(event.block_hashes, token_chunks, strict=True):
            kwargs = {
                "block_hash": int(block_hash),
                "token_ids": token_ids,
                "block_size": event.block_size,
            }
            if event.lora_id is not None:
                kwargs["lora_id"] = event.lora_id
            blocks.append(common_pb2.KvBlock(**kwargs))

        stored_kwargs: dict[str, object] = {"blocks": blocks}
        if event.parent_block_hash is not None:
            stored_kwargs["parent_block_hash"] = int(event.parent_block_hash)

        return common_pb2.KvCacheEvent(
            stored=common_pb2.KvBlocksStored(**stored_kwargs),
        )

    if isinstance(event, BlockRemoved):
        return common_pb2.KvCacheEvent(
            removed=common_pb2.KvBlocksRemoved(
                block_hashes=[int(block_hash) for block_hash in event.block_hashes]
            ),
        )

    if isinstance(event, AllBlocksCleared):
        return common_pb2.KvCacheEvent(cleared=common_pb2.KvCacheCleared())

    raise TypeError(f"Unsupported vLLM KV event: {type(event)!r}")


class VllmKvEventTranslator:
    """Translate decoded vLLM KV event batches into SMG protobuf batches."""

    def translate_batch(
        self,
        batch: VllmKvEventBatch,
        sequence_number: int,
    ) -> common_pb2.KvEventBatch:
        translated = common_pb2.KvEventBatch(
            sequence_number=sequence_number,
            timestamp=batch.ts,
            events=[_translate_event(event) for event in batch.events],
        )
        if batch.data_parallel_rank is not None:
            translated.dp_rank = batch.data_parallel_rank
        return translated


class KvEventReplayBuffer:
    """Own replay-window state, per-rank sequence tracking, and global ordering."""

    def __init__(self, max_batches: int) -> None:
        self._buffer: deque[common_pb2.KvEventBatch] = deque(maxlen=max(1, max_batches))
        self._next_sequence_number = 1
        self._last_rank_sequence: dict[int, int] = {}

    @property
    def batches(self) -> deque[common_pb2.KvEventBatch]:
        return self._buffer

    @property
    def next_sequence_number(self) -> int:
        return self._next_sequence_number

    @property
    def last_rank_sequence(self) -> dict[int, int]:
        return self._last_rank_sequence

    def normalize_requested_sequence(self, next_seq: int) -> int:
        if not self._buffer:
            return next_seq

        oldest = self._buffer[0].sequence_number
        producer_next = self._next_sequence_number

        if next_seq and next_seq < oldest:
            logger.warning(
                "Requested KV event replay from expired sequence %s; oldest=%s",
                next_seq,
                oldest,
            )
            return oldest

        if next_seq and next_seq > producer_next:
            logger.warning(
                "Requested KV event replay from future sequence %s; producer_next=%s. "
                "Assuming producer restart and replaying from oldest=%s",
                next_seq,
                producer_next,
                oldest,
            )
            return oldest

        return next_seq

    def replay_from(self, next_seq: int) -> list[common_pb2.KvEventBatch]:
        normalized = self.normalize_requested_sequence(next_seq)
        return [batch for batch in self._buffer if batch.sequence_number >= normalized]

    def should_wake(self, next_seq: int) -> bool:
        return bool(
            self._buffer
            and (
                self._buffer[-1].sequence_number >= next_seq
                or next_seq > self._next_sequence_number
            )
        )

    def should_skip_rank_sequence(self, rank: int, rank_seq: int) -> bool:
        return rank_seq == self._last_rank_sequence.get(rank, -1)

    def maybe_log_rank_sequence_regression(self, rank: int, rank_seq: int) -> None:
        last_rank_seq = self._last_rank_sequence.get(rank, -1)
        if rank_seq < last_rank_seq:
            logger.warning(
                "KV event publisher sequence regressed for rank %s: previous=%s received=%s. "
                "Assuming publisher restart and accepting new batches from this rank",
                rank,
                last_rank_seq,
                rank_seq,
            )

    def append(self, rank: int, rank_seq: int, batch: common_pb2.KvEventBatch) -> None:
        self._buffer.append(batch)
        self._next_sequence_number += 1
        self._last_rank_sequence[rank] = rank_seq


class VllmKvEventBridge:
    """
    Collect vLLM KV events from ZMQ publishers and expose them as a single
    gRPC-friendly event stream with a global replay buffer.
    """

    def __init__(
        self,
        kv_events_config: KVEventsConfig | None,
        data_parallel_size: int = 1,
    ) -> None:
        self._config = kv_events_config
        self._data_parallel_size = max(1, data_parallel_size)
        self._decoder = msgspec.msgpack.Decoder(type=VllmKvEventBatch)
        self._ctx = zmq.asyncio.Context.instance()
        self._translator = VllmKvEventTranslator()
        self._replay = KvEventReplayBuffer(
            max_batches=max(1, getattr(kv_events_config, "buffer_steps", 10_000))
            if kv_events_config is not None
            else 1,
        )
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._fatal_error: Exception | None = None

    @property
    def _buffer(self) -> deque[common_pb2.KvEventBatch]:
        return self._replay.batches

    @property
    def _next_sequence_number(self) -> int:
        return self._replay.next_sequence_number

    @property
    def _last_rank_sequence(self) -> dict[int, int]:
        return self._replay.last_rank_sequence

    @property
    def enabled(self) -> bool:
        return bool(
            self._config is not None
            and self._config.enable_kv_cache_events
            and self._config.publisher == "zmq"
            and self._config.endpoint
        )

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._run(), name="vllm-kv-event-bridge")

    async def shutdown(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def subscribe(
        self,
        start_sequence_number: int,
    ) -> AsyncGenerator[common_pb2.KvEventBatch, None]:
        if not self.enabled:
            raise RuntimeError("KV event bridge is disabled")

        self.start()
        next_seq = max(0, int(start_sequence_number))

        async with self._condition:
            replay = self._replay.replay_from(next_seq)
            if replay:
                next_seq = replay[0].sequence_number
            else:
                next_seq = self._replay.normalize_requested_sequence(next_seq)

        for batch in replay:
            yield batch
            next_seq = max(next_seq, batch.sequence_number + 1)

        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._fatal_error is not None
                    or self._closed
                    or self._replay.should_wake(next_seq)
                )
                if self._fatal_error is not None:
                    raise self._fatal_error
                if self._closed:
                    return
                live = self._replay.replay_from(next_seq)
                if live:
                    next_seq = live[0].sequence_number
                else:
                    next_seq = self._replay.normalize_requested_sequence(next_seq)

            for batch in live:
                yield batch
                next_seq = max(next_seq, batch.sequence_number + 1)

    async def _run(self) -> None:
        sub_sockets: list[tuple[int, zmq.asyncio.Socket]] = []
        replay_sockets: list[tuple[int, zmq.asyncio.Socket]] = []
        try:
            config = self._config
            if config is None:
                return

            topic = config.topic.encode("utf-8")

            for rank in range(self._data_parallel_size):
                endpoint = _connect_endpoint(config.endpoint, rank)
                if endpoint is None:
                    continue

                sub = self._ctx.socket(zmq.SUB)
                sub.setsockopt(zmq.SUBSCRIBE, topic)
                sub.connect(endpoint)
                sub_sockets.append((rank, sub))

                replay_endpoint = _connect_endpoint(config.replay_endpoint, rank)
                if replay_endpoint:
                    replay = self._ctx.socket(zmq.REQ)
                    replay.connect(replay_endpoint)
                    replay_sockets.append((rank, replay))

            for rank, replay in replay_sockets:
                await self._replay_rank(rank, replay)

            poller = zmq.asyncio.Poller()
            socket_to_rank: dict[zmq.asyncio.Socket, int] = {}
            for rank, sub in sub_sockets:
                poller.register(sub, zmq.POLLIN)
                socket_to_rank[sub] = rank

            while not self._closed:
                events = dict(await poller.poll(timeout=1000))
                for sub, _ in events.items():
                    rank = socket_to_rank[sub]
                    frames = await sub.recv_multipart()
                    if len(frames) != 3:
                        logger.warning("Invalid KV event frame from rank %s: %s", rank, frames)
                        continue
                    _, seq_bytes, payload = frames
                    rank_seq = int.from_bytes(seq_bytes, "big")
                    await self._ingest_rank_batch(rank, rank_seq, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("KV event bridge failed")
            self._fatal_error = exc
        finally:
            for _, sock in replay_sockets:
                sock.close(linger=0)
            for _, sock in sub_sockets:
                sock.close(linger=0)
            async with self._condition:
                self._condition.notify_all()

    async def _replay_rank(self, rank: int, replay: zmq.asyncio.Socket) -> None:
        await replay.send((0).to_bytes(8, "big"))
        while not self._closed:
            if not await replay.poll(timeout=_REPLAY_TIMEOUT_MS):
                break
            frames = await replay.recv_multipart()
            if not frames or not frames[-1]:
                break
            if len(frames) != 2:
                logger.warning("Invalid replay frame from rank %s: %s", rank, frames)
                continue
            seq_bytes, payload = frames
            rank_seq = int.from_bytes(seq_bytes, "big", signed=True)
            if rank_seq < 0:
                continue
            await self._ingest_rank_batch(rank, rank_seq, payload)

    async def _ingest_rank_batch(self, rank: int, rank_seq: int, payload: bytes) -> None:
        if self._replay.should_skip_rank_sequence(rank, rank_seq):
            return
        self._replay.maybe_log_rank_sequence_regression(rank, rank_seq)

        batch = self._decoder.decode(payload)
        translated = self._translator.translate_batch(
            batch,
            sequence_number=self._replay.next_sequence_number,
        )
        self._replay.append(rank, rank_seq, translated)

        async with self._condition:
            self._condition.notify_all()
