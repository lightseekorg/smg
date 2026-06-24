"""Integration test for kv_events.stream_kv_events over a real ZMQ PUB/SUB pair.

No vLLM required: the msgpack decoder is replaced with an injected fake.
Run with: pytest grpc_servicer/tests/test_vllm_kv_events_stream.py -v
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")
zmq = pytest.importorskip("zmq")
import zmq.asyncio  # noqa: E402, F811

_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "kv_events.py"
_spec = importlib.util.spec_from_file_location("vllm_kv_events", _MODULE_PATH)
kv_events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_events)


# Class name must be "BlockStored" for kv_events.convert_event dispatch.
class BlockStored:
    def __init__(self):
        self.block_hashes = [1]
        self.parent_block_hash = None
        self.token_ids = [1, 2]
        self.block_size = 2
        self.lora_id = None


class KVEventBatch:
    def __init__(self, seq):
        self.ts = float(seq)
        self.events = [BlockStored()]
        self.data_parallel_rank = None


def _seq_bytes(n: int) -> bytes:
    return n.to_bytes(8, "big")


@pytest.mark.asyncio
async def test_stream_yields_batches_in_order():
    ctx = zmq.asyncio.Context.instance()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    collected = []
    try:
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub.subscribe(b"")
        sub.connect(f"tcp://127.0.0.1:{port}")
        await asyncio.sleep(0.2)  # allow SUB connection to establish before publishing

        # decode ignores payload bytes and returns a prebuilt fake batch keyed by seq.
        def fake_decode(payload: bytes):
            return KVEventBatch(int.from_bytes(payload, "big"))

        async def _noop():
            return None

        async def consume():
            async for batch in kv_events.stream_kv_events(
                sub,
                fake_decode,
                send_initial_metadata=_noop,
                is_cancelled=lambda: len(collected) >= 2,
            ):
                collected.append(batch)

        consumer = asyncio.create_task(consume())
        for seq in (1, 2):
            await pub.send_multipart([b"kv", _seq_bytes(seq), seq.to_bytes(8, "big")])
            await asyncio.sleep(0.05)
        await asyncio.wait_for(consumer, timeout=5)
    finally:
        pub.close(linger=0)
        sub.close(linger=0)

    assert [b.sequence_number for b in collected] == [1, 2]
    assert collected[0].events[0].stored.blocks[0].block_hash == 1


@pytest.mark.asyncio
async def test_stream_skips_short_frames_and_decode_errors():
    ctx = zmq.asyncio.Context.instance()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    collected = []
    try:
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub.subscribe(b"")
        sub.connect(f"tcp://127.0.0.1:{port}")
        await asyncio.sleep(0.2)

        def fake_decode(payload: bytes):
            if payload == b"bad":
                raise ValueError("boom")
            return KVEventBatch(7)

        async def _noop():
            return None

        async def consume():
            async for batch in kv_events.stream_kv_events(
                sub,
                fake_decode,
                send_initial_metadata=_noop,
                is_cancelled=lambda: len(collected) >= 1,
            ):
                collected.append(batch)

        consumer = asyncio.create_task(consume())
        await pub.send_multipart([b"kv", _seq_bytes(1)])  # short frame -> skipped
        await asyncio.sleep(0.05)
        await pub.send_multipart([b"kv", _seq_bytes(2), b"bad"])  # decode error -> skipped
        await asyncio.sleep(0.05)
        await pub.send_multipart([b"kv", _seq_bytes(3), b"ok"])  # good -> yielded
        await asyncio.wait_for(consumer, timeout=5)
    finally:
        pub.close(linger=0)
        sub.close(linger=0)

    assert [b.sequence_number for b in collected] == [3]
