"""Integration test: TokenSpeed-shaped ZMQ KV events → proto via stream_kv_events.

No TokenSpeed required — batches are encoded with local ``msgspec`` structs that
mirror ``tokenspeed.runtime.pd.kv_events`` (array_like + tagged union), and the
shared streaming loop runs with a real ``msgspec`` decoder, exactly as the
servicer does. This exercises the full ZMQ-frame → msgpack-decode → proto path.

Run with: pytest grpc_servicer/tests/test_tokenspeed_kv_events_stream.py -v
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")
zmq = pytest.importorskip("zmq")
msgspec = pytest.importorskip("msgspec")
import zmq.asyncio  # noqa: E402, F811

_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "kv_events.py"
_spec = importlib.util.spec_from_file_location("shared_kv_events", _MODULE_PATH)
kv_events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_events)


# Mirror tokenspeed.runtime.pd.kv_events wire structs (array_like + tagged
# union); the class names double as the union tags and the convert_event
# dispatch keys.
class KVCacheEvent(msgspec.Struct, array_like=True, omit_defaults=True, tag=True):
    pass


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
    attn_dp_rank: int | None = None


def _seq_bytes(n: int) -> bytes:
    return n.to_bytes(8, "big")


@pytest.mark.asyncio
async def test_stream_decodes_tokenspeed_batches():
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    ctx = zmq.asyncio.Context.instance()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    collected = []
    try:
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub.subscribe(b"")
        sub.connect(f"tcp://127.0.0.1:{port}")
        await asyncio.sleep(0.2)  # allow SUB connection before publishing

        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockStored(
                        block_hashes=[111],
                        parent_block_hash=None,
                        token_ids=[1, 2],
                        block_size=2,
                    )
                ],
                attn_dp_rank=0,
            ),
            KVEventBatch(ts=2.0, events=[BlockRemoved(block_hashes=[111])]),
        ]

        async def _noop():
            return None

        async def consume():
            async for batch in kv_events.stream_kv_events(
                sub,
                decoder.decode,
                send_initial_metadata=_noop,
                is_cancelled=lambda: len(collected) >= 2,
            ):
                collected.append(batch)

        consumer = asyncio.create_task(consume())
        for seq, batch in enumerate(batches):
            await pub.send_multipart([b"", _seq_bytes(seq), encoder.encode(batch)])
            await asyncio.sleep(0.05)
        await asyncio.wait_for(consumer, timeout=5)
    finally:
        pub.close(linger=0)
        sub.close(linger=0)

    assert [b.sequence_number for b in collected] == [0, 1]
    stored = collected[0].events[0].stored
    assert stored.blocks[0].block_hash == 111
    assert list(stored.blocks[0].token_ids) == [1, 2]
    assert collected[0].dp_rank == 0  # attn_dp_rank 0 → proto dp_rank 0
    assert collected[1].events[0].WhichOneof("data") == "removed"
