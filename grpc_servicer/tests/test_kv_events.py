import asyncio
import time

import pytest

pytest.importorskip("torch")

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
)
from vllm.utils.network_utils import get_open_ports_list, get_tcp_uri

from smg_grpc_servicer.vllm.kv_events import (
    KvEventReplayBuffer,
    UnsupportedKvEventLayoutError,
    VllmKvEventBridge,
    VllmKvEventTranslator,
)
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer


def _make_config() -> KVEventsConfig:
    ports = get_open_ports_list(2)
    ports.sort()
    return KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=get_tcp_uri("127.0.0.1", ports[0]).replace("127.0.0.1", "*"),
        replay_endpoint=get_tcp_uri("127.0.0.1", ports[1]).replace("127.0.0.1", "*"),
        buffer_steps=128,
        topic="kv-test",
    )


def test_kv_event_bridge_replays_and_streams_live_events():
    async def run() -> None:
        config = _make_config()
        publisher = EventPublisherFactory.create(config, 0)
        bridge = VllmKvEventBridge(config, data_parallel_size=1)
        bridge.start()

        try:
            await asyncio.sleep(0.2)

            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[101, 102],
                            parent_block_hash=100,
                            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                            block_size=4,
                            lora_id=7,
                            medium="GPU",
                            lora_name="adapter",
                        )
                    ],
                    data_parallel_rank=0,
                )
            )

            await asyncio.sleep(0.3)

            stream = bridge.subscribe(0)
            first = await anext(stream)
            assert first.sequence_number == 1
            assert first.dp_rank == 0
            assert len(first.events) == 1
            assert first.events[0].HasField("stored")
            assert first.events[0].stored.parent_block_hash == 100
            assert [block.block_hash for block in first.events[0].stored.blocks] == [101, 102]
            assert [list(block.token_ids) for block in first.events[0].stored.blocks] == [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]
            assert [block.lora_id for block in first.events[0].stored.blocks] == [7, 7]

            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockRemoved(block_hashes=[101], medium="GPU"),
                        AllBlocksCleared(),
                    ],
                    data_parallel_rank=0,
                )
            )

            second = await asyncio.wait_for(anext(stream), timeout=2)
            assert second.sequence_number == 2
            assert second.dp_rank == 0
            assert second.events[0].HasField("removed")
            assert list(second.events[0].removed.block_hashes) == [101]
            assert second.events[1].HasField("cleared")

            await stream.aclose()
        finally:
            await bridge.shutdown()
            publisher.shutdown()

    asyncio.run(run())


def test_kv_event_bridge_assigns_global_sequence_numbers_across_dp_ranks():
    async def run() -> None:
        config = _make_config()
        publisher_rank0 = EventPublisherFactory.create(config, 0)
        publisher_rank1 = EventPublisherFactory.create(config, 1)
        bridge = VllmKvEventBridge(config, data_parallel_size=2)
        bridge.start()

        try:
            await asyncio.sleep(0.2)

            publisher_rank0.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[201],
                            parent_block_hash=None,
                            token_ids=[9, 10, 11, 12],
                            block_size=4,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                    data_parallel_rank=0,
                )
            )
            publisher_rank1.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[BlockRemoved(block_hashes=[301], medium="GPU")],
                    data_parallel_rank=1,
                )
            )

            stream = bridge.subscribe(0)
            batch_a = await asyncio.wait_for(anext(stream), timeout=2)
            batch_b = await asyncio.wait_for(anext(stream), timeout=2)

            assert [batch_a.sequence_number, batch_b.sequence_number] == [1, 2]
            assert {batch_a.dp_rank, batch_b.dp_rank} == {0, 1}

            await stream.aclose()
        finally:
            await bridge.shutdown()
            publisher_rank0.shutdown()
            publisher_rank1.shutdown()

    asyncio.run(run())


def test_kv_event_bridge_replays_from_head_when_requested_sequence_is_in_future():
    async def run() -> None:
        config = _make_config()
        publisher = EventPublisherFactory.create(config, 0)
        bridge = VllmKvEventBridge(config, data_parallel_size=1)
        bridge.start()

        try:
            stream = bridge.subscribe(25)
            await asyncio.sleep(0.2)

            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[401],
                            parent_block_hash=None,
                            token_ids=[1, 2, 3, 4],
                            block_size=4,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                    data_parallel_rank=0,
                )
            )

            first = await asyncio.wait_for(anext(stream), timeout=2)
            assert first.sequence_number == 1
            assert first.dp_rank == 0

            await stream.aclose()
        finally:
            await bridge.shutdown()
            publisher.shutdown()

    asyncio.run(run())


def test_kv_event_bridge_fails_closed_on_unsupported_blockstored_layout():
    async def run() -> None:
        config = _make_config()
        publisher = EventPublisherFactory.create(config, 0)
        bridge = VllmKvEventBridge(config, data_parallel_size=1)
        bridge.start()

        try:
            await asyncio.sleep(0.2)

            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[501, 502],
                            parent_block_hash=None,
                            token_ids=[1, 2, 3, 4],
                            block_size=4,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                    data_parallel_rank=0,
                )
            )

            stream = bridge.subscribe(0)
            with pytest.raises(UnsupportedKvEventLayoutError):
                await asyncio.wait_for(anext(stream), timeout=2)
        finally:
            await bridge.shutdown()
            publisher.shutdown()

    asyncio.run(run())


def test_kv_event_bridge_accepts_batches_after_rank_sequence_regression():
    async def run() -> None:
        import msgspec.msgpack

        config = _make_config()
        bridge = VllmKvEventBridge(config, data_parallel_size=1)

        first_payload = msgspec.msgpack.encode(
            KVEventBatch(
                ts=time.time(),
                events=[
                    BlockStored(
                        block_hashes=[601],
                        parent_block_hash=None,
                        token_ids=[1, 2, 3, 4],
                        block_size=4,
                        lora_id=None,
                        medium="GPU",
                        lora_name=None,
                    )
                ],
                data_parallel_rank=0,
            )
        )
        second_payload = msgspec.msgpack.encode(
            KVEventBatch(
                ts=time.time(),
                events=[BlockRemoved(block_hashes=[601], medium="GPU")],
                data_parallel_rank=0,
            )
        )

        await bridge._ingest_rank_batch(0, 5, first_payload)
        await bridge._ingest_rank_batch(0, 1, second_payload)

        buffered = list(bridge._buffer)
        assert [batch.sequence_number for batch in buffered] == [1, 2]
        assert bridge._last_rank_sequence[0] == 1
        assert buffered[1].events[0].HasField("removed")

    asyncio.run(run())


def test_kv_event_bridge_caught_up_subscriber_waits_for_new_batch():
    async def run() -> None:
        import msgspec.msgpack

        config = _make_config()
        bridge = VllmKvEventBridge(config, data_parallel_size=1)
        bridge._task = asyncio.create_task(asyncio.sleep(60))

        try:
            first_payload = msgspec.msgpack.encode(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[701],
                            parent_block_hash=None,
                            token_ids=[1, 2, 3, 4],
                            block_size=4,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                    data_parallel_rank=0,
                )
            )
            second_payload = msgspec.msgpack.encode(
                KVEventBatch(
                    ts=time.time(),
                    events=[BlockRemoved(block_hashes=[701], medium="GPU")],
                    data_parallel_rank=0,
                )
            )

            await bridge._ingest_rank_batch(0, 1, first_payload)

            stream = bridge.subscribe(2)
            next_batch_task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.1)
            assert not next_batch_task.done()

            await bridge._ingest_rank_batch(0, 2, second_payload)
            second = await asyncio.wait_for(next_batch_task, timeout=2)
            assert second.sequence_number == 2
            assert second.events[0].HasField("removed")

            await stream.aclose()
        finally:
            await bridge.shutdown()

    asyncio.run(run())


def test_subscribe_kv_events_sends_initial_metadata_before_first_event():
    async def run() -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.initial_metadata: list[tuple[()]] = []

            async def send_initial_metadata(self, metadata: tuple[()]) -> None:
                self.initial_metadata.append(metadata)

        class FakeBridge:
            enabled = True

            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def subscribe(self, start_sequence_number: int):
                assert start_sequence_number == 11
                self.started.set()
                await self.release.wait()
                yield "batch"

        bridge = FakeBridge()
        servicer = VllmEngineServicer.__new__(VllmEngineServicer)
        servicer.kv_event_bridge = bridge
        request = type("Request", (), {"start_sequence_number": 11})()
        context = FakeContext()

        stream = VllmEngineServicer.SubscribeKvEvents(servicer, request, context)
        next_batch_task = asyncio.create_task(anext(stream))

        await bridge.started.wait()
        assert context.initial_metadata == [()]
        assert not next_batch_task.done()

        bridge.release.set()
        assert await asyncio.wait_for(next_batch_task, timeout=2) == "batch"
        await stream.aclose()

    asyncio.run(run())


def test_kv_event_translator_preserves_optional_dp_rank_absence():
    translator = VllmKvEventTranslator()

    translated = translator.translate_batch(
        KVEventBatch(
            ts=123.0,
            events=[AllBlocksCleared()],
            data_parallel_rank=None,
        ),
        sequence_number=7,
    )

    assert translated.sequence_number == 7
    assert translated.timestamp == 123.0
    assert translated.events[0].HasField("cleared")
    assert not translated.HasField("dp_rank")


def test_kv_event_replay_buffer_normalizes_replay_window():
    from smg_grpc_proto.generated import common_pb2

    replay = KvEventReplayBuffer(max_batches=2)
    replay.append(
        rank=0,
        rank_seq=1,
        batch=common_pb2.KvEventBatch(sequence_number=1),
    )
    replay.append(
        rank=0,
        rank_seq=2,
        batch=common_pb2.KvEventBatch(sequence_number=2),
    )
    replay.append(
        rank=0,
        rank_seq=3,
        batch=common_pb2.KvEventBatch(sequence_number=3),
    )

    assert [batch.sequence_number for batch in replay.replay_from(1)] == [2, 3]
    assert [batch.sequence_number for batch in replay.replay_from(99)] == [2, 3]
    assert replay.last_rank_sequence[0] == 3
