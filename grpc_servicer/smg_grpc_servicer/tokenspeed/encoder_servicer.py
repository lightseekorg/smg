"""TokenSpeed EPD encode servicer.

Receives ``Encode`` RPCs from the gateway and forwards them to a vision-only
encode worker (the engine's ``run_encode_loop``) over the same AsyncLLM
scheduler-input channel the LM uses. The encode worker runs the vision tower and
ships the resulting image embeddings to prefill workers over Mooncake; this
servicer only acks (the embeddings never flow back through the gateway).
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING

from smg_grpc_proto.generated import (
    tokenspeed_encoder_pb2,
    tokenspeed_encoder_pb2_grpc,
)

from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _lazy_encode_request():
    from tokenspeed.runtime.disaggregation.embedding.encode_worker import EncodeRequest

    return EncodeRequest


def _lazy_mm_item():
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    return Modality, MultimodalDataItem


class TokenSpeedEncoderServicer(tokenspeed_encoder_pb2_grpc.TokenSpeedEncoderServicer):
    """gRPC servicer fronting the engine's encode worker over AsyncLLM."""

    def __init__(
        self,
        async_llm: "AsyncLLM",
        server_args: "ServerArgs",
        scheduler_info: dict,
        health_servicer=None,
    ):
        self.async_llm = async_llm
        # EPD_PIXEL_SHM: ship pixels to the scheduler process as POSIX-SHM
        # handles instead of pickling the raw tensor over ZMQ (the dominant
        # per-image ingest cost: ~25ms at 1080p, ~580ms at 4K). The decision is
        # made once per item in _items_from_proto; the encode worker
        # materializes (or unlinks, on a cache hit) the segment on its side.
        self._pixel_shm = bool(os.environ.get("EPD_PIXEL_SHM"))

        # The encode worker hosts its OWN Mooncake bootstrap server (it is the
        # data source); prefill workers discover it at (this host, the
        # disaggregation bootstrap port). Both live on this node.
        from tokenspeed.runtime.utils.network import get_local_ip_by_remote

        self._bootstrap_host = get_local_ip_by_remote()
        self._bootstrap_port = server_args.disaggregation_bootstrap_port

        # Spatial merge factor for post-merge token counts (Qwen vision default 2).
        self._merge_size = self._resolve_merge_size()

        # EPD RDMA (M3): persistent NIXL puller agent. When the gateway ships
        # pixel_values as a remote handle (SMG_MM_PIXEL_RDMA), this agent READs the
        # buffer from the gateway's exported memory instead of receiving it inline.
        # Lazy + gated so non-RDMA deploys never import NIXL.
        self._nixl_agent = None
        # Gateways whose metadata is fetched + loaded (the gateway's arena is
        # registered ONCE at its init, so its metadata is fixed -> handshake once,
        # never per image). Guarded so concurrent first-images don't double-fetch.
        self._rdma_md_ready = set()  # (ip, port)
        self._rdma_md_lock = threading.Lock()
        # Pre-registered landing pool (one MR for the worker's life): a ring of fixed
        # slots the one-sided READ lands into, so no register/deregister per image.
        self._landing = None
        self._landing_np = None
        self._landing_base = 0
        self._landing_slot_bytes = 0
        self._landing_free = None
        if os.environ.get("SMG_MM_PIXEL_RDMA") in ("1", "true"):
            try:
                try:
                    from nixl_cu13._api import nixl_agent, nixl_agent_config
                except ImportError:
                    from nixl._api import nixl_agent, nixl_agent_config
                import torch

                # Initiator side: a listen thread (ephemeral port) is needed so the
                # bidirectional metadata exchange with the gateway's listener works.
                self._nixl_agent = nixl_agent(
                    f"smg-encode-{self._bootstrap_host}-{self._bootstrap_port}",
                    nixl_agent_config(
                        enable_listen_thread=True, listen_port=0, backends=["UCX"]
                    ),
                )
                # Register the landing pool ONCE; per image we only lease a slot.
                # Depth = max concurrent in-flight pixel pulls per encode worker. A
                # too-shallow ring backpressures the puller and stalls the ViT behind
                # it (slots aren't freed fast enough), so under encode-bound load /
                # over-saturation throughput caps well below the compute limit
                # (2E4P2D conc144 measured 5.5 at 16 slots vs 6.25 at 64, 0 ring
                # exhaustion). Default 64 mirrors the engine's E->P embedding-send
                # ring (TOKENSPEED_EPD_ENCODE_RING_SLOTS, set to 64 in the conc8/16
                # cliff fix) -- the pixel-pull ring was the leg left at the shallow
                # 16. At 32MiB/slot this is only 2GiB pinned/worker.
                slot_bytes = int(
                    os.environ.get("SMG_RDMA_SLOT_BYTES", 32 * 1024 * 1024)
                )
                n_slots = int(os.environ.get("SMG_RDMA_LANDING_SLOTS", 64))
                self._landing = torch.empty(
                    n_slots * slot_bytes, dtype=torch.uint8, pin_memory=True
                )
                self._landing_np = self._landing.numpy()
                self._landing_base = self._landing.data_ptr()
                self._landing_slot_bytes = slot_bytes
                reg = self._nixl_agent.get_reg_descs(
                    [(self._landing_base, n_slots * slot_bytes, 0, "")], "DRAM"
                )
                self._nixl_agent.register_memory(reg)
                self._landing_free = queue.Queue()
                for i in range(n_slots):
                    self._landing_free.put(i)
                logger.info(
                    "EPD RDMA: encode puller up (landing %d slots x %d B)",
                    n_slots,
                    slot_bytes,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("EPD RDMA: NIXL init failed (%s); inline only", e)
                self._nixl_agent = None

        self.async_llm.auto_create_handle_loop()
        logger.info("TokenSpeedEncoderServicer initialized")

    def _ensure_remote_ready(self, gw_name, ip, port, remote, room):
        """One-time metadata handshake per gateway. The gateway's pixel arena is
        registered ONCE at its init, so its metadata is fixed: we
        fetch_remote_metadata and wait for it to load exactly ONCE per (ip, port)
        (send_local_metadata is skipped by default for the one-sided rc READ; see
        SMG_RDMA_SEND_MD below), then every later image reuses the loaded remote
        agent (no per-image fetch -- that, plus the per-image register below, was
        the v1 control overhead that made RDMA slower than inline)."""
        key = (ip, port)
        if key in self._rdma_md_ready:
            return
        with self._rdma_md_lock:
            if key in self._rdma_md_ready:
                return
            agent = self._nixl_agent
            agent.fetch_remote_metadata(gw_name, ip, port)
            # For a one-sided READ the initiator only needs the target's md (fetched
            # above) -- the gateway never reads from us, so it never needs ours.
            # Pushing our md to the gateway makes its listener thread loadRemoteMD
            # our md, which over rc returns NIXL_ERR_NOT_ALLOWED (the gateway can't
            # load a peer md for rc) and stalls the reverse QP -- under cross-node
            # load this is the flaky pixel-pull hang (handshake holds _rdma_md_lock,
            # every other image blocks behind it, encode stalls after parse with idle
            # GPUs). The push is unnecessary for the one-sided READ and only matters
            # for a tcp transport, so DEFAULT to skipping it; SMG_RDMA_SEND_MD=1
            # re-enables it for tcp deploys. (Was: send-by-default gated by
            # SMG_RDMA_NO_SEND_MD, which left rc deploys hanging out of the box.)
            if os.environ.get("SMG_RDMA_SEND_MD") in ("1", "true"):
                agent.send_local_metadata(ip, port)
            ready = False
            for _ in range(5000):
                if agent.check_remote_metadata(gw_name, remote):
                    ready = True
                    break
                time.sleep(0.001)
            if not ready:
                raise RuntimeError(f"NIXL remote metadata not ready room={room}")
            self._rdma_md_ready.add(key)

    def _feature_from_remote(self, td, room: int, cast_to):
        """PULL the pixel tensor from the gateway's pre-registered arena (one-sided
        READ into our pre-registered landing pool), then reconstruct it like inline.

        Returns ``(feature, content_hash)``: a plain CPU tensor with ``None`` hash
        by default, or, under ``EPD_PIXEL_SHM``, a published ``ShmTensorHandle``
        plus the content hash computed off the slot bytes (the item carries the
        hash so ``set_pad_value`` never needs the raw tensor again).

        Descriptor wire format (gateway rdma.rs): [slot_addr u64 LE][gen u64 LE][port u16 LE][ip].
        Hot path per image: handshake-once (see _ensure_remote_ready) + lease a free
        landing slot + READ slot_addr -> our slot (no register/deregister, no metadata
        re-fetch) + length-assert + copy out before freeing the slot. The transfer
        notif is tagged with the room so the gateway returns its slot.
        """
        import numpy as np
        import torch

        agent = self._nixl_agent
        if self._landing_free is None:
            raise RuntimeError("EPD RDMA: remote payload but landing pool unavailable")
        # Descriptor wire format (gateway rdma.rs export_pixel_buffer):
        #   [slot_addr u64 LE][gen u64 LE][port u16 LE][ip utf8]
        # The slot itself is framed [gen u64][payload nbytes][gen u64]; nbytes is the
        # PAYLOAD size (proto field), so we READ nbytes + _FRAME and re-check both gen
        # stamps against the descriptor's gen to reject a recycled/torn slot.
        _GEN = 8  # must match rdma.rs GEN_BYTES
        _FRAME = 2 * _GEN  # must match rdma.rs FRAME_OVERHEAD
        desc = bytes(td.remote.descriptor)
        nbytes = int(td.remote.nbytes)
        if len(desc) < 8 + _GEN + 2:
            raise ValueError("remote descriptor too short")
        remote_addr = int.from_bytes(desc[:8], "little")
        expected_gen = int.from_bytes(desc[8 : 8 + _GEN], "little")
        port = int.from_bytes(desc[8 + _GEN : 10 + _GEN], "little")
        ip = desc[10 + _GEN :].decode()
        gw_name = "smg-gateway-encode"
        if nbytes + _FRAME > self._landing_slot_bytes:
            raise ValueError(
                f"remote pixel {nbytes}B (+{_FRAME}B gen frame) exceeds landing slot "
                f"{self._landing_slot_bytes}B"
            )

        remote = agent.get_xfer_descs([(remote_addr, nbytes + _FRAME, 0)], "DRAM")
        self._ensure_remote_ready(gw_name, ip, port, remote, room)

        # Lease a slot with patient, VISIBLE backpressure. Failing here aborts a
        # request whose prefill is already dispatched (the gateway turns it into
        # an embedding-receive abort), and a burst of those cascades into a
        # mass-abort storm (prefill tears down receive MRs mid-write -> RDMA
        # access violations). Slow admission is strictly cheaper than that.
        wait_budget = float(os.environ.get("SMG_RDMA_LANDING_WAIT_S", 120))
        t_acq = time.monotonic()
        slot = None
        while slot is None:
            try:
                slot = self._landing_free.get(timeout=5)
            except queue.Empty:
                waited = time.monotonic() - t_acq
                if waited >= wait_budget:
                    raise RuntimeError(
                        f"EPD RDMA: landing-ring starvation for {waited:.0f}s "
                        f"(room={room}); raise SMG_RDMA_LANDING_SLOTS / "
                        f"SMG_RDMA_LANDING_WAIT_S"
                    ) from None
                logger.warning(
                    "EPD RDMA: landing ring exhausted for %.0fs (room=%s, qsize=%d);"
                    " backpressuring",
                    waited,
                    room,
                    self._landing_free.qsize(),
                )
        try:
            off = slot * self._landing_slot_bytes
            laddr = self._landing_base + off
            local = agent.get_xfer_descs([(laddr, nbytes + _FRAME, 0)], "DRAM")
            h = agent.initialize_xfer("READ", local, remote, gw_name, str(room).encode())
            read_deadline = time.monotonic() + float(
                os.environ.get("SMG_RDMA_READ_TIMEOUT_S", 60)
            )
            spins = 0
            state = agent.transfer(h)
            while state in ("PROC", "IN_PROG"):
                # Yield between polls: under EPD_INGEST_OFFLOOP up to ~32 worker
                # threads poll concurrently, and a no-sleep spin starves the GIL
                # away from the hash/publish work that RETURNS slots -- the ring
                # runs dry and admission collapses. Spin briefly for small-READ
                # latency, then back off.
                spins += 1
                if spins > 64:
                    time.sleep(0.0005)
                if time.monotonic() > read_deadline:
                    agent.release_xfer_handle(h)
                    raise RuntimeError(f"NIXL READ timed out room={room}")
                state = agent.check_xfer_state(h)
            agent.release_xfer_handle(h)
            if state != "DONE":
                raise RuntimeError(f"NIXL READ state={state} room={room}")

            shape = list(td.shape)
            itemsize = 2 if td.dtype == "bfloat16" else np.dtype(td.dtype).itemsize
            expected = itemsize
            for d in shape:
                expected *= d
            if nbytes != expected:
                raise ValueError(
                    f"remote pixel size mismatch: nbytes={nbytes} expected={expected} "
                    f"(shape={shape} dtype={td.dtype})"
                )
            # Validate the generation stamps bracketing the payload. A slot the gateway
            # recycled (TTL misconfig / env drift between the two sides) or a READ torn
            # by a concurrent reuse leaves header/trailer != the descriptor's gen; fail
            # the room (the request aborts and the gateway re-attaches inline on retry)
            # rather than feed another image's pixels to the ViT. Correctness here does
            # not depend on the TTL value -- the stamps catch any aliased READ.
            hdr = int.from_bytes(self._landing_np[off : off + _GEN].tobytes(), "little")
            trl = int.from_bytes(
                self._landing_np[off + _GEN + nbytes : off + _FRAME + nbytes].tobytes(),
                "little",
            )
            if hdr != expected_gen or trl != expected_gen:
                raise ValueError(
                    f"EPD RDMA: slot generation mismatch room={room} "
                    f"expected={expected_gen} header={hdr} trailer={trl} "
                    f"(slot recycled/torn under a live READ -> wrong-image guard)"
                )
            # Reinterpret the slot bytes as the tensor; copy OUT (clone/.to) before
            # the finally returns the slot, since the slot is then reused. Payload sits
            # between the gen stamps (offset +_GEN).
            slot_np = self._landing_np[off + _GEN : off + _GEN + nbytes]
            if td.dtype == "bfloat16":
                t = torch.from_numpy(
                    slot_np.view(np.uint16).reshape(shape)
                ).view(torch.bfloat16)
            else:
                t = torch.from_numpy(slot_np.view(np.dtype(td.dtype)).reshape(shape))
            copied = False
            if cast_to is not None and t.dtype != cast_to and t.is_floating_point():
                t = t.to(cast_to)
                copied = True
            if self._pixel_shm:
                # Fused landing->SHM: hash + publish straight off the slot view.
                # publish() itself is the single copy out of the slot, so the
                # plain path's intermediate tensor materialization is skipped
                # entirely (hash_feature only reads the bytes).
                from tokenspeed.runtime.multimodal.hash import hash_feature
                from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle

                feat_hash = hash_feature(t)
                return ShmTensorHandle.publish(t), feat_hash
            # Copy out before the finally returns the slot for reuse.
            return (t if copied else t.clone()), None
        finally:
            self._landing_free.put(slot)

    def _resolve_merge_size(self) -> int:
        hf_config = getattr(self.async_llm.model_config, "hf_config", None)
        vision_config = getattr(hf_config, "vision_config", None)
        return int(getattr(vision_config, "spatial_merge_size", 2) or 2)

    def _items_from_proto(self, mm_inputs, bootstrap_room: int = 0):
        """Reconstruct the engine MultimodalDataItem(s) for the encode worker.

        Unlike the prefill leg, the encode worker NEEDS each item's encoder_input
        (it runs the tower). It also needs each item's post-merge token count so the executor
        can split the tower output; the gateway ships grid_thw but not
        placeholders to encode, so derive the count from grid_thw and set it as
        the item's single offset span (the offset positions are irrelevant to the
        encode side, only the count matters).
        """
        Modality, MultimodalDataItem = _lazy_mm_item()
        model_dtype = getattr(self.async_llm.model_config, "dtype", None)

        # mm_inputs is itemized (one MultimodalItem per image, each owning its
        # encoder_input + model_specific_tensors). The gateway sends one item per
        # Encode RPC keyed by bootstrap_room, but iterate generally.
        items = []
        for item_proto in mm_inputs.items:
            # The feature's CROSS-PROCESS representation is decided here, once, for
            # both payload arms: a plain CPU tensor by default, or (EPD_PIXEL_SHM) a
            # POSIX-SHM handle so the ZMQ hop to the scheduler pickles ~KB instead of
            # the 19-77MB pixels. The content hash is computed on the real bytes
            # before the swap and pre-set on the item.
            td = item_proto.encoder_input
            if td.WhichOneof("payload") == "remote":
                # EPD RDMA: PULL the pixels from the gateway's exported NIXL memory
                # (under EPD_PIXEL_SHM the slot bytes go straight to SHM, one copy).
                feature, feat_hash = self._feature_from_remote(
                    td, bootstrap_room, model_dtype
                )
            else:
                feature = TokenSpeedSchedulerServicer._tensor_from_proto(
                    td, cast_to=model_dtype
                )
                feat_hash = None
                if self._pixel_shm:
                    from tokenspeed.runtime.multimodal.hash import hash_feature
                    from tokenspeed.runtime.multimodal.shm_transport import (
                        ShmTensorHandle,
                    )

                    feat_hash = hash_feature(feature)
                    feature = ShmTensorHandle.publish(feature)
            model_specific = {
                name: TokenSpeedSchedulerServicer._tensor_from_proto(t, cast_to=model_dtype)
                for name, t in item_proto.model_specific_tensors.items()
            }

            grid = model_specific.get("image_grid_thw")
            if grid is None:
                # Tolerate the legacy "grid_thws" key (older gateway builds emit it on
                # the encode RPC); mirrors the engine kimi_k25 _grid() helper's tolerance.
                grid = model_specific.get("grid_thws")
            if grid is None:
                raise ValueError(
                    "encode request is missing image_grid_thw/grid_thws; "
                    f"have keys={sorted(model_specific.keys())}"
                )
            # grid is [num_images, 3] = (t, h, w) in patch units, per image.
            merge = self._merge_size
            offsets = []
            cursor = 0
            for row in grid.tolist():
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                span = t * (h // merge) * (w // merge)
                offsets.append((cursor, cursor + span - 1))
                cursor += span

            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=feat_hash,
                feature=feature,
                model_specific_data=model_specific,
                offsets=offsets,
            )
            item.set_pad_value()
            items.append(item)
        return items

    async def Encode(self, request, context):
        # The gateway sends one image per RPC, so one EncodeItemAssignment / room.
        # (Multiple items per RPC would need per-item rooms; not used today.) The
        # room also tags the RDMA free-notif, so resolve it before reading pixels.
        bootstrap_room = request.items[0].bootstrap_room if request.items else 0

        if os.environ.get("EPD_INGEST_OFFLOOP"):
            # Per-image ingest (~63ms: proto->tensor ~40ms + pickle ~20ms)
            # BLOCKS the lone asyncio event loop, so grpc.aio cannot deliver the
            # next Encode message until the previous one is fully ingested --
            # measured as the per-worker ~78ms serial pixel lane (the light cap).
            # Split it: parse + pickle on a worker thread (overlapping across
            # images; the GIL is released in the tensor copy/cast), then the
            # cheap zmq send back ON the loop -- send_to_scheduler is a
            # zmq.asyncio socket whose send() needs the running loop (and this
            # keeps it single-writer). Fire-and-forget like the legacy
            # send_pyobj (the returned Future is intentionally dropped).
            payload = await asyncio.to_thread(
                self._parse_and_pickle, request, bootstrap_room
            )
            self.async_llm.engine_core_client.send_to_scheduler.send(payload)
        else:
            self._ingest(request, bootstrap_room)
        return tokenspeed_encoder_pb2.EncodeResponse(accepted=True)

    def _build_encode_request(self, request, bootstrap_room):
        """Proto -> engine EncodeRequest (the expensive per-image parse)."""
        items = []
        if request.HasField("mm_inputs"):
            items = self._items_from_proto(request.mm_inputs, bootstrap_room)

        EncodeRequest = _lazy_encode_request()
        return EncodeRequest(
            request_id=request.request_id or uuid.uuid4().hex,
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
            bootstrap_room=bootstrap_room,
            items=items,
        )

    def _parse_and_pickle(self, request, bootstrap_room) -> bytes:
        """Worker-thread half of the off-loop ingest: parse + pickle. The
        pickled bytes match what send_pyobj would produce, so the scheduler's
        recv_pyobj is unchanged."""
        import pickle

        encode_request = self._build_encode_request(request, bootstrap_room)
        return pickle.dumps(encode_request, protocol=pickle.DEFAULT_PROTOCOL)

    def _ingest(self, request, bootstrap_room) -> None:
        """Legacy on-loop ingest (parse + submit on the event loop)."""
        encode_request = self._build_encode_request(request, bootstrap_room)
        self.async_llm.submit_encode(encode_request)

    async def shutdown(self) -> None:
        """No persistent per-request state to drain (encode is fire-and-forget)."""
        return None
