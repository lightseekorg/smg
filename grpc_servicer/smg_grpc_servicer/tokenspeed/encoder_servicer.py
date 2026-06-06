"""TokenSpeed EPD encode servicer.

Receives ``Encode`` RPCs from the gateway and forwards them to a vision-only
encode worker (the engine's ``run_encode_loop``) over the same AsyncLLM
scheduler-input channel the LM uses. The encode worker runs the vision tower and
ships the resulting image embeddings to prefill workers over Mooncake; this
servicer only acks (the embeddings never flow back through the gateway).
"""

from __future__ import annotations

import logging
import os
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
    from tokenspeed.runtime.pd.encode_worker import EncodeRequest

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
        self.server_args = server_args
        self.scheduler_info = scheduler_info
        self.health_servicer = health_servicer

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
        self._rdma_md_sent = set()  # (ip, port) we've already send_local_metadata'd
        if os.environ.get("SMG_MM_PIXEL_RDMA") in ("1", "true"):
            try:
                try:
                    from nixl_cu13._api import nixl_agent, nixl_agent_config
                except ImportError:
                    from nixl._api import nixl_agent, nixl_agent_config

                # Initiator side: a listen thread (ephemeral port) is needed so the
                # bidirectional metadata exchange with the gateway's listener works.
                self._nixl_agent = nixl_agent(
                    f"smg-encode-{self._bootstrap_host}-{self._bootstrap_port}",
                    nixl_agent_config(
                        enable_listen_thread=True, listen_port=0, backends=["UCX"]
                    ),
                )
                logger.info("EPD RDMA: encode NIXL puller agent up")
            except Exception as e:  # noqa: BLE001
                logger.error("EPD RDMA: NIXL agent init failed (%s); inline only", e)

        self.async_llm.auto_create_handle_loop()
        logger.info("TokenSpeedEncoderServicer initialized")

    def _feature_from_remote(self, td, room: int, cast_to):
        """PULL the pixel tensor from the gateway's exported NIXL memory (one-sided
        READ), then reconstruct it exactly like the inline path.

        Descriptor wire format (gateway rdma.rs): [addr u64 LE][port u16 LE][ip utf8].
        Cross-node NIXL needs the bidirectional metadata exchange against the
        gateway's listener (fetch_remote_metadata + send_local_metadata) BEFORE the
        one-sided READ -- a one-way add_remote_agent hits the gateway's non-listening
        ephemeral worker port ("Connection refused"). Then READ nbytes from addr,
        tag the transfer with the room so the gateway frees the MR, length-assert.
        """
        import time

        import numpy as np
        import torch

        agent = self._nixl_agent
        desc = bytes(td.remote.descriptor)
        nbytes = int(td.remote.nbytes)
        if len(desc) < 10:
            raise ValueError("remote descriptor too short")
        remote_addr = int.from_bytes(desc[:8], "little")
        port = int.from_bytes(desc[8:10], "little")
        ip = desc[10:].decode()
        gw_name = "smg-gateway-encode"

        # Bidirectional metadata exchange via the gateway's listener. Re-fetch each
        # call so a newly-registered region becomes visible; send our own md once per
        # endpoint so the gateway can complete the connection wireup.
        agent.fetch_remote_metadata(gw_name, ip, port)
        if (ip, port) not in self._rdma_md_sent:
            agent.send_local_metadata(ip, port)
            self._rdma_md_sent.add((ip, port))

        landing = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        laddr = landing.data_ptr()
        reg = agent.get_reg_descs([(laddr, nbytes, 0, "")], "DRAM")
        agent.register_memory(reg)
        try:
            remote = agent.get_xfer_descs([(remote_addr, nbytes, 0)], "DRAM")
            # Wait for this region's metadata to load before building the xfer.
            ready = False
            for _ in range(5000):
                if agent.check_remote_metadata(gw_name, remote):
                    ready = True
                    break
                time.sleep(0.001)
            if not ready:
                raise RuntimeError(f"NIXL remote metadata not ready room={room}")
            local = agent.get_xfer_descs([(laddr, nbytes, 0)], "DRAM")
            h = agent.initialize_xfer("READ", local, remote, gw_name, str(room).encode())
            state = agent.transfer(h)
            while state in ("PROC", "IN_PROG"):
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
            raw = landing.numpy().tobytes()
        finally:
            agent.deregister_memory(reg)

        if td.dtype == "bfloat16":
            t = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            ).view(torch.bfloat16)
        else:
            t = torch.from_numpy(
                np.frombuffer(raw, dtype=np.dtype(td.dtype)).reshape(shape)
            )
        if cast_to is not None and t.dtype != cast_to and t.is_floating_point():
            return t.to(cast_to)
        return t.clone()

    def _resolve_merge_size(self) -> int:
        hf_config = getattr(self.async_llm.model_config, "hf_config", None)
        vision_config = getattr(hf_config, "vision_config", None)
        return int(getattr(vision_config, "spatial_merge_size", 2) or 2)

    def _items_from_proto(self, mm_inputs, bootstrap_room: int = 0):
        """Reconstruct the engine MultimodalDataItem(s) for the encode worker.

        Unlike the prefill leg, the encode worker NEEDS pixel_values (it runs the
        tower). It also needs each item's post-merge token count so the executor
        can split the tower output; the gateway ships grid_thw but not
        placeholders to encode, so derive the count from grid_thw and set it as
        the item's single offset span (the offset positions are irrelevant to the
        encode side, only the count matters).
        """
        Modality, MultimodalDataItem = _lazy_mm_item()
        model_dtype = getattr(self.async_llm.model_config, "dtype", None)

        td = mm_inputs.pixel_values
        if td.WhichOneof("payload") == "remote":
            # EPD RDMA: PULL the pixels from the gateway's exported NIXL memory.
            feature = self._feature_from_remote(td, bootstrap_room, model_dtype)
        else:
            feature = TokenSpeedSchedulerServicer._tensor_from_proto(
                td, cast_to=model_dtype
            )
        model_specific = {
            name: TokenSpeedSchedulerServicer._tensor_from_proto(td, cast_to=model_dtype)
            for name, td in mm_inputs.model_specific_tensors.items()
        }

        grid = model_specific.get("image_grid_thw")
        if grid is None:
            raise ValueError("encode request is missing image_grid_thw")
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
            feature=feature,
            model_specific_data=model_specific,
            offsets=offsets,
        )
        item.set_pad_value()
        return [item]

    async def Encode(self, request, context):
        # The gateway sends one image per RPC, so one EncodeItemAssignment / room.
        # (Multiple items per RPC would need per-item rooms; not used today.) The
        # room also tags the RDMA free-notif, so resolve it before reading pixels.
        bootstrap_room = request.items[0].bootstrap_room if request.items else 0

        items = []
        if request.HasField("mm_inputs"):
            items = self._items_from_proto(request.mm_inputs, bootstrap_room)

        EncodeRequest = _lazy_encode_request()
        encode_request = EncodeRequest(
            request_id=request.request_id or uuid.uuid4().hex,
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
            bootstrap_room=bootstrap_room,
            items=items,
        )
        self.async_llm.submit_encode(encode_request)
        return tokenspeed_encoder_pb2.EncodeResponse(accepted=True)

    async def shutdown(self) -> None:
        """No persistent per-request state to drain (encode is fire-and-forget)."""
        return None
