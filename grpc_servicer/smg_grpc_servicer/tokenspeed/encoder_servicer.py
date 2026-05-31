"""TokenSpeed EPD encode servicer.

Receives ``Encode`` RPCs from the gateway and forwards them to a vision-only
encode worker (the engine's ``run_encode_loop``) over the same AsyncLLM
scheduler-input channel the LM uses. The encode worker runs the vision tower and
ships the resulting image embeddings to prefill workers over Mooncake; this
servicer only acks (the embeddings never flow back through the gateway).
"""

from __future__ import annotations

import logging
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

        self.async_llm.auto_create_handle_loop()
        logger.info("TokenSpeedEncoderServicer initialized")

    def _resolve_merge_size(self) -> int:
        hf_config = getattr(self.async_llm.model_config, "hf_config", None)
        vision_config = getattr(hf_config, "vision_config", None)
        return int(getattr(vision_config, "spatial_merge_size", 2) or 2)

    def _items_from_proto(self, mm_inputs):
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

        feature = TokenSpeedSchedulerServicer._tensor_from_proto(
            mm_inputs.pixel_values, cast_to=model_dtype
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
        items = []
        if request.HasField("mm_inputs"):
            items = self._items_from_proto(request.mm_inputs)

        # The gateway sends one image per RPC, so one EncodeItemAssignment / room.
        # (Multiple items per RPC would need per-item rooms; not used today.)
        bootstrap_room = request.items[0].bootstrap_room if request.items else 0

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
