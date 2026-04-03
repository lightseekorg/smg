"""
MLX Engine gRPC Servicer

Implements the VllmEngine proto service backed by mlx-lm's BatchGenerator
for Apple Silicon inference.
"""

import logging

from mlx_lm.generate import SequenceStateMachine
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc

logger = logging.getLogger(__name__)


class MlxEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """gRPC servicer implementing the VllmEngine service for MLX backends."""

    @staticmethod
    def _build_sampler(sampling_params):
        """Convert proto SamplingParams to an mlx-lm sampler callable."""
        temp = sampling_params.temperature if sampling_params.HasField("temperature") else 0.0
        return make_sampler(
            temp=temp,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
        )

    @staticmethod
    def _build_logits_processors(sampling_params):
        """Convert proto SamplingParams to a list of mlx-lm logits processors."""
        logit_bias = dict(sampling_params.logit_bias) if sampling_params.logit_bias else None
        rep_pen = sampling_params.repetition_penalty if sampling_params.repetition_penalty else None
        freq_pen = sampling_params.frequency_penalty if sampling_params.frequency_penalty else None
        pres_pen = sampling_params.presence_penalty if sampling_params.presence_penalty else None
        return make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=rep_pen,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
        )

    @staticmethod
    def _build_state_machine(sampling_params, eos_token_ids):
        """Build a SequenceStateMachine from stop_token_ids and EOS tokens."""
        stop_sequences = []

        if not sampling_params.ignore_eos:
            for eos_id in eos_token_ids:
                stop_sequences.append(((eos_id,), None))

        for tid in sampling_params.stop_token_ids:
            stop_sequences.append(((tid,), None))

        if not stop_sequences:
            return SequenceStateMachine()

        return SequenceStateMachine(
            transitions={"normal": stop_sequences},
            initial="normal",
        )
