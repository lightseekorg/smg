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

    @staticmethod
    def _build_output_logprobs(token_id, logprobs_array, num_logprobs):
        """Build OutputLogProbs proto from an mlx logprobs array."""
        if num_logprobs is None:
            return None

        import mlx.core as mx

        token_logprob = logprobs_array[token_id].item()

        top_k = min(num_logprobs, logprobs_array.shape[0])
        top_indices = mx.argpartition(logprobs_array, kth=-top_k)[-top_k:]
        top_values = logprobs_array[top_indices]
        sort_order = mx.argsort(top_values)[::-1]
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]

        top_logprobs = vllm_engine_pb2.TopLogProbs(
            token_ids=[int(i) for i in top_indices.tolist()],
            values=[float(v) for v in top_values.tolist()],
        )

        return vllm_engine_pb2.OutputLogProbs(
            token_ids=[token_id],
            token_logprobs=[token_logprob],
            top_logprobs=[top_logprobs],
        )

    @staticmethod
    def _chunk_response(token_ids, prompt_tokens, completion_tokens, cached_tokens, index, output_logprobs=None):
        """Build a GenerateStreamChunk response."""
        chunk = vllm_engine_pb2.GenerateStreamChunk(
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            index=index,
        )
        if output_logprobs is not None:
            chunk.output_logprobs.CopyFrom(output_logprobs)
        return vllm_engine_pb2.GenerateResponse(chunk=chunk)

    @staticmethod
    def _complete_response(output_ids, finish_reason, prompt_tokens, completion_tokens, cached_tokens, index, output_logprobs=None, matched_token_id=None):
        """Build a GenerateComplete response."""
        kwargs = {}
        if matched_token_id is not None:
            kwargs["matched_token_id"] = matched_token_id

        complete = vllm_engine_pb2.GenerateComplete(
            output_ids=output_ids,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            index=index,
            **kwargs,
        )
        if output_logprobs is not None:
            complete.output_logprobs.CopyFrom(output_logprobs)
        return vllm_engine_pb2.GenerateResponse(complete=complete)
