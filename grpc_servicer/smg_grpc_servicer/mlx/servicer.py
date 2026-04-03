"""
MLX Engine gRPC Servicer

Implements the VllmEngine proto service backed by mlx-lm's BatchGenerator
for Apple Silicon inference.
"""

import grpc
import hashlib
import io
import logging
import os
import time
import zipfile

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

    _TOKENIZER_FILES = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    }

    @staticmethod
    def _build_tokenizer_zip(model_dir):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in sorted(os.listdir(model_dir)):
                if filename in MlxEngineServicer._TOKENIZER_FILES:
                    filepath = os.path.join(model_dir, filename)
                    if os.path.isfile(filepath):
                        zf.write(filepath, filename)
        zip_bytes = buf.getvalue()
        sha256 = hashlib.sha256(zip_bytes).hexdigest()
        return zip_bytes, sha256

    @staticmethod
    def _chunk_tokenizer_zip(zip_bytes, sha256, chunk_size=512 * 1024):
        from smg_grpc_proto.generated import common_pb2
        total = len(zip_bytes)
        offset = 0
        while offset < total:
            end = min(offset + chunk_size, total)
            is_last = end == total
            yield common_pb2.GetTokenizerChunk(
                data=zip_bytes[offset:end],
                sha256=sha256 if is_last else "",
            )
            offset = end

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetModelInfoResponse:
        config = self.model_config

        eos = config.get("eos_token_id")
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = eos
        else:
            eos_token_ids = []

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=self.model_path,
            is_generation=True,
            max_context_length=config.get("max_position_embeddings", 0),
            vocab_size=config.get("vocab_size", 0),
            supports_vision=False,
            served_model_name=self.model_path,
            tokenizer_path=self.model_path,
            model_type=config.get("model_type", ""),
            architectures=config.get("architectures", []),
            eos_token_ids=eos_token_ids,
            pad_token_id=config.get("pad_token_id") or 0,
            bos_token_id=config.get("bos_token_id") or 0,
            max_req_input_len=config.get("max_position_embeddings", 0),
        )

    async def GetServerInfo(
        self,
        request: vllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetServerInfoResponse:
        return vllm_engine_pb2.GetServerInfoResponse(
            server_type="mlx-grpc",
            active_requests=self._active_requests,
            uptime_seconds=time.time() - self.start_time,
            kv_connector="",
            kv_role="",
        )
