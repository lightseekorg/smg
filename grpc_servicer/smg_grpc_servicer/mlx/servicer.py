"""
MLX Engine gRPC Servicer

Implements the VllmEngine proto service backed by mlx-lm's BatchGenerator
for Apple Silicon inference.
"""

import asyncio
import grpc
import hashlib
import io
import logging
import os
import threading
import time
import zipfile

from mlx_lm.generate import SequenceStateMachine
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc

logger = logging.getLogger(__name__)


class MlxEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """gRPC servicer implementing the VllmEngine service for MLX backends."""

    def __init__(self, batch_generator, model_path, model_dir, model_config, eos_token_ids, start_time):
        self.batch_generator = batch_generator
        self.model_path = model_path
        self.model_dir = model_dir
        self.model_config = model_config
        self._eos_token_ids = eos_token_ids
        self.start_time = start_time
        self._active_requests = 0
        self._request_uid_map = {}
        self._uid_queues = {}
        self._shutdown_event = asyncio.Event()
        self._loop = None
        self._gen_thread = None
        logger.info("MlxEngineServicer initialized for model %s", model_path)

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

    def start_generation_loop(self):
        self._loop = asyncio.get_event_loop()
        self._gen_thread = threading.Thread(
            target=self._generation_loop, daemon=True, name="mlx-gen-loop"
        )
        self._gen_thread.start()
        logger.info("Generation loop started")

    def stop_generation_loop(self):
        self._shutdown_event.set()
        if self._gen_thread and self._gen_thread.is_alive():
            self._gen_thread.join(timeout=5.0)
        logger.info("Generation loop stopped")

    def _generation_loop(self):
        _stream_ctx = None
        try:
            import mlx.core as _mx
            from mlx_lm.generate import generation_stream
            _stream_ctx = lambda: _mx.stream(generation_stream)
        except ImportError:
            pass

        while not self._shutdown_event.is_set():
            try:
                if _stream_ctx is not None:
                    with _stream_ctx():
                        prompt_responses, gen_responses = self.batch_generator.next()
                else:
                    prompt_responses, gen_responses = self.batch_generator.next()
            except Exception:
                logger.exception("Error in generation loop")
                continue

            if not prompt_responses and not gen_responses:
                import time as _time
                _time.sleep(0.001)
                continue

            for r in gen_responses:
                queue = self._uid_queues.get(r.uid)
                if queue is not None:
                    self._loop.call_soon_threadsafe(queue.put_nowait, r)
                if r.finish_reason is not None:
                    try:
                        self.batch_generator.remove([r.uid])
                    except Exception:
                        logger.exception("Error removing uid %d", r.uid)

    async def Generate(self, request, context):
        request_id = request.request_id
        try:
            input_type = request.WhichOneof("input")
            if input_type != "tokenized":
                raise ValueError("MLX servicer requires tokenized input")

            token_ids = list(request.tokenized.input_ids)
            sp = request.sampling_params

            if sp.n > 1:
                raise ValueError("n > 1 not supported by MLX backend")
            constraint = sp.WhichOneof("constraint")
            if constraint is not None:
                raise ValueError(f"Structured output ({constraint}) not supported by MLX backend")
            if sp.HasField("prompt_logprobs"):
                raise ValueError("prompt_logprobs not supported by MLX backend")

            sampler = self._build_sampler(sp)
            logits_processors = self._build_logits_processors(sp)
            state_machine = self._build_state_machine(sp, self._eos_token_ids)
            max_tokens = sp.max_tokens if sp.HasField("max_tokens") else 256
            num_logprobs = sp.logprobs if sp.HasField("logprobs") else None

            if sp.HasField("seed"):
                import mlx.core as mx
                mx.random.seed(sp.seed)

            uids = self.batch_generator.insert(
                prompts=[token_ids],
                max_tokens=[max_tokens],
                samplers=[sampler],
                logits_processors=[logits_processors],
                state_machines=[state_machine],
            )
            uid = uids[0]
            self._request_uid_map[request_id] = uid

            queue = asyncio.Queue()
            self._uid_queues[uid] = queue
            self._active_requests += 1
            prompt_tokens = len(token_ids)

            try:
                if request.stream:
                    while True:
                        r = await queue.get()
                        output_logprobs = self._build_output_logprobs(
                            r.token, r.logprobs, num_logprobs
                        )

                        if r.finish_reason is not None:
                            yield self._chunk_response(
                                token_ids=[r.token],
                                prompt_tokens=prompt_tokens,
                                completion_tokens=1,
                                cached_tokens=0, index=0,
                                output_logprobs=output_logprobs,
                            )
                            matched_token_id = None
                            if r.match_sequence:
                                matched_token_id = r.match_sequence[0] if len(r.match_sequence) == 1 else None
                            yield self._complete_response(
                                output_ids=[], finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=0, cached_tokens=0, index=0,
                                matched_token_id=matched_token_id,
                            )
                            break
                        else:
                            yield self._chunk_response(
                                token_ids=[r.token],
                                prompt_tokens=prompt_tokens,
                                completion_tokens=1,
                                cached_tokens=0, index=0,
                                output_logprobs=output_logprobs,
                            )
                else:
                    all_output_ids = []
                    while True:
                        r = await queue.get()
                        all_output_ids.append(r.token)
                        if r.finish_reason is not None:
                            matched_token_id = None
                            if r.match_sequence:
                                matched_token_id = r.match_sequence[0] if len(r.match_sequence) == 1 else None
                            yield self._complete_response(
                                output_ids=all_output_ids,
                                finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=len(all_output_ids),
                                cached_tokens=0, index=0,
                                matched_token_id=matched_token_id,
                            )
                            break
            finally:
                self._active_requests -= 1
                self._request_uid_map.pop(request_id, None)
                self._uid_queues.pop(uid, None)

        except ValueError as e:
            logger.warning("Generate invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Generate failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Abort(self, request, context):
        for request_id in request.request_ids:
            uid = self._request_uid_map.pop(request_id, None)
            if uid is not None:
                self._uid_queues.pop(uid, None)
                try:
                    self.batch_generator.remove([uid])
                except Exception:
                    logger.warning("Failed to remove uid %d for request %s", uid, request_id)
        return vllm_engine_pb2.AbortResponse()

    async def HealthCheck(self, request, context):
        return vllm_engine_pb2.HealthCheckResponse(healthy=True, message="OK")

    async def Embed(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Embed not supported by MLX backend")

    async def GetTokenizer(self, request, context):
        try:
            zip_bytes, sha256 = self._build_tokenizer_zip(self.model_dir)
            async for chunk in self._async_chunk_tokenizer(zip_bytes, sha256):
                yield chunk
        except Exception as e:
            logger.exception("GetTokenizer failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def _async_chunk_tokenizer(self, zip_bytes, sha256):
        for chunk in self._chunk_tokenizer_zip(zip_bytes, sha256):
            yield chunk

    async def SubscribeKvEvents(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "SubscribeKvEvents not supported by MLX backend")
        # yield is never reached but makes this an async generator for gRPC streaming
        yield  # pragma: no cover
