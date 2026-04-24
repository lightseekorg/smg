"""
MLX Engine gRPC Servicer

Implements the MlxEngine proto service backed by mlx-lm's BatchGenerator
for Apple Silicon inference.
"""

import asyncio
import hashlib
import io
import logging
import os
import threading
import time
import zipfile

import grpc
import mlx.core as mx
from mlx_lm.generate import SequenceStateMachine, generation_stream
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from smg_grpc_proto import mlx_engine_pb2, mlx_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)


class MlxEngineServicer(mlx_engine_pb2_grpc.MlxEngineServicer):
    """gRPC servicer implementing the MlxEngine service for MLX backends.

    Thread-safety model
    -------------------
    mlx-lm's ``BatchGenerator`` is not thread-safe: ``next()`` mutates
    ``_prompt_batch`` / ``_generation_batch`` / ``_unprocessed_sequences``
    while running mlx kernels (which release the GIL), and ``remove()``
    rebuilds those same structures. This servicer runs ``next()`` on a
    background thread and services ``insert()`` / ``remove()`` from the
    asyncio event loop, so we need synchronization.

    ``self._gen_lock`` is acquired by:
      * the gen thread for the whole ``next() + dispatch + finished-remove``
        block (one critical section per loop iteration — keeps the batch
        snapshot consistent);
      * the event loop around ``insert() + _uid_queues[uid] = queue`` in
        ``Generate`` (fast path, also closes the register-queue-before-first-
        response race);
      * the event loop around ``remove()`` in ``Abort`` (rare).

    ``insert()`` itself only appends to a ``deque`` and increments a counter
    (both atomic under the GIL), but the lock still wraps it so the uid
    can't become visible to the gen thread before its queue is registered.

    Cost model: the event loop can block up to one ``next()`` step
    (~10-50 ms on M-series) while the gen thread holds the lock. Acceptable
    for single-worker Mac inference; if you need 1000+ concurrent reqs/s,
    refactor to a command-queue / actor model (see vLLM's AsyncLLMEngine).
    """

    def __init__(
        self, batch_generator, model_path, model_dir, model_config, eos_token_ids, start_time
    ):
        self.batch_generator = batch_generator
        self.model_path = model_path
        self.model_dir = model_dir
        self.model_config = model_config
        self._eos_token_ids = eos_token_ids
        self.start_time = start_time
        self._active_requests = 0
        self._request_uid_map = {}
        self._uid_queues = {}
        self._shutdown_event = threading.Event()
        self._loop = None
        self._gen_thread = None
        # Protects mlx-lm BatchGenerator state + self._uid_queues against
        # the background gen thread. See class docstring.
        self._gen_lock = threading.Lock()
        # Resolve context length once — config doesn't change at runtime,
        # and Generate was previously scanning these keys on every request.
        self._ctx_limit = 0
        for key in ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length"):
            val = model_config.get(key)
            if isinstance(val, int) and val > 0:
                self._ctx_limit = val
                break
        logger.info("MlxEngineServicer initialized for model %s", model_path)

    @staticmethod
    def _build_sampler(sampling_params):
        """Convert proto SamplingParams to an mlx-lm sampler callable."""
        # When temperature is unset, default to 1.0 to match vLLM/SGLang/TRT-LLM
        # behavior. mlx-lm's make_sampler defaults to 0.0 (greedy), which would
        # silently diverge for requests that omit temperature.
        temp = sampling_params.temperature if sampling_params.HasField("temperature") else 1.0
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
    def _matched_stop_token(response):
        """Return the matched stop token id if the response matched a single-token stop."""
        ms = response.match_sequence
        return ms[0] if ms and len(ms) == 1 else None

    @staticmethod
    def _build_output_logprobs(token_id, logprobs_array, num_logprobs):
        """Build OutputLogProbs proto from an mlx logprobs array."""
        # num_logprobs == 0 would make top_k == 0 and `[-0:]` would slice the
        # entire vocabulary — guard explicitly.
        if num_logprobs is None or num_logprobs <= 0:
            return None

        token_logprob = logprobs_array[token_id].item()

        top_k = min(num_logprobs, logprobs_array.shape[0])
        top_indices = mx.argpartition(logprobs_array, kth=-top_k)[-top_k:]
        top_values = logprobs_array[top_indices]
        sort_order = mx.argsort(top_values)[::-1]
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]

        top_logprobs = mlx_engine_pb2.TopLogProbs(
            token_ids=[int(i) for i in top_indices.tolist()],
            values=[float(v) for v in top_values.tolist()],
        )

        return mlx_engine_pb2.OutputLogProbs(
            token_ids=[token_id],
            token_logprobs=[token_logprob],
            top_logprobs=[top_logprobs],
        )

    @staticmethod
    def _chunk_response(
        token_ids, prompt_tokens, completion_tokens, cached_tokens, index, output_logprobs=None
    ):
        """Build a GenerateStreamChunk response."""
        chunk = mlx_engine_pb2.GenerateStreamChunk(
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            index=index,
        )
        if output_logprobs is not None:
            chunk.output_logprobs.CopyFrom(output_logprobs)
        return mlx_engine_pb2.GenerateResponse(chunk=chunk)

    @staticmethod
    def _complete_response(
        output_ids,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        index,
        output_logprobs=None,
        matched_token_id=None,
    ):
        """Build a GenerateComplete response."""
        kwargs = {}
        if matched_token_id is not None:
            kwargs["matched_stop_token_id"] = matched_token_id

        complete = mlx_engine_pb2.GenerateComplete(
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
        return mlx_engine_pb2.GenerateResponse(complete=complete)

    _TOKENIZER_FILES = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tiktoken.model",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
        # Chat template sidecars (newer HF convention, transformers>=4.43).
        # Required for models like Gemma 4 whose tokenizer_config.json does
        # NOT embed chat_template; router-side discover_chat_template_in_dir
        # relies on these being present in the bundle.
        "chat_template.json",
        "chat_template.jinja",
    }
    # Additional extension-based matches for tiktoken-style BPE artifacts
    # (e.g. `cl100k_base.tiktoken`). The router-side Rust tokenizer loader
    # accepts these as valid directory tokenizers.
    _TOKENIZER_SUFFIXES = (".tiktoken",)

    @staticmethod
    def _build_tokenizer_zip(model_dir):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in sorted(os.listdir(model_dir)):
                matched = filename in MlxEngineServicer._TOKENIZER_FILES or filename.endswith(
                    MlxEngineServicer._TOKENIZER_SUFFIXES
                )
                if matched:
                    filepath = os.path.join(model_dir, filename)
                    if os.path.isfile(filepath):
                        zf.write(filepath, filename)
        zip_bytes = buf.getvalue()
        sha256 = hashlib.sha256(zip_bytes).hexdigest()
        return zip_bytes, sha256

    @staticmethod
    def _chunk_tokenizer_zip(zip_bytes, sha256, chunk_size=512 * 1024):
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
        request: mlx_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> mlx_engine_pb2.GetModelInfoResponse:
        config = self.model_config

        # Reuse the resolved EOS IDs so GetModelInfo agrees with the stop
        # behavior we actually apply in generation (server.py falls back to
        # tokenizer-derived IDs when config.json has none).
        eos_token_ids = list(self._eos_token_ids)

        # Use the pre-resolved context limit so GetModelInfo reports the
        # same value Generate enforces (config keys vary across model
        # families — see __init__).
        return mlx_engine_pb2.GetModelInfoResponse(
            model_path=self.model_path,
            is_generation=True,
            max_context_length=self._ctx_limit,
            vocab_size=config.get("vocab_size", 0),
            served_model_name=self.model_path,
            model_type=config.get("model_type", ""),
            architectures=config.get("architectures", []),
            eos_token_ids=eos_token_ids,
            pad_token_id=config.get("pad_token_id") or 0,
            bos_token_id=config.get("bos_token_id") or 0,
            max_req_input_len=self._ctx_limit,
        )

    async def GetServerInfo(
        self,
        request: mlx_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> mlx_engine_pb2.GetServerInfoResponse:
        return mlx_engine_pb2.GetServerInfoResponse(
            server_type="mlx-grpc",
            active_requests=self._active_requests,
            uptime_seconds=time.time() - self.start_time,
        )

    def start_generation_loop(self):
        self._loop = asyncio.get_running_loop()
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
        while not self._shutdown_event.is_set():
            prompt_responses: list = []
            gen_responses: list = []
            try:
                # Single critical section: next() + dispatch + finished-remove.
                # Holding the lock across dispatch keeps the batch snapshot
                # consistent with the responses we just produced; event-loop
                # insert/remove serializes naturally against this block.
                with self._gen_lock:
                    with mx.stream(generation_stream):
                        prompt_responses, gen_responses = self.batch_generator.next()

                    for r in gen_responses:
                        queue = self._uid_queues.get(r.uid)
                        if queue is not None:
                            self._loop.call_soon_threadsafe(queue.put_nowait, r)
                        if r.finish_reason is not None:
                            try:
                                self.batch_generator.remove([r.uid])
                            except Exception:
                                logger.exception("Error removing uid %d", r.uid)
            except Exception:
                logger.exception("Error in generation loop")
                continue

            if not prompt_responses and not gen_responses:
                # Idle — release the lock and briefly sleep so the event
                # loop can slot in insert/remove work without contention.
                time.sleep(0.001)

    async def Generate(self, request, context):
        request_id = request.request_id
        try:
            input_type = request.WhichOneof("input")
            if input_type != "tokenized":
                raise ValueError("MLX servicer requires tokenized input")

            token_ids = list(request.tokenized.input_ids)
            sp = request.sampling_params

            sampler = self._build_sampler(sp)
            logits_processors = self._build_logits_processors(sp)
            state_machine = self._build_state_machine(sp, self._eos_token_ids)
            # When max_tokens is unset, cap at remaining context (matches
            # vLLM/SGLang semantics: unbounded within model limits, not a
            # silent 256-token truncation). Fall back to 256 if the model
            # config didn't advertise a context length.
            if sp.HasField("max_tokens"):
                max_tokens = sp.max_tokens
            elif self._ctx_limit > 0:
                max_tokens = max(self._ctx_limit - len(token_ids), 1)
            else:
                max_tokens = 256
            num_logprobs = sp.logprobs if sp.HasField("logprobs") else None

            if sp.HasField("seed"):
                mx.random.seed(sp.seed)

            queue: asyncio.Queue = asyncio.Queue()
            # Insert + queue registration must be atomic against the gen
            # thread's next()+remove block so a one-step completion can't
            # be dispatched and removed before its queue is visible.
            with self._gen_lock:
                uids = self.batch_generator.insert(
                    prompts=[token_ids],
                    max_tokens=[max_tokens],
                    samplers=[sampler],
                    logits_processors=[logits_processors],
                    state_machines=[state_machine],
                )
                uid = uids[0]
                self._uid_queues[uid] = queue
            self._request_uid_map[request_id] = uid
            self._active_requests += 1
            prompt_tokens = len(token_ids)

            try:
                if request.stream:
                    completion_tokens = 0
                    while True:
                        r = await queue.get()
                        if r is None:
                            # Sentinel from Abort — terminate the stream.
                            break
                        completion_tokens += 1
                        yield self._chunk_response(
                            token_ids=[r.token],
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=0,
                            index=0,
                            output_logprobs=self._build_output_logprobs(
                                r.token, r.logprobs, num_logprobs
                            ),
                        )
                        if r.finish_reason is not None:
                            yield self._complete_response(
                                output_ids=[],
                                finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                cached_tokens=0,
                                index=0,
                                matched_token_id=self._matched_stop_token(r),
                            )
                            break
                else:
                    all_output_ids = []
                    # Aggregate per-token logprobs across the whole sequence so
                    # the final GenerateComplete carries logprobs for every
                    # generated token (not just the last step).
                    agg_token_ids: list[int] = []
                    agg_token_logprobs: list[float] = []
                    agg_top: list = []
                    while True:
                        r = await queue.get()
                        if r is None:
                            # Sentinel from Abort — terminate without emitting.
                            break
                        all_output_ids.append(r.token)
                        step = self._build_output_logprobs(r.token, r.logprobs, num_logprobs)
                        if step is not None:
                            agg_token_ids.extend(step.token_ids)
                            agg_token_logprobs.extend(step.token_logprobs)
                            agg_top.extend(step.top_logprobs)
                        if r.finish_reason is not None:
                            seq_logprobs = None
                            if agg_token_ids:
                                seq_logprobs = mlx_engine_pb2.OutputLogProbs(
                                    token_ids=agg_token_ids,
                                    token_logprobs=agg_token_logprobs,
                                    top_logprobs=agg_top,
                                )
                            yield self._complete_response(
                                output_ids=all_output_ids,
                                finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=len(all_output_ids),
                                cached_tokens=0,
                                index=0,
                                output_logprobs=seq_logprobs,
                                matched_token_id=self._matched_stop_token(r),
                            )
                            break
            finally:
                self._active_requests -= 1
                self._request_uid_map.pop(request_id, None)
                self._uid_queues.pop(uid, None)
                # Ensure the backend request is removed on any Generate exit
                # (client disconnect, deadline, CancelledError, unexpected
                # exception). Without this, a cancelled request keeps decoding
                # until its own stop/max-tokens condition, wasting batch slots.
                # Safe to double-call: gen thread's finish-path remove and
                # Abort's remove both land here if racing, and remove() is
                # idempotent-ish (raises on unknown uid, which we swallow).
                with self._gen_lock:
                    try:
                        self.batch_generator.remove([uid])
                    except Exception:
                        # Already removed by the gen thread or Abort — fine.
                        pass

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
                queue = self._uid_queues.pop(uid, None)
                if queue is not None:
                    # Drain already-buffered tokens so Generate stops emitting
                    # output immediately rather than flushing a backlog of
                    # stale chunks before seeing the sentinel.
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    # Wake the Generate waiter blocked on queue.get() so it
                    # exits cleanly instead of hanging until transport cancel.
                    queue.put_nowait(None)
                # remove() races with the gen thread's next() without the
                # lock — see class docstring.
                with self._gen_lock:
                    try:
                        self.batch_generator.remove([uid])
                    except Exception:
                        logger.warning("Failed to remove uid %d for request %s", uid, request_id)
        return mlx_engine_pb2.AbortResponse()

    async def HealthCheck(self, request, context):
        # Reflect actual servicer state so the router can stop routing to us
        # when the generation thread is dead or we're shutting down.
        if self._shutdown_event.is_set():
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="servicer shutting down"
            )
        if self._gen_thread is None:
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="generation loop not started"
            )
        if not self._gen_thread.is_alive():
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="generation thread exited"
            )
        return mlx_engine_pb2.HealthCheckResponse(healthy=True, message="OK")

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
