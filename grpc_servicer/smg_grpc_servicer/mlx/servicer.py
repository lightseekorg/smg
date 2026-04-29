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


def _set_future_result_safe(future: asyncio.Future, result) -> None:
    """``future.set_result`` that no-ops on already-completed futures.

    Used by the gen thread to wake a Generate awaiter via
    ``call_soon_threadsafe``. Generate's task may be cancelled (which
    cancels the future) between the time the gen thread schedules this
    callback and the time the loop runs it; ``set_result`` would then
    raise ``InvalidStateError``. Cleanup of the inserted batch slot is
    handled by Generate's CancelledError path.
    """
    if not future.done():
        future.set_result(result)


def _set_future_exception_safe(future: asyncio.Future, exc: BaseException) -> None:
    """``future.set_exception`` that no-ops on already-completed futures.

    Same race as :func:`_set_future_result_safe`.
    """
    if not future.done():
        future.set_exception(exc)


class _PendingRequest:
    """A Generate() call queued to enter the next fresh batch.

    Holds the inputs we need to feed BatchGenerator.insert() plus an
    asyncio.Future that the generation thread resolves with the assigned
    uid once the request actually enters the batch. Generate() awaits
    that future before it starts pulling tokens off ``queue``.
    """

    __slots__ = (
        "token_ids",
        "max_tokens",
        "sampler",
        "logits_processors",
        "state_machine",
        "queue",
        "uid_future",
        "request_id",
    )

    def __init__(
        self,
        token_ids,
        max_tokens,
        sampler,
        logits_processors,
        state_machine,
        queue,
        uid_future,
        request_id,
    ):
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.logits_processors = logits_processors
        self.state_machine = state_machine
        self.queue = queue
        self.uid_future = uid_future
        self.request_id = request_id


class MlxEngineServicer(mlx_engine_pb2_grpc.MlxEngineServicer):
    """gRPC servicer implementing the MlxEngine service for MLX backends.

    Concurrency model: drain-and-batch
    ----------------------------------
    mlx-lm's ``BatchGenerator`` does not support inserting new sequences
    while the active batch is in its decode phase: the rope offset cache
    is sized at the start of decode and a mid-decode ``insert()`` leaves
    it out of sync with the new batch shape, which surfaces as

        ValueError: [rope] offset must be a scalar or vector with N
            elements but has shape (N-1).

    inside ``mx.fast.rope`` on the next ``_step()``. mlx-lm.server avoids
    this by serving each batch to completion before accepting the next
    set of requests; we mirror that here.

    Flow:

      * Incoming ``Generate`` calls build a :class:`_PendingRequest` and
        push it onto ``self._pending``, then await ``uid_future``.
      * The generation thread's main loop, between iterations, checks
        whether the active batch has drained (``_active_uids`` empty).
        When it has, it drains ``_pending`` in one shot and feeds the
        whole list to a single ``BatchGenerator.insert()`` call — this
        keeps batching for concurrent arrivals while ensuring no insert
        ever happens during decode.
      * Each request's ``uid_future`` is resolved as soon as its uid is
        known, so ``Generate`` can register its uid for ``Abort`` lookup
        and start consuming tokens from its per-uid queue.

    Trade-off: a request arriving while a batch is mid-decode waits for
    that batch to drain before its first token. That's the same behavior
    as ``mlx-lm.server`` and is the correctness fix for the rope crash;
    re-introducing true dynamic batching is a separate optimization that
    requires fixes in mlx-lm's BatchGenerator.

    Thread-safety
    -------------
    ``self._gen_lock`` protects all mutations of the BatchGenerator and
    of ``_active_uids`` / ``_uid_queues``. It is acquired by:

      * the gen thread, around the whole drain-pending + ``next()`` +
        dispatch + finished-``remove()`` block (one critical section per
        loop iteration);
      * the event loop in ``Generate``'s ``finally`` for the
        client-disconnect cleanup ``remove()``;
      * the event loop in ``Abort`` for the in-flight ``remove()``.

    ``self._pending_lock`` protects the pending list/index and is held
    only briefly: append in ``Generate``, drain in the gen thread, pop
    in ``Abort``. The gen thread *does* nest it inside ``_gen_lock``
    during drain-and-fill (we hold ``_gen_lock`` for the whole
    iteration and grab ``_pending_lock`` only to swap pending into the
    batch). All other sites acquire just one of the two locks at a
    time, so this is the only nesting direction in the codebase —
    adding any path that acquires ``_gen_lock`` while already holding
    ``_pending_lock`` would deadlock.

    Cost model: the event loop can block up to one ``next()`` step
    (~10–50 ms on M-series) while the gen thread holds ``_gen_lock``.
    Acceptable for single-worker Mac inference; if you need
    1000+ concurrent req/s, refactor to a command-queue / actor model
    (see vLLM's AsyncLLMEngine).
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
        # Set of uids currently live in the BatchGenerator. Mutated only
        # under ``_gen_lock``. The gen thread inspects ``not _active_uids``
        # to decide whether it's safe to drain ``_pending`` into a fresh
        # ``insert()``.
        self._active_uids: set[int] = set()
        self._shutdown_event = threading.Event()
        self._loop = None
        self._gen_thread = None
        # Protects mlx-lm BatchGenerator state + ``_uid_queues`` +
        # ``_active_uids`` against the background gen thread. See class
        # docstring.
        self._gen_lock = threading.Lock()
        # Drain-and-batch state. New ``Generate`` calls land here and
        # wait for the gen thread to pull them into a fresh batch once
        # the previous batch drains. Index by request_id so ``Abort``
        # can cancel a request that hasn't entered the batch yet.
        self._pending: list[_PendingRequest] = []
        self._pending_by_request_id: dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()
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
            inserted_this_iter = False
            try:
                with self._gen_lock:
                    # Phase 1: drain-and-fill. _pending_lock is nested
                    # inside _gen_lock here — the only nesting site in
                    # the codebase (see class docstring).
                    if not self._active_uids:
                        with self._pending_lock:
                            batch = self._pending[:]
                            self._pending.clear()
                            # NOTE: do NOT pop from _pending_by_request_id
                            # yet — keeping pending entries indexed until
                            # insert() succeeds means Abort() can still
                            # cancel a request that lost its insert race.

                        # Filter out requests whose uid_future was already
                        # cancelled (client disconnected before drain).
                        batch = [p for p in batch if not p.uid_future.cancelled()]

                        if batch:
                            try:
                                uids = self.batch_generator.insert(
                                    prompts=[p.token_ids for p in batch],
                                    max_tokens=[p.max_tokens for p in batch],
                                    samplers=[p.sampler for p in batch],
                                    logits_processors=[p.logits_processors for p in batch],
                                    state_machines=[p.state_machine for p in batch],
                                )
                            except Exception as e:
                                # Wake every waiter with a real error so
                                # Generate exits with INTERNAL instead of
                                # hanging on uid_future forever.
                                logger.exception(
                                    "BatchGenerator.insert failed for batch of %d", len(batch)
                                )
                                with self._pending_lock:
                                    for p in batch:
                                        self._pending_by_request_id.pop(p.request_id, None)
                                for p in batch:
                                    self._loop.call_soon_threadsafe(
                                        _set_future_exception_safe, p.uid_future, e
                                    )
                                # Skip phase 2 — nothing was inserted.
                                continue

                            # insert() succeeded — finalize each request.
                            with self._pending_lock:
                                for p in batch:
                                    self._pending_by_request_id.pop(p.request_id, None)
                            for uid, p in zip(uids, batch):
                                self._uid_queues[uid] = p.queue
                                self._active_uids.add(uid)
                                # Publish the request_id -> uid mapping
                                # BEFORE waking Generate. Otherwise Abort
                                # can land in the gap between set_result
                                # and Generate's own assignment and miss
                                # the mapping entirely (silent no-op).
                                self._request_uid_map[p.request_id] = uid
                                self._loop.call_soon_threadsafe(
                                    _set_future_result_safe, p.uid_future, uid
                                )
                            inserted_this_iter = True

                    # Phase 2: drive the active batch one step. Skip if the
                    # batch is empty *and* we didn't just insert anything —
                    # next() on an empty batch is wasted work.
                    if self._active_uids:
                        with mx.stream(generation_stream):
                            prompt_responses, gen_responses = self.batch_generator.next()

                        for r in gen_responses:
                            queue = self._uid_queues.get(r.uid)
                            if queue is not None:
                                self._loop.call_soon_threadsafe(queue.put_nowait, r)
                            if r.finish_reason is not None:
                                # Discard from _active_uids ONLY after a
                                # successful remove. _active_uids is the
                                # gate that admits new pending requests;
                                # if remove() fails for a real backend
                                # reason (not just an already-removed
                                # uid), discarding would let drain-and-
                                # fill insert into a still-live batch
                                # and reintroduce the rope crash this
                                # whole change is fixing.
                                try:
                                    self.batch_generator.remove([r.uid])
                                    self._active_uids.discard(r.uid)
                                except Exception:
                                    logger.exception(
                                        "BatchGenerator.remove failed for uid %d "
                                        "(keeping it in _active_uids to preserve "
                                        "the drain gate)",
                                        r.uid,
                                    )
            except Exception:
                logger.exception("Error in generation loop")
                continue

            if (
                not prompt_responses
                and not gen_responses
                and not inserted_this_iter
                and not self._active_uids
            ):
                # Truly idle — sleep so the event loop can append to
                # _pending without contending on the gen lock.
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
            uid_future: asyncio.Future = asyncio.get_running_loop().create_future()
            pending = _PendingRequest(
                token_ids=token_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                state_machine=state_machine,
                queue=queue,
                uid_future=uid_future,
                request_id=request_id,
            )
            # Hand off to the gen thread. It'll insert this request into
            # the next fresh batch and resolve uid_future once the uid
            # is assigned. We register on _pending_by_request_id first so
            # an Abort that races with this append can still find us.
            with self._pending_lock:
                self._pending_by_request_id[request_id] = pending
                self._pending.append(pending)
            try:
                uid = await uid_future
            except asyncio.CancelledError:
                # Two cases:
                #  (1) Gen thread hasn't drained us yet — scrub from
                #      pending so it doesn't insert a doomed request.
                #  (2) Gen thread inserted us right before the cancel
                #      landed. We must still run the same backend
                #      cleanup the normal finally would have done,
                #      otherwise the uid keeps decoding and the batch
                #      never drains.
                with self._pending_lock:
                    self._pending_by_request_id.pop(request_id, None)
                    try:
                        self._pending.remove(pending)
                    except ValueError:
                        pass
                # Recover the uid: prefer the future's result, but fall
                # back to _request_uid_map (the gen thread publishes the
                # mapping *before* scheduling set_result, so it's
                # populated even when the cancel arrived first and put
                # the future into the cancelled state).
                inserted_uid = None
                if uid_future.done() and not uid_future.cancelled():
                    try:
                        inserted_uid = uid_future.result()
                    except Exception:
                        inserted_uid = None
                with self._gen_lock:
                    if inserted_uid is None:
                        inserted_uid = self._request_uid_map.pop(request_id, None)
                    else:
                        self._request_uid_map.pop(request_id, None)
                    if inserted_uid is not None:
                        self._uid_queues.pop(inserted_uid, None)
                        try:
                            self.batch_generator.remove([inserted_uid])
                            self._active_uids.discard(inserted_uid)
                        except Exception:
                            # Don't drop _active_uids on a failed remove
                            # — keep the gate honest so drain-and-fill
                            # doesn't insert into a partial batch.
                            logger.exception(
                                "Failed to remove uid %s during cancel cleanup",
                                inserted_uid,
                            )
                raise
            # Note: _request_uid_map[request_id] = uid is published by
            # the gen thread BEFORE waking us, so Abort can find this
            # request immediately. Don't re-set it here (no-op anyway).
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
                # until its own stop/max-tokens condition, wasting batch slots
                # — and, crucially, holds drain-and-batch up so new requests
                # in _pending can't enter until this one finishes naturally.
                # Safe to double-call: the gen thread's finish-path remove
                # and Abort's remove both land here if racing, and remove()
                # raises on unknown uid (which we swallow).
                with self._gen_lock:
                    try:
                        self.batch_generator.remove([uid])
                        self._active_uids.discard(uid)
                    except Exception:
                        # Already removed by the gen thread or Abort —
                        # in those paths, _active_uids was already
                        # discarded after the successful remove. If the
                        # remove failed for a real backend reason on
                        # *every* path, the uid stays in _active_uids
                        # so drain-and-fill won't insert into a still-
                        # live batch.
                        pass

        except ValueError as e:
            logger.warning("Generate invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Generate failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Abort(self, request, context):
        for request_id in request.request_ids:
            # Case A: request hasn't entered the batch yet — pop from
            # pending and cancel its uid_future so Generate exits cleanly.
            with self._pending_lock:
                pending = self._pending_by_request_id.pop(request_id, None)
                if pending is not None:
                    try:
                        self._pending.remove(pending)
                    except ValueError:
                        pass
            if pending is not None:
                if not pending.uid_future.done():
                    pending.uid_future.cancel()
                continue

            # Case B: already inserted — existing in-flight remove path.
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
                        self._active_uids.discard(uid)
                    except Exception:
                        # Same invariant as Generate's finally: only
                        # discard from _active_uids on a successful
                        # remove. If this fails because the gen thread
                        # already removed, _active_uids was already
                        # cleared on that success path.
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
