"""
MLX gRPC Server

Standalone gRPC server entrypoint for MLX inference.
CLI: python -m smg_grpc_servicer.mlx.server --model <path> --port 50051
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from smg_grpc_proto import mlx_engine_pb2, mlx_engine_pb2_grpc

from smg_grpc_servicer.mlx.health_servicer import MlxHealthServicer
from smg_grpc_servicer.mlx.servicer import MlxEngineServicer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MLX gRPC inference server")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listen port")
    parser.add_argument("--host", default="0.0.0.0", help="gRPC listen address")
    parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Max concurrent prefill requests"
    )
    parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Max concurrent generation requests"
    )
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer via mlx-lm."""
    logger.info("Loading model: %s", args.model)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    logger.info("Model loaded successfully")

    model_dir = args.model
    if not os.path.isdir(model_dir):
        model_dir = snapshot_download(
            args.model,
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "special_tokens*",
                "merges.txt",
                "vocab.json",
                "added_tokens.json",
                # Chat template sidecars (Gemma 4, Llama 3.1+, newer models).
                "chat_template.json",
                "chat_template.jinja",
                # tiktoken-style tokenizer artifacts — must stay in sync
                # with MlxEngineServicer._TOKENIZER_FILES / _SUFFIXES.
                "tiktoken.model",
                "*.tiktoken",
            ],
        )

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)

    eos = model_config.get("eos_token_id")
    if isinstance(eos, int):
        eos_token_ids = [eos]
    elif isinstance(eos, list):
        eos_token_ids = eos
    else:
        eos_token_ids = list(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else []

    return model, tokenizer, model_dir, model_config, eos_token_ids


def _warmup(batch_generator):
    """Run one end-to-end token through the batch generator so the first
    real request doesn't pay JIT/kernel compilation cost."""
    logger.info("Running warmup generation...")
    try:
        uids = batch_generator.insert(prompts=[[1]], max_tokens=[1])
        for _ in range(10):
            _, gen_responses = batch_generator.next()
            if any(r.finish_reason is not None for r in gen_responses if r.uid == uids[0]):
                break
        batch_generator.remove(uids)
        logger.info("Warmup complete")
    except Exception:
        logger.warning("Warmup failed (non-fatal)", exc_info=True)


async def serve_grpc(args):
    """Start the MLX gRPC server."""
    start_time = time.time()

    model, tokenizer, model_dir, model_config, eos_token_ids = load_model(args)

    batch_generator = BatchGenerator(
        model,
        completion_batch_size=args.completion_batch_size,
        prefill_batch_size=args.prefill_batch_size,
    )
    logger.info(
        "BatchGenerator created (prefill=%d, completion=%d)",
        args.prefill_batch_size,
        args.completion_batch_size,
    )

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    health_servicer = MlxHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    servicer = MlxEngineServicer(
        batch_generator=batch_generator,
        model_path=args.model,
        model_dir=model_dir,
        model_config=model_config,
        eos_token_ids=eos_token_ids,
        start_time=start_time,
    )
    mlx_engine_pb2_grpc.add_MlxEngineServicer_to_server(servicer, server)

    SERVICE_NAMES = (
        mlx_engine_pb2.DESCRIPTOR.services_by_name["MlxEngine"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    listen_addr = f"{args.host}:{args.port}"
    bound_port = server.add_insecure_port(listen_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to {listen_addr}")

    # Warmup BEFORE starting the generation loop (batch_generator.next() is
    # not thread-safe — only one caller at a time).
    _warmup(batch_generator)
    servicer.start_generation_loop()

    # Only accept RPCs after the generation loop is running. Otherwise a
    # Generate RPC could slip into the window between server.start() and
    # start_generation_loop() and block forever on queue.get() because no
    # gen thread is dispatching tokens. HealthCheck always returns OK, so
    # the router can't use it to detect this window.
    await server.start()
    health_servicer.set_serving()
    logger.info("gRPC server listening on %s — model: %s", listen_addr, args.model)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down...")
        health_servicer.set_not_serving()
        # Stop accepting new RPCs first so in-flight requests can still
        # drain against the running generation thread. Stopping the gen
        # loop first would leave new/in-flight RPCs stranded.
        await server.stop(5.0)
        servicer.stop_generation_loop()
        batch_generator.close()
        logger.info("Server stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(serve_grpc(args))


if __name__ == "__main__":
    main()
