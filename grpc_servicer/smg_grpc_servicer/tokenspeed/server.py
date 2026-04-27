"""Standalone TokenSpeed gRPC server.

Mirrors ``smg_grpc_servicer.sglang.server.serve_grpc``:

1. Launch TokenSpeed's scheduler subprocess(es) + AsyncLLM via
   :func:`smg_grpc_servicer.tokenspeed.scheduler_launcher.launch_engine`.
2. Start a ``grpc.aio`` server that advertises the
   ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` service so the SMG
   router's ``DetectBackendStep`` identifies the worker natively.
3. Warm the worker up with a tiny generation request, flip the health
   servicer to SERVING, and await SIGTERM/SIGINT for graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from tokenspeed.runtime.utils.server_args import ServerArgs

from smg_grpc_servicer.tokenspeed.health_servicer import TokenSpeedHealthServicer
from smg_grpc_servicer.tokenspeed.scheduler_launcher import launch_engine
from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

logger = logging.getLogger(__name__)


async def serve_grpc(server_args: ServerArgs) -> None:
    """Run the TokenSpeed gRPC server until a shutdown signal is received."""

    logger.info("Launching TokenSpeed scheduler + AsyncLLM...")
    async_llm, scheduler_info = launch_engine(server_args)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            # Match SGLang's more-permissive keepalive defaults so long
            # prefill stalls don't trip GOAWAY in the Rust client.
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    health_servicer = TokenSpeedHealthServicer(
        async_llm=async_llm,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    servicer = TokenSpeedSchedulerServicer(
        async_llm=async_llm,
        server_args=server_args,
        scheduler_info=scheduler_info,
        health_servicer=health_servicer,
    )
    tokenspeed_scheduler_pb2_grpc.add_TokenSpeedSchedulerServicer_to_server(servicer, server)

    service_names = (
        tokenspeed_scheduler_pb2.DESCRIPTOR.services_by_name["TokenSpeedScheduler"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)
    logger.info("TokenSpeed gRPC server listening on %s", listen_addr)

    await server.start()

    # Warmup on a background thread so the async server can handle the probe.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(server_args, health_servicer),
        daemon=True,
    )
    warmup_thread.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows and some exotic envs don't support loop.add_signal_handler.
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down TokenSpeed gRPC server")
        try:
            await servicer.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("servicer.shutdown() raised")
        await server.stop(5.0)
        if warmup_thread.is_alive():
            warmup_thread.join(timeout=5.0)


def _wait_and_warmup(
    server_args: ServerArgs,
    health_servicer: TokenSpeedHealthServicer,
) -> None:
    """Probe the gRPC server until it can generate one token, then set SERVING.

    We hit the external port (not the in-process servicer) so the warmup
    exercises the same code path a production caller would — including the
    gRPC transport, proto codec, and scheduler IPC.
    """
    if os.getenv("TOKENSPEED_SKIP_GRPC_WARMUP", "0").lower() in ("1", "true", "yes"):
        logger.info("TOKENSPEED_SKIP_GRPC_WARMUP=1 — skipping warmup")
        health_servicer.set_serving()
        return

    grpc_url = f"{server_args.host}:{server_args.port}"
    channel = grpc.insecure_channel(
        grpc_url,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel)

    # Wait until GetModelInfo round-trips — that's the quickest confirmation
    # that the gRPC server is both bound and has a live AsyncLLM behind it.
    deadline = time.time() + 180
    connected = False
    while time.time() < deadline:
        try:
            stub.GetModelInfo(
                tokenspeed_scheduler_pb2.GetModelInfoRequest(),
                timeout=5,
            )
            connected = True
            break
        except Exception as e:  # noqa: BLE001
            logger.debug("Warmup: GetModelInfo not ready yet: %s", e)
            time.sleep(1)

    if not connected:
        logger.error("TokenSpeed gRPC warmup failed: GetModelInfo never succeeded")
        channel.close()
        return

    # TokenSpeed serves generative LLMs only (the proto has no Embed RPC), so
    # the warmup is always a 1-token generate.
    try:
        warmup = tokenspeed_scheduler_pb2.GenerateRequest(
            request_id=f"WARMUP_{time.time()}",
            tokenized=tokenspeed_scheduler_pb2.TokenizedInput(
                input_ids=[0],
                original_text="warmup",
            ),
            sampling_params=tokenspeed_scheduler_pb2.SamplingParams(
                temperature=0.0,
                max_new_tokens=1,
            ),
            stream=False,
        )
        final = None
        for resp in stub.Generate(warmup, timeout=600):
            final = resp
        if final is None or not final.HasField("complete"):
            logger.warning(
                "Warmup Generate returned no Complete frame (last=%r)",
                final,
            )
        else:
            logger.info("Warmup generation succeeded")
    except Exception as e:  # noqa: BLE001
        logger.warning("TokenSpeed warmup failed: %s", e)
    finally:
        channel.close()

    health_servicer.set_serving()
    logger.info("TokenSpeed gRPC server is ready to serve")
