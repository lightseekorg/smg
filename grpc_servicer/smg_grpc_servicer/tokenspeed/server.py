"""Standalone TokenSpeed gRPC server."""

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
            # VLM pixel_values can be large; raise to gRPC's max (2 GiB - 1).
            ("grpc.max_send_message_length", (2 << 30) - 1),
            ("grpc.max_receive_message_length", (2 << 30) - 1),
            # Permissive keepalive so long prefill stalls don't trip GOAWAY.
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

    Hits the external port so the warmup exercises transport, proto codec,
    and scheduler IPC end-to-end.
    """
    if os.getenv("TOKENSPEED_SKIP_GRPC_WARMUP", "0").lower() in ("1", "true", "yes"):
        logger.info("TOKENSPEED_SKIP_GRPC_WARMUP=1 — skipping warmup")
        health_servicer.set_serving()
        return

    # Wildcard bind hosts aren't routable as destinations; dial loopback instead.
    warmup_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(server_args.host, server_args.host)
    grpc_url = f"{warmup_host}:{server_args.port}"
    channel = grpc.insecure_channel(
        grpc_url,
        options=[
            ("grpc.max_send_message_length", (2 << 30) - 1),
            ("grpc.max_receive_message_length", (2 << 30) - 1),
        ],
    )
    stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel)

    # GetModelInfo is the quickest confirmation the server is bound + the
    # engine is alive.
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

    # Generative only — warmup is a 1-token generate.
    warmup_ok = False
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
            warmup_ok = True
    except Exception as e:  # noqa: BLE001
        logger.warning("TokenSpeed warmup failed: %s", e)
    finally:
        channel.close()

    if warmup_ok:
        health_servicer.set_serving()
        logger.info("TokenSpeed gRPC server is ready to serve")
    else:
        # Stays NOT_SERVING so K8s readiness keeps this worker out of rotation.
        logger.error("TokenSpeed gRPC warmup did not produce a complete frame")
