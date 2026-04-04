"""
SGLang gRPC Server

Standalone gRPC server entrypoint with integrated scheduler.
Handles server lifecycle, TLS, warmup, and shutdown.
"""

import asyncio
import json
import logging
import os
import signal
import threading
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer
from smg_grpc_servicer.sglang.request_manager import GrpcRequestManager
from smg_grpc_servicer.sglang.scheduler_launcher import launch_scheduler_process_only
from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

logger = logging.getLogger(__name__)


def _derive_metrics_port(grpc_port: int) -> int:
    """Derive metrics port from gRPC port using offset + wrap.

    Preferred mapping is `grpc_port + METRICS_PORT_OFFSET`. If that exceeds the
    valid TCP range, wrap to `grpc_port - METRICS_PORT_OFFSET`.
    """

    MAX_TCP_PORT = 65535
    METRICS_PORT_OFFSET = 10000

    if grpc_port < 0 or grpc_port > MAX_TCP_PORT:
        raise ValueError(f"Invalid gRPC port {grpc_port}; expected in [0, {MAX_TCP_PORT}]")

    if grpc_port + METRICS_PORT_OFFSET <= MAX_TCP_PORT:
        return grpc_port + METRICS_PORT_OFFSET

    wrapped_port = grpc_port - METRICS_PORT_OFFSET
    if wrapped_port < 0:
        raise ValueError(
            f"Failed to derive metrics port from gRPC port {grpc_port} with offset "
            f"{METRICS_PORT_OFFSET}"
        )
    return wrapped_port


def _format_host_port(host: str, port: int) -> str:
    """Format host:port and bracket IPv6 literals for gRPC address parsing."""
    if host.startswith("[") and host.endswith("]"):
        return f"{host}:{port}"
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


async def serve_grpc(
    server_args: ServerArgs,
    model_info: dict | None = None,
):
    """Start the standalone gRPC server with integrated scheduler."""

    # Set up Prometheus multiprocess directory before launching schedulers so that
    # child scheduler processes inherit the env var and write metrics to shared files.
    if server_args.enable_metrics:
        from sglang.srt.utils import set_prometheus_multiproc_dir

        set_prometheus_multiproc_dir()

    # Start bootstrap server BEFORE launching scheduler processes (only in PREFILL mode)
    # This ensures the bootstrap server is ready when prefill schedulers try to register
    bootstrap_server = None
    if server_args.disaggregation_mode == "prefill":
        bootstrap_server = start_disagg_service(server_args)
        if bootstrap_server:
            logger.info(
                "Bootstrap server started for disaggregation mode on %s:%s",
                server_args.host,
                server_args.disaggregation_bootstrap_port,
            )

    # Launch only the scheduler process(es) (no tokenizer/detokenizer needed for gRPC)
    logger.info("Launching scheduler process(es)...")
    scheduler_info, port_args, scheduler_procs = launch_scheduler_process_only(
        server_args=server_args,
    )

    # Load model config to get HF config info (same as TokenizerManager does)
    model_config = ModelConfig.from_server_args(server_args)

    # Update model info from scheduler info and model config
    if model_info is None:
        # Extract classification labels from HuggingFace config (if available)
        # Match logic in serving_classify.py::_get_id2label_mapping
        hf_config = model_config.hf_config
        id2label = getattr(hf_config, "id2label", None)
        num_labels = getattr(hf_config, "num_labels", 0) or 0

        # If no id2label but num_labels exists, create default mapping
        if not id2label and num_labels:
            id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        elif id2label and not num_labels:
            num_labels = len(id2label)

        # Convert to JSON string for proto transport
        # id2label is a dict like {0: "negative", 1: "positive"}
        id2label_json = json.dumps(id2label) if id2label else ""

        model_info = {
            "model_name": server_args.model_path,
            "max_context_length": scheduler_info.get(
                "max_total_num_tokens", server_args.context_length or 8192
            ),
            "vocab_size": scheduler_info.get("vocab_size", 128256),
            "supports_vision": scheduler_info.get("supports_vision", False),
            "model_type": getattr(hf_config, "model_type", None),
            "architectures": getattr(hf_config, "architectures", None),
            "max_req_input_len": scheduler_info.get("max_req_input_len", 8192),
            "eos_token_ids": scheduler_info.get("eos_token_ids", []),
            "pad_token_id": scheduler_info.get("pad_token_id", 0),
            "bos_token_id": scheduler_info.get("bos_token_id", 1),
            # Classification model support
            "id2label_json": id2label_json,
            "num_labels": num_labels or 0,
        }

    # Create request manager with the correct port args
    # Note: We pass None for bootstrap_server since it's already started above
    request_manager = GrpcRequestManager(
        server_args=server_args,
        port_args=port_args,
        bootstrap_server=bootstrap_server,
    )

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            # Allow client HTTP/2 keepalive pings every 10s+.
            # Without this, the gRPC C-core default (300s minimum) causes
            # GOAWAY when clients send pings more frequently during long
            # requests (e.g. prefill) where no DATA frames flow.
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    # Create standard health service (for Kubernetes probes)
    health_servicer = SGLangHealthServicer(
        request_manager=request_manager,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Add SGLang service
    servicer = SGLangSchedulerServicer(
        request_manager=request_manager,
        server_args=server_args,
        model_info=model_info,
        scheduler_info=scheduler_info,
        health_servicer=health_servicer,
    )
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)

    # Enable reflection
    SERVICE_NAMES = (
        sglang_scheduler_pb2.DESCRIPTOR.services_by_name["SglangScheduler"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start server
    listen_addr = _format_host_port(server_args.host, server_args.port)
    if server_args.ssl_certfile and server_args.ssl_keyfile:
        if server_args.ssl_keyfile_password:
            raise ValueError(
                "gRPC mode does not support encrypted SSL key files "
                "(--ssl-keyfile-password). Please provide an unencrypted key "
                "file when using --grpc-mode."
            )

        def _read_ssl_file(filepath: str, description: str) -> bytes:
            try:
                with open(filepath, "rb") as f:
                    return f.read()
            except OSError as e:
                raise ValueError(f"Failed to read {description} '{filepath}': {e}") from e

        private_key = _read_ssl_file(server_args.ssl_keyfile, "SSL key file")
        certificate_chain = _read_ssl_file(server_args.ssl_certfile, "SSL certificate file")
        root_certificates = None
        if server_args.ssl_ca_certs:
            root_certificates = _read_ssl_file(server_args.ssl_ca_certs, "SSL CA certificates file")

        if server_args.enable_ssl_refresh:
            # Use dynamic credentials so gRPC re-reads certs on each
            # new connection via the fetcher callback.
            _cert_mtime = os.path.getmtime(server_args.ssl_certfile)
            _key_mtime = os.path.getmtime(server_args.ssl_keyfile)
            _ca_mtime = (
                os.path.getmtime(server_args.ssl_ca_certs) if server_args.ssl_ca_certs else None
            )

            def _cert_config_fetcher():
                nonlocal _cert_mtime, _key_mtime, _ca_mtime
                try:
                    new_cert_mt = os.path.getmtime(server_args.ssl_certfile)
                    new_key_mt = os.path.getmtime(server_args.ssl_keyfile)
                    new_ca_mt = (
                        os.path.getmtime(server_args.ssl_ca_certs)
                        if server_args.ssl_ca_certs
                        else None
                    )

                    if (
                        new_cert_mt == _cert_mtime
                        and new_key_mt == _key_mtime
                        and new_ca_mt == _ca_mtime
                    ):
                        return None  # No change

                    new_key = _read_ssl_file(server_args.ssl_keyfile, "SSL key file")
                    new_cert = _read_ssl_file(server_args.ssl_certfile, "SSL certificate file")
                    new_root = None
                    if server_args.ssl_ca_certs:
                        new_root = _read_ssl_file(
                            server_args.ssl_ca_certs,
                            "SSL CA certificates file",
                        )

                    logger.info("gRPC SSL certificate change detected, reloading.")
                    config = grpc.ssl_server_certificate_configuration(
                        [(new_key, new_cert)],
                        root_certificates=new_root,
                    )

                    # Update mtimes only after successful reload
                    _cert_mtime = new_cert_mt
                    _key_mtime = new_key_mt
                    _ca_mtime = new_ca_mt

                    return config
                except Exception:
                    logger.exception(
                        "Failed to reload gRPC SSL certificates — "
                        "continuing with previous certificates."
                    )
                    return None

            try:
                initial_config = grpc.ssl_server_certificate_configuration(
                    [(private_key, certificate_chain)],
                    root_certificates=root_certificates,
                )
                credentials = grpc.dynamic_ssl_server_credentials(
                    initial_config,
                    _cert_config_fetcher,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create gRPC dynamic SSL credentials. "
                    f"Verify that --ssl-keyfile and --ssl-certfile contain "
                    f"valid, matching PEM data. Underlying error: {e}"
                ) from e
            logger.info("gRPC SSL certificate auto-refresh enabled.")
        else:
            try:
                credentials = grpc.ssl_server_credentials(
                    [(private_key, certificate_chain)],  # pairs: (key, cert)
                    root_certificates=root_certificates,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create gRPC SSL credentials. Verify that "
                    f"--ssl-keyfile and --ssl-certfile contain valid, matching "
                    f"PEM data. Underlying error: {e}"
                ) from e
        bound_port = server.add_secure_port(listen_addr, credentials)
        if bound_port == 0:
            raise RuntimeError(
                f"Failed to bind gRPC TLS server to {listen_addr}. "
                f"Check that the port is available and SSL credentials are valid."
            )
        logger.info(f"gRPC server (TLS) listening on {listen_addr}")
    else:
        server.add_insecure_port(listen_addr)
        logger.info(f"gRPC server listening on {listen_addr}")

    metrics_httpd = None
    metrics_thread = None
    warmup_thread = None
    try:
        await server.start()

        # Start a lightweight HTTP server to serve /metrics for Prometheus scraping.
        # The gRPC port can't serve HTTP, so this sidecar runs on a dedicated HTTP port.
        # Use a deterministic large offset with wrap to avoid adjacent-port collisions.
        if server_args.enable_metrics:
            metrics_port = _derive_metrics_port(server_args.port)
            metrics_httpd, metrics_thread = _start_metrics_http_server(
                server_args.host, metrics_port
            )

        # Start warmup in a separate thread
        warmup_thread = threading.Thread(
            target=_wait_and_warmup_grpc,
            args=(server_args, health_servicer),
        )
        warmup_thread.start()

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler():
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        await stop_event.wait()
    finally:
        logger.info("Shutting down gRPC server")

        # Shutdown request manager first - this closes ZMQ sockets and stops background tasks
        await servicer.shutdown()

        # Stop the gRPC server
        await server.stop(5.0)

        # Gracefully shut down the metrics HTTP server so metrics are flushed
        if metrics_httpd is not None:
            logger.info("Shutting down HTTP metrics server")
            metrics_httpd.shutdown()
            metrics_httpd.server_close()
            if metrics_thread is not None and metrics_thread.is_alive():
                metrics_thread.join(timeout=5.0)

        # Wait for warmup thread to finish
        if warmup_thread is not None and warmup_thread.is_alive():
            logger.info("Waiting for warmup thread to finish...")
            warmup_thread.join(timeout=5.0)

        # Terminate scheduler processes before exiting to avoid atexit hang
        # The scheduler processes have SIGINT ignored, so they won't get KeyboardInterrupt
        for i, proc in enumerate(scheduler_procs):
            if proc.is_alive():
                logger.info(f"Terminating scheduler process {i}...")
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():
                    logger.warning(f"Scheduler process {i} did not terminate, killing...")
                    proc.kill()
                    proc.join(timeout=1.0)

        logger.info("All scheduler processes terminated")


def _start_metrics_http_server(host: str, port: int):
    """Start a background HTTP server that serves Prometheus metrics from PROMETHEUS_MULTIPROC_DIR.

    Uses the same multiprocess collector pattern as TGL's HTTP mode
    (see sglang.srt.utils.common.add_prometheus_middleware).

    Returns the HTTPServer and worker thread so the caller can stop and join cleanly.
    """
    import ipaddress
    import socket
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics" or self.path.startswith("/metrics?"):
                output = generate_latest(registry)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.end_headers()
                self.wfile.write(output)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    class IPv6HTTPServer(HTTPServer):
        address_family = socket.AF_INET6

    bind_host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    prefer_ipv6 = False
    try:
        prefer_ipv6 = ipaddress.ip_address(bind_host).version == 6
    except ValueError:
        pass

    if prefer_ipv6:
        httpd = IPv6HTTPServer((bind_host, port), MetricsHandler)
    else:
        try:
            httpd = HTTPServer((bind_host, port), MetricsHandler)
        except socket.gaierror:
            # Hostname may resolve only to IPv6 on this system.
            httpd = IPv6HTTPServer((bind_host, port), MetricsHandler)

    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    logger.info("Metrics HTTP server listening on %s:%s", bound_host, bound_port)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="metrics-http-server")
    thread.start()

    return httpd, thread


def _execute_grpc_server_warmup(server_args: ServerArgs):
    """Execute warmup for gRPC server by checking health and sending test request."""
    try:
        # Connect to the gRPC server
        grpc_url = _format_host_port(server_args.host, server_args.port)
        channel = grpc.insecure_channel(
            grpc_url,
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 256),
                ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ],
        )
        stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)

        # Wait until the server is launched (poll GetModelInfo)
        success = False
        last_error = None
        for _ in range(120):
            time.sleep(1)
            try:
                request = sglang_scheduler_pb2.GetModelInfoRequest()
                response = stub.GetModelInfo(request, timeout=5)
                success = True
                break
            except Exception as e:
                last_error = str(e)
                pass

        if not success:
            error_msg = (
                "gRPC server warmup failed: Could not connect to server"
                f" after 120 seconds. Last error: {last_error}"
            )
            logger.error(error_msg)
            channel.close()
            kill_process_tree(os.getpid())
            return False

        # Get model info to determine if it's generation or embedding
        is_generation = response.is_generation

        # Send a warmup request
        logger.info("Sending warmup request to gRPC server...")
        max_new_tokens = 8 if is_generation else 1

        if is_generation:
            warmup_request_kwargs = {
                "request_id": f"WARMUP_{time.time()}",
                "tokenized": sglang_scheduler_pb2.TokenizedInput(
                    input_ids=[
                        123,
                        456,
                        789,
                        234,
                        567,
                        890,
                        345,
                    ],  # Random-looking but safe token IDs
                    original_text="warmup request",
                ),
                "sampling_params": sglang_scheduler_pb2.SamplingParams(
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                ),
                "stream": False,
            }

            # Set disaggregation params if needed
            if server_args.disaggregation_mode != DisaggregationMode.NULL.value:
                warmup_request_kwargs["disaggregated_params"] = (
                    sglang_scheduler_pb2.DisaggregatedParams(
                        bootstrap_host=FAKE_BOOTSTRAP_HOST,
                        bootstrap_room=0,
                    )
                )

            warmup_request = sglang_scheduler_pb2.GenerateRequest(**warmup_request_kwargs)

            # Send the warmup request
            try:
                responses = list(stub.Generate(warmup_request, timeout=600))
                # Check if we got a valid complete response (errors use gRPC status, not in-band)
                if responses and responses[-1].HasField("complete"):
                    logger.info("gRPC warmup request completed successfully")
                    success = True
                else:
                    logger.warning("gRPC warmup request returned no complete response")
                    success = False
            except Exception as e:
                error_msg = f"gRPC warmup request failed: {e}"
                logger.error(error_msg)
                channel.close()
                kill_process_tree(os.getpid())
                return False
        else:
            # For embedding models
            warmup_request = sglang_scheduler_pb2.EmbedRequest(
                request_id=f"WARMUP_{time.time()}",
                tokenized=sglang_scheduler_pb2.TokenizedInput(
                    input_ids=[10, 11, 12],
                    original_text="test embedding",
                ),
            )

            try:
                response = stub.Embed(warmup_request, timeout=600)
                # EmbedResponse is flat; errors use gRPC status, not in-band
                if response.embedding_dim > 0:
                    logger.info("gRPC warmup request completed successfully")
                    success = True
                else:
                    logger.warning("gRPC warmup: embed response has zero embedding_dim")
                    success = False
            except Exception as e:
                error_msg = f"gRPC warmup request failed: {e}"
                logger.error(error_msg)
                channel.close()
                kill_process_tree(os.getpid())
                return False

        channel.close()
        return success

    except Exception as e:
        error_msg = f"gRPC warmup failed with exception: {e}\n{get_exception_traceback()}"
        logger.error(error_msg)
        try:
            channel.close()
        except Exception:
            pass
        kill_process_tree(os.getpid())
        return False


def _wait_and_warmup_grpc(
    server_args: ServerArgs,
    health_servicer: SGLangHealthServicer | None = None,
):
    """Wait for gRPC server to be ready and execute warmup."""
    if not server_args.skip_server_warmup:
        if not _execute_grpc_server_warmup(server_args):
            return
    else:
        logger.info("Skipping gRPC server warmup (skip_server_warmup=True)")

    # Mark health service as SERVING after warmup completes
    if health_servicer:
        health_servicer.set_serving()

    logger.info("The server is fired up and ready to roll!")
