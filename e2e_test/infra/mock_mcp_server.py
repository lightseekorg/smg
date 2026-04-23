"""In-process mock MCP server for deterministic e2e tests.

Overview
--------
This module provides a reusable, in-process mock ``MCP`` server built on top of
the official ``mcp`` SDK (``FastMCP``). It exposes ``streamable``-HTTP transport
on a local port so the gateway's MCP client (which already speaks streamable
HTTP against Brave in production) can connect without any code changes to the
Rust side.

Why a mock?
-----------
The existing MCP e2e test (``e2e_test/messages/test_mcp_tool.py``) depends on a
real, external Brave MCP server. That is:

* **Flaky in CI** — network partitions or upstream outages blow up tests that
  should be validating our gateway's plumbing, not Brave's availability.
* **Slow** — real searches take seconds; we want sub-100 ms tool calls.
* **Non-deterministic** — responses change over time, so byte-for-byte
  assertions are impossible.

The mock fixes all three. Responses are fixed constants, the server runs in the
same Python process as the test, and there is no external dependency to
coordinate in CI.

Registering a new tool
----------------------
To add a new tool (e.g., ``web_search``) for a future R6.x test, do two things:

1. Write the tool function as a top-level ``@FastMCP.tool`` decorator inside
   ``_register_tools``. Return a plain ``dict`` or ``str`` — the SDK will wrap
   it into the MCP tool-result envelope.
2. (Optional) If you want ``last_call_args`` introspection, append to
   ``self._call_log`` inside the tool body. The existing ``image_generation``
   tool shows the pattern.

The gateway wires the tool to an OpenAI built-in type via the MCP config
``builtin_type`` (``image_generation``, ``web_search_preview``, ``file_search``,
``code_interpreter``). See ``e2e_test/responses/conftest.py`` for the config
fixture that ties a mock tool to a builtin type.

Lifetime
--------
``MockMcpServer`` can be used as a context manager or managed manually via
``start()`` / ``stop()``. Starting allocates a free port (``port=0``) unless an
explicit port was passed in. Stopping asks the underlying ``uvicorn.Server`` to
exit and joins the background thread.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# =============================================================================
# Deterministic payloads
# =============================================================================

# 1x1 transparent PNG, base64-encoded. Hard-coded so tests can make byte-for-byte
# assertions without depending on a compression library's exact output.
#
# Generated with: ``python -c "import base64; print(base64.b64encode(bytes.fromhex(
#   '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
#   '0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082')).decode())"``
IMAGE_GENERATION_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQott"
    "AAAAABJRU5ErkJggg=="
)


# =============================================================================
# Mock server
# =============================================================================


class MockMcpServer:
    """In-process MCP server exposing a streamable-HTTP endpoint.

    Parameters
    ----------
    host:
        Bind address. Defaults to ``127.0.0.1``.
    port:
        TCP port. ``0`` (the default) asks the OS for a free port that is
        captured at ``start()`` time via a short-lived socket probe.
    log_level:
        ``uvicorn`` log level. Defaults to ``warning`` to keep test output
        clean. Pass ``info`` or ``debug`` when troubleshooting.
    ready_timeout:
        Seconds to wait for the server's ``started`` flag to flip before
        giving up.

    Attributes
    ----------
    url:
        Reachable URL pointing at the streamable HTTP endpoint
        (``http://host:port/mcp``). Populated after ``start()``.
    image_generation_png_base64:
        The deterministic base64 PNG the ``image_generation`` tool returns.
        Exposed as an instance attribute so tests can reference it without
        importing the module-level constant.
    last_call_args:
        Read-only snapshot of the last tool invocation's arguments. Used by
        the override-assertion test to verify the gateway pinned arguments
        correctly before dispatching.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        log_level: str = "warning",
        ready_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self._configured_port = port
        self.port: int | None = None if port == 0 else port
        self._log_level = log_level
        self._ready_timeout = ready_timeout

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._call_log: list[dict[str, Any]] = []

        # The server's streamable HTTP mount path. FastMCP fixes this at /mcp
        # and the gateway's MCP client is configured with the same path.
        self._mount_path = "/mcp"

        # Public conveniences — let tests read the deterministic payload without
        # reaching into module-level constants.
        self.image_generation_png_base64 = IMAGE_GENERATION_PNG_BASE64

    # ------------------------------------------------------------------
    # Public URL
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Return the MCP streamable-HTTP URL. Requires ``start()`` first."""
        if self.port is None:
            raise RuntimeError("MockMcpServer not started — call start() first")
        return f"http://{self.host}:{self.port}{self._mount_path}"

    @property
    def last_call_args(self) -> dict[str, Any] | None:
        """Arguments the last tool invocation received, or ``None`` if no calls yet."""
        if not self._call_log:
            return None
        return dict(self._call_log[-1])

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """All observed calls, in order, as a copy. Handy for multi-tool assertions."""
        return [dict(entry) for entry in self._call_log]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Start the server in a background thread and return its URL."""
        if self._server is not None:
            raise RuntimeError("MockMcpServer already started")

        if self.port is None:
            self.port = _pick_free_port(self.host)

        fastmcp = self._build_fastmcp()
        app = fastmcp.streamable_http_app()

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=self._log_level,
            lifespan="on",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._run_server,
            name=f"MockMcpServer:{self.port}",
            daemon=True,
        )
        self._thread.start()
        self._wait_ready()
        logger.info("MockMcpServer ready at %s", self.url)
        return self.url

    def stop(self) -> None:
        """Ask the server to exit and join the background thread."""
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("MockMcpServer thread did not exit cleanly")
        self._server = None
        self._thread = None

    def __enter__(self) -> MockMcpServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_server(self) -> None:
        # uvicorn.Server.serve() is an async coroutine. Thread-local event loop.
        assert self._server is not None
        asyncio.run(self._server.serve())

    def _wait_ready(self) -> None:
        assert self._server is not None
        deadline = time.perf_counter() + self._ready_timeout
        while time.perf_counter() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"MockMcpServer did not become ready within {self._ready_timeout}s"
        )

    def _build_fastmcp(self) -> FastMCP:
        """Construct the FastMCP app and register tools.

        Kept as a method (rather than module-level) so that each ``start()``
        call produces an isolated FastMCP instance — important for session
        isolation across tests that run back-to-back.
        """
        fastmcp = FastMCP(
            name="smg-e2e-mock",
            host=self.host,
            stateless_http=True,
        )
        self._register_tools(fastmcp)
        return fastmcp

    def _register_tools(self, fastmcp: FastMCP) -> None:
        """Attach all supported tools. Add new tools here.

        Each tool appends to ``self._call_log`` so tests can assert on what
        arguments reached the server. The FastMCP tool-return convention is to
        return a plain ``dict`` (or ``str``); FastMCP wraps it into the MCP
        tool-result envelope automatically.
        """
        call_log = self._call_log
        deterministic_image = self.image_generation_png_base64

        @fastmcp.tool(
            name="image_generation",
            description=(
                "Mock image_generation tool. Returns a deterministic 1x1 "
                "transparent PNG as base64. Never calls out of process."
            ),
        )
        def image_generation(
            prompt: str,
            size: str = "1024x1024",
            quality: str = "standard",
            moderation: str = "auto",
            output_format: str = "png",
        ) -> dict[str, str]:
            call_log.append(
                {
                    "tool": "image_generation",
                    "arguments": {
                        "prompt": prompt,
                        "size": size,
                        "quality": quality,
                        "moderation": moderation,
                        "output_format": output_format,
                    },
                }
            )
            return {
                "result": deterministic_image,
                "revised_prompt": prompt,
                "status": "completed",
            }

        # -----------------------------------------------------------------
        # TODO(R6.x follow-ups): uncomment these stubs as each built-in tool
        # gets its own e2e coverage. The signatures match the OpenAI
        # built-in tool contracts documented in the audit (§R6).
        # -----------------------------------------------------------------
        #
        # @fastmcp.tool(name="web_search", description="Mock web_search tool.")
        # def web_search(query: str, count: int = 3) -> dict[str, list[dict]]:
        #     call_log.append({"tool": "web_search", "arguments": {"query": query, "count": count}})
        #     return {
        #         "results": [
        #             {
        #                 "title": f"Mock result for {query}",
        #                 "url": "https://example.com/mock",
        #                 "snippet": "Deterministic mock snippet.",
        #             }
        #         ]
        #     }
        #
        # @fastmcp.tool(name="file_search", description="Mock file_search tool.")
        # def file_search(query: str, max_results: int = 3) -> dict[str, list[dict]]:
        #     call_log.append({"tool": "file_search", "arguments": {"query": query,
        #                                                             "max_results": max_results}})
        #     return {"results": [{"file_id": "file_mock_1", "score": 1.0,
        #                          "content": [{"type": "text", "text": "mock"}]}]}
        #
        # @fastmcp.tool(name="code_interpreter", description="Mock code_interpreter tool.")
        # def code_interpreter(code: str) -> dict[str, str]:
        #     call_log.append({"tool": "code_interpreter", "arguments": {"code": code}})
        #     return {"stdout": "mock stdout", "stderr": "", "status": "completed"}


# =============================================================================
# Port utility
# =============================================================================


def _pick_free_port(host: str) -> int:
    """Ask the OS for a free TCP port on ``host`` and return it.

    There is an unavoidable race between closing the probe socket and binding
    the uvicorn listener, but it is the same pattern used elsewhere in
    ``e2e_test/infra/process_utils.py`` (``get_open_port``) and is good enough
    for local test orchestration.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
