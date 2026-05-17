"""Scheduler subprocess launcher for the TokenSpeed gRPC server.

Delegates to TokenSpeed's ``_launch_subprocesses`` and returns the
``AsyncLLM`` + scheduler-info dict the gRPC server cares about.
"""

from __future__ import annotations

import logging
from typing import Any

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.entrypoints.engine import _launch_subprocesses
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def launch_engine(
    server_args: ServerArgs,
    port_args: PortArgs | None = None,
) -> tuple[AsyncLLM, dict[str, Any]]:
    """Launch the scheduler subprocess(es) and return the live ``AsyncLLM``.

    Raises ``RuntimeError`` on non-rank-0 nodes (which return ``None`` and
    block forever on the dummy health server — they never serve gRPC).
    """
    async_llm, _template_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args,
        port_args=port_args,
    )

    if async_llm is None:
        raise RuntimeError(
            "launch_engine() returned no AsyncLLM — only rank 0 may serve gRPC traffic."
        )

    logger.info(
        "TokenSpeed engine ready: max_total_num_tokens=%s max_req_input_len=%s",
        scheduler_info.get("max_total_num_tokens"),
        scheduler_info.get("max_req_input_len"),
    )
    return async_llm, scheduler_info
