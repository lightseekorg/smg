"""Scheduler subprocess launcher for the TokenSpeed gRPC server.

Mirrors ``smg_grpc_servicer.sglang.scheduler_launcher`` but delegates to
TokenSpeed's ``_launch_subprocesses``: we get back a fully-initialised
``AsyncLLM`` along with the scheduler info dict. All scheduler/DP-controller
spawning, multiprocessing start-method, and env priming already live inside
``_launch_subprocesses`` — we only wrap it to return what the gRPC server
cares about and to keep the call site symmetric with the sibling backends.
"""

from __future__ import annotations

import logging
from typing import Any

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.entrypoints.engine import _launch_subprocesses
from tokenspeed.runtime.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def launch_engine(
    server_args: ServerArgs,
    port_args: PortArgs | None = None,
) -> tuple[AsyncLLM, dict[str, Any]]:
    """Launch TokenSpeed scheduler subprocess(es) and the main-process AsyncLLM.

    Returns:
        A tuple ``(async_llm, scheduler_info)``. ``async_llm`` is the live
        :class:`AsyncLLM` that the gRPC servicer will drive. ``scheduler_info``
        is the dict rank-0 sent back once its scheduler was ready (contains
        e.g. ``max_total_num_tokens``, ``max_req_input_len``, ...).

    Raises:
        RuntimeError: If rank-0 scheduler fails to initialize. The original
        ``_launch_subprocesses`` surfaces this by re-raising the EOF/assertion
        error — we propagate it unchanged.
    """
    async_llm, _template_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args,
        port_args=port_args,
    )

    # Non-zero rank nodes return (None, None, None) from _launch_subprocesses
    # and block forever on the dummy health server — they never reach the gRPC
    # server. Guard against callers relying on this return on secondary nodes.
    if async_llm is None:
        raise RuntimeError(
            "launch_engine() returned no AsyncLLM. This means the current node "
            "is not rank 0 in a multi-node deployment, or the scheduler died "
            "during initialization. Only rank 0 may serve gRPC traffic."
        )

    logger.info(
        "TokenSpeed engine ready: max_total_num_tokens=%s max_req_input_len=%s",
        scheduler_info.get("max_total_num_tokens"),
        scheduler_info.get("max_req_input_len"),
    )
    return async_llm, scheduler_info
