"""vLLM-specific KV-events config resolution.

The wire-format conversion + ZMQ streaming helpers live in the engine-neutral
``smg_grpc_servicer.kv_events`` module and are re-exported here for backwards
compatibility (existing imports and tests reference them via this module).
"""

import logging

from smg_grpc_servicer.kv_events import (
    convert_batch,
    convert_event,
    endpoint_for_rank,
    stream_kv_events,
    to_int64,
)

__all__ = [
    "convert_batch",
    "convert_event",
    "endpoint_for_rank",
    "resolve_kv_events_config",
    "stream_kv_events",
    "to_int64",
]

logger = logging.getLogger(__name__)


def resolve_kv_events_config(engine: object):
    """Return the vLLM KVEventsConfig iff ZMQ event publishing is enabled, else None.

    Reads ``engine.vllm_config.kv_events_config`` via getattr so it tolerates any
    vLLM version and is testable with a fake engine.
    """
    vllm_config = getattr(engine, "vllm_config", None)
    cfg = getattr(vllm_config, "kv_events_config", None)
    if cfg is None:
        return None
    if not getattr(cfg, "enable_kv_cache_events", False):
        logger.info("vLLM KV cache events not enabled; SubscribeKvEvents disabled")
        return None
    if getattr(cfg, "publisher", None) != "zmq":
        logger.info(
            "vLLM KV events publisher is %r, not 'zmq'; SubscribeKvEvents disabled",
            getattr(cfg, "publisher", None),
        )
        return None
    logger.info("vLLM KV events enabled: endpoint=%s", getattr(cfg, "endpoint", "?"))
    return cfg
