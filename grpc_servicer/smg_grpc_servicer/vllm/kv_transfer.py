"""KV-transfer param passthrough between the vLLM engine proto and connector dicts."""

import json
import logging

from smg_grpc_proto import vllm_engine_pb2

logger = logging.getLogger(__name__)


def params_from_request(
    request: vllm_engine_pb2.GenerateRequest,
) -> dict | None:
    """Extract KV-transfer params; JSON field preferred, legacy typed field as fallback.

    Raises:
        ValueError: If the JSON field is malformed or the legacy field is invalid.
    """
    if request.HasField("kv_transfer_params_json"):
        try:
            params = json.loads(request.kv_transfer_params_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid kv_transfer_params_json: {e}") from e
        if not isinstance(params, dict):
            raise ValueError("kv_transfer_params_json must be a JSON object")
        return params
    if request.HasField("kv_transfer_params"):
        remote_host = request.kv_transfer_params.remote_host
        remote_port = request.kv_transfer_params.remote_port
        if not remote_host or not (1 <= remote_port <= 65535):
            raise ValueError(
                "Invalid kv_transfer_params: remote_host must be set and remote_port must be in [1, 65535]."
            )
        return {"remote_host": remote_host, "remote_port": remote_port}
    return None


def params_to_response_fields(
    params: dict | None,
) -> tuple[vllm_engine_pb2.KvTransferParams | None, str | None]:
    """Map engine-returned params to (legacy typed message, JSON field) for GenerateComplete."""
    if not params:
        return None, None

    params_json = None
    try:
        params_json = json.dumps(params)
    except (TypeError, ValueError):
        logger.warning("Dropping non-JSON-serializable kv_transfer_params: %r", params)

    # Legacy mirror for old routers; built only when host/port are valid (Mooncake shape)
    legacy = None
    remote_host = params.get("remote_host", "")
    remote_port = params.get("remote_port", 0)
    if (
        isinstance(remote_host, str)
        and remote_host
        and isinstance(remote_port, int)
        and 1 <= remote_port <= 65535
    ):
        legacy = vllm_engine_pb2.KvTransferParams(
            remote_host=remote_host,
            remote_port=remote_port,
        )
    return legacy, params_json
