"""BFCL-to-OpenAI format converter.

Converts BFCL function definitions to OpenAI-compatible tool calling format,
handling BFCL-specific schema quirks (non-standard JSON Schema types).
"""

from __future__ import annotations


def _fix_parameter_type(params: dict) -> dict:
    """Convert BFCL's non-standard types to valid JSON Schema types recursively."""
    result = dict(params)
    ptype = result.get("type")
    if ptype == "dict":
        result["type"] = "object"
    elif ptype == "float":
        result["type"] = "number"
    props = result.get("properties")
    if isinstance(props, dict):
        result["properties"] = {
            k: _fix_parameter_type(v) if isinstance(v, dict) else v for k, v in props.items()
        }
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = _fix_parameter_type(items)
    return result


def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tools format.

    Handles the BFCL-specific quirks:
      - parameters.type "dict" → "object"
      - parameters.type "float" → "number"
      - Wraps in {"type": "function", "function": ...}
    """
    tools = []
    for fn in bfcl_functions:
        fixed_fn = dict(fn)
        if "parameters" in fixed_fn:
            fixed_fn["parameters"] = _fix_parameter_type(fixed_fn["parameters"])
        tools.append({"type": "function", "function": fixed_fn})
    return tools
