"""Source-level contract tests for current SGLang request construction.

These tests intentionally avoid importing SGLang so the compatibility contract
can be checked in the lightweight gRPC-servicer test environment.
"""

import ast
import unittest
from pathlib import Path

SERVICER_PATH = Path(__file__).resolve().parents[1] / "smg_grpc_servicer" / "sglang" / "servicer.py"
CURRENT_REQUIRED_FIELDS = {
    "rid",
    "input_text",
    "input_ids",
    "input_embeds",
    "mm_inputs",
    "token_type_ids",
    "sampling_params",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "stream",
}


def _constructor_keywords(function_name: str) -> list[set[str]]:
    module = ast.parse(SERVICER_PATH.read_text())
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name
    )
    return [
        {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "TokenizedGenerateReqInput"
    ]


class TestSGLangGenerateRequestContract(unittest.TestCase):
    def test_generate_converter_supplies_current_required_fields(self):
        calls = _constructor_keywords("_convert_generate_request")
        self.assertEqual(len(calls), 1)
        self.assertEqual(CURRENT_REQUIRED_FIELDS - calls[0], set())

    def test_health_probe_supplies_current_required_fields(self):
        calls = _constructor_keywords("HealthCheck")
        self.assertEqual(len(calls), 1)
        self.assertEqual(CURRENT_REQUIRED_FIELDS - calls[0], set())


if __name__ == "__main__":
    unittest.main()
