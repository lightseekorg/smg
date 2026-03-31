"""
Mock inference worker for mesh profiling.

Handles:
  GET  /health          → {"status": "healthy"}
  GET  /v1/models       → model list
  GET  /get_model_info  → model info
  POST /v1/chat/completions → streaming SSE chunks (mimics real engine)

Usage: python3 scripts/mock_worker.py <port>
"""

import http.server
import json
import sys
import time
import uuid


class MockWorkerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "healthy"})
        elif self.path == "/v1/models":
            self._json_response(
                {
                    "data": [
                        {
                            "id": "mock-model",
                            "object": "model",
                            "root": "mock-model",
                            "max_model_len": 4096,
                            "owned_by": "vllm",
                        }
                    ]
                }
            )
        elif self.path == "/get_model_info":
            self._json_response(
                {
                    "model_path": "mock-model",
                    "is_generation": True,
                    "max_total_num_tokens": 4096,
                }
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                req = {}

            if req.get("stream", False):
                self._stream_response(req)
            else:
                self._non_stream_response(req)
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_response(self, req):
        """Stream SSE chunks mimicking a real inference engine."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model = req.get("model", "mock-model")

        # Stream 10-20 token chunks
        num_chunks = 15
        words = [
            "The",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
            "and",
            "then",
            "it",
            "runs",
            "away",
            "fast",
        ]

        for i, word in enumerate(words[:num_chunks]):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " "}
                        if i > 0
                        else {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Final chunk with finish_reason
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": num_chunks,
                "total_tokens": 20 + num_chunks,
            },
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _non_stream_response(self, req):
        """Non-streaming response."""
        self._json_response(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }
        )

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress request logs


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    server = http.server.HTTPServer(("127.0.0.1", port), MockWorkerHandler)
    server.serve_forever()
