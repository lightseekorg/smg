"""Realtime API E2E tests.

Tests for the Realtime WebSocket proxy endpoints including:
- Session lifecycle (connect, session.created, session.update)
- Text generation (single-turn and multi-turn conversations)
- Response cancellation mid-stream
- Response format validation (session.created, response.done, response.text.delta)
- Error handling (invalid events, missing model, missing auth)
"""
