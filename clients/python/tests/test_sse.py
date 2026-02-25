"""Tests for SSE parser."""

from smg_client._sse import SseEvent


def test_sse_event_json():
    event = SseEvent(data='{"id": "abc", "text": "hello"}')
    parsed = event.json()
    assert parsed["id"] == "abc"
    assert parsed["text"] == "hello"


def test_sse_event_with_event_type():
    event = SseEvent(data='{"type": "message_start"}', event="message_start")
    assert event.event == "message_start"
    parsed = event.json()
    assert parsed["type"] == "message_start"
