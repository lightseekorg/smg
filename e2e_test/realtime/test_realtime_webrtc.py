"""E2E tests for the Realtime WebRTC relay (/v1/realtime/calls).

Tests the gateway's ability to relay WebRTC sessions via SDP signaling:
- SDP offer/answer exchange (direct and multipart content types)
- SDP answer format validation (ICE, DTLS, audio media)
- Error handling (missing model, invalid SDP, missing auth, wrong method)
- Concurrency (parallel SDP offers)
- Hangup call lifecycle
- Data channel events (session lifecycle, text response, error handling)

Prerequisites:
- OPENAI_API_KEY environment variable set

Usage:
    pytest e2e_test/realtime/test_realtime_webrtc.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest
import requests

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"

# Minimal SDP offer — enough for the signaling endpoint to accept and return
# an answer.  We only test the HTTP signaling layer, not actual media flow.
MINIMAL_SDP_OFFER = (
    "v=0\r\n"
    "o=- 0 0 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "a=group:BUNDLE 0 1\r\n"
    "a=extmap-allow-mixed\r\n"
    "a=msid-semantic: WMS\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "c=IN IP4 0.0.0.0\r\n"
    "a=rtcp:9 IN IP4 0.0.0.0\r\n"
    "a=ice-ufrag:test\r\n"
    "a=ice-pwd:testpasswordtestpassword\r\n"
    "a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
    ":00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\n"
    "a=setup:actpass\r\n"
    "a=mid:0\r\n"
    "a=sendrecv\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
    "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
    "c=IN IP4 0.0.0.0\r\n"
    "a=ice-ufrag:test\r\n"
    "a=ice-pwd:testpasswordtestpassword\r\n"
    "a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
    ":00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\n"
    "a=setup:actpass\r\n"
    "a=mid:1\r\n"
    "a=sctp-port:5000\r\n"
    "a=max-message-size:262144\r\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_sdp(calls_url, auth_headers, *, model=None, sdp=MINIMAL_SDP_OFFER, timeout=30):
    """POST an SDP offer to the calls endpoint and return the response."""
    url = f"{calls_url}?model={model}" if model else calls_url
    return requests.post(
        url,
        data=sdp,
        headers={**auth_headers, "Content-Type": "application/sdp"},
        timeout=timeout,
    )


def _post_multipart(calls_url, auth_headers, *, session_config, sdp=MINIMAL_SDP_OFFER, timeout=30):
    """POST a multipart SDP offer with session config and return the response."""
    files = {
        "sdp": ("offer.sdp", sdp, "application/sdp"),
        "session": ("session.json", json.dumps(session_config), "application/json"),
    }
    return requests.post(
        calls_url,
        files=files,
        headers=auth_headers,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway():
    """Launch a cloud gateway in OpenAI mode for realtime WebRTC tests."""
    from infra import launch_cloud_gateway

    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")

    gw = launch_cloud_gateway("openai", history_backend="memory")
    yield gw
    gw.shutdown()


@pytest.fixture(scope="module")
def dc_gateway():
    """Dedicated gateway for data-channel tests.

    Signaling tests leave zombie bridges (fake SDP, no real ICE peer) that
    consume sockets and poll-loop CPU. A fresh gateway avoids resource
    starvation that causes DTLS handshake timeouts in CI.
    """
    from infra import launch_cloud_gateway

    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")

    gw = launch_cloud_gateway("openai", history_backend="memory")
    yield gw
    gw.shutdown()


@pytest.fixture()
def calls_url(gateway):
    """Build the realtime calls URL."""
    return f"http://{gateway.host}:{gateway.port}/v1/realtime/calls"


@pytest.fixture()
def auth_headers():
    """Build the Authorization headers."""
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestRealtimeWebRtcSignaling:
    """E2E tests for the WebRTC SDP signaling layer."""

    def test_sdp_offer_answer_direct(self, calls_url, auth_headers):
        """POST application/sdp offer should return an SDP answer with 201."""
        resp = _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL)
        assert resp.status_code == 201, (
            f"Expected 201 CREATED, got {resp.status_code}: {resp.text[:500]}"
        )
        content_type = resp.headers.get("Content-Type", "")
        assert "application/sdp" in content_type, (
            f"Expected application/sdp Content-Type, got: {content_type}"
        )
        # Answer SDP must start with "v=0"
        answer_sdp = resp.text
        assert answer_sdp.startswith("v=0"), f"Answer SDP doesn't look valid: {answer_sdp[:200]}"
        logger.info("Direct SDP: got %d-byte answer SDP", len(answer_sdp))

    def test_sdp_offer_answer_multipart(self, calls_url, auth_headers):
        """POST multipart/form-data with sdp + session fields should return SDP answer."""
        resp = _post_multipart(
            calls_url,
            auth_headers,
            session_config={
                "model": REALTIME_MODEL,
                "type": "realtime",
                "audio": {"output": {"voice": "alloy"}},
            },
        )
        assert resp.status_code == 201, (
            f"Expected 201 CREATED, got {resp.status_code}: {resp.text[:500]}"
        )
        answer_sdp = resp.text
        assert answer_sdp.startswith("v=0"), f"Answer SDP doesn't look valid: {answer_sdp[:200]}"
        logger.info("Multipart SDP: got %d-byte answer SDP", len(answer_sdp))

    def test_missing_model_returns_error(self, calls_url, auth_headers):
        """POST without model should return 400."""
        resp = _post_sdp(calls_url, auth_headers, timeout=10)
        assert resp.status_code == 400, f"Expected 400 for missing model, got {resp.status_code}"

    def test_missing_auth_returns_unauthorized(self, calls_url):
        """POST without Authorization header should return 401."""
        resp = _post_sdp(calls_url, {}, model=REALTIME_MODEL, timeout=10)
        assert resp.status_code == 401, f"Expected 401 for missing auth, got {resp.status_code}"

    def test_unsupported_content_type_returns_error(self, calls_url, auth_headers):
        """POST with unsupported Content-Type should return 400."""
        url = f"{calls_url}?model={REALTIME_MODEL}"
        resp = requests.post(
            url,
            data='{"sdp": "fake"}',
            headers={**auth_headers, "Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for unsupported content type, got {resp.status_code}"
        )

    def test_invalid_sdp_returns_error(self, calls_url, auth_headers):
        """POST with invalid SDP body should return 400."""
        resp = _post_sdp(
            calls_url,
            auth_headers,
            model=REALTIME_MODEL,
            sdp="this is not valid SDP",
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400 for invalid SDP, got {resp.status_code}"

    def test_multipart_missing_sdp_returns_error(self, calls_url, auth_headers):
        """POST multipart without sdp field should return 400."""
        # Not using _post_multipart since we intentionally omit the sdp field
        session_config = json.dumps({"model": REALTIME_MODEL, "type": "realtime"})
        resp = requests.post(
            calls_url,
            files={
                "session": ("session.json", session_config, "application/json"),
            },
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing sdp field, got {resp.status_code}"
        )

    def test_multipart_missing_model_returns_error(self, calls_url, auth_headers):
        """POST multipart with session but no model should return 400."""
        resp = _post_multipart(
            calls_url,
            auth_headers,
            session_config={"audio": {"output": {"voice": "alloy"}}},
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing model in session, got {resp.status_code}"
        )

    def test_empty_body_returns_error(self, calls_url, auth_headers):
        """POST with empty SDP body should return an error."""
        resp = _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL, sdp="", timeout=10)
        assert resp.status_code == 400, f"Expected 400 for empty SDP body, got {resp.status_code}"

    def test_get_method_not_allowed(self, calls_url, auth_headers):
        """GET /v1/realtime/calls should return 405 Method Not Allowed."""
        url = f"{calls_url}?model={REALTIME_MODEL}"
        resp = requests.get(url, headers=auth_headers, timeout=10)
        assert resp.status_code == 405, f"Expected 405 for GET method, got {resp.status_code}"

    def test_put_method_not_allowed(self, calls_url, auth_headers):
        """PUT /v1/realtime/calls should return 405 Method Not Allowed."""
        url = f"{calls_url}?model={REALTIME_MODEL}"
        resp = requests.put(
            url,
            data=MINIMAL_SDP_OFFER,
            headers={**auth_headers, "Content-Type": "application/sdp"},
            timeout=10,
        )
        assert resp.status_code == 405, f"Expected 405 for PUT method, got {resp.status_code}"

    def test_multipart_extra_fields_accepted(self, calls_url, auth_headers):
        """POST multipart with extra unknown fields should be silently ignored by SMG."""
        session_config = json.dumps({"model": REALTIME_MODEL, "type": "realtime"})
        resp = requests.post(
            calls_url,
            files={
                "sdp": ("offer.sdp", MINIMAL_SDP_OFFER, "application/sdp"),
                "session": ("session.json", session_config, "application/json"),
                "extra": ("extra.txt", "some extra data", "text/plain"),
            },
            headers=auth_headers,
            timeout=30,
        )
        assert resp.status_code == 201, (
            f"Expected 201 CREATED with extra fields, got {resp.status_code}: {resp.text[:500]}"
        )

    def test_concurrent_calls(self, calls_url, auth_headers):
        """Multiple concurrent SDP offers should all succeed."""
        num_calls = 3

        def _make_call():
            return _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL)

        with ThreadPoolExecutor(max_workers=num_calls) as pool:
            futures = [pool.submit(_make_call) for _ in range(num_calls)]
            results = [f.result() for f in as_completed(futures)]

        for i, resp in enumerate(results):
            assert resp.status_code == 201, (
                f"Concurrent call {i} failed: {resp.status_code} {resp.text[:200]}"
            )
            assert resp.text.startswith("v=0"), (
                f"Concurrent call {i} SDP invalid: {resp.text[:200]}"
            )
        logger.info("All %d concurrent calls succeeded", num_calls)


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestRealtimeWebRtcAnswerFormat:
    """Validate the SDP answer format returned by the signaling endpoint."""

    def test_answer_sdp_format(self, calls_url, auth_headers):
        """SDP answer must have correct Content-Type, required SDP fields, ICE, DTLS, and audio."""
        resp = _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL)
        assert resp.status_code == 201

        # Content-Type
        ct = resp.headers.get("Content-Type", "")
        assert "application/sdp" in ct, f"Expected application/sdp, got: {ct}"

        # Required SDP fields
        answer = resp.text
        assert "v=0" in answer, "Missing v= line in SDP answer"
        assert "\no=" in answer or answer.startswith("v=0\r\no="), "Missing o= line"
        assert "\ns=" in answer or "\r\ns=" in answer, "Missing s= line"
        assert "\nm=" in answer or "\r\nm=" in answer, "Missing m= (media) line"

        # ICE credentials
        assert "a=ice-ufrag:" in answer, "Missing ice-ufrag in SDP answer"
        assert "a=ice-pwd:" in answer, "Missing ice-pwd in SDP answer"

        # DTLS fingerprint
        assert "a=fingerprint:" in answer, "Missing DTLS fingerprint in SDP answer"

        # Audio media
        assert "m=audio" in answer, "Missing audio media line in SDP answer"
        logger.info("SDP answer format validated (%d bytes)", len(answer))

    def test_multipart_answer_matches_direct(self, calls_url, auth_headers):
        """Multipart and direct SDP flows should return structurally similar answers."""
        direct_resp = _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL)
        multi_resp = _post_multipart(
            calls_url,
            auth_headers,
            session_config={"model": REALTIME_MODEL, "type": "realtime"},
        )
        assert direct_resp.status_code == 201
        assert multi_resp.status_code == 201

        # Both must be valid SDP with audio
        for label, text in [("direct", direct_resp.text), ("multipart", multi_resp.text)]:
            assert text.startswith("v=0"), f"{label} SDP doesn't start with v=0"
            assert "m=audio" in text, f"{label} SDP missing audio media"
            assert "a=ice-ufrag:" in text, f"{label} SDP missing ICE credentials"
        logger.info("Direct and multipart answers are structurally consistent")


# ---------------------------------------------------------------------------
# Data channel event tests (requires aiortc)
# ---------------------------------------------------------------------------

RECV_TIMEOUT = 60  # seconds — DTLS handshake can be slow in CI


def _parse_event(raw: str | bytes) -> dict | None:
    """Parse a JSON event from a data channel message."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Non-JSON data channel message: %s", str(raw)[:200])
        return None


async def _recv_dc_event(
    dc, *, event_type: str | None = None, timeout: float = RECV_TIMEOUT
) -> dict:
    """Receive the next JSON event from a data channel, optionally filtering by type."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    queue: asyncio.Queue = asyncio.Queue()

    def _on_message(msg):
        queue.put_nowait(msg)

    dc.on("message", _on_message)
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for data channel event type={event_type}")
            raw = await asyncio.wait_for(queue.get(), timeout=remaining)
            event = _parse_event(raw)
            if event is None:
                continue
            if event_type is None or event.get("type") == event_type:
                return event
            logger.debug(
                "Skipping DC event %s while waiting for %s",
                event.get("type"),
                event_type,
            )
    finally:
        dc.remove_all_listeners("message")


async def _collect_dc_response_text(dc, *, timeout: float = RECV_TIMEOUT) -> str:
    """Collect text deltas from the data channel until response.done."""
    parts: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    queue: asyncio.Queue = asyncio.Queue()

    def _on_message(msg):
        queue.put_nowait(msg)

    dc.on("message", _on_message)
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for response.done on data channel")
            raw = await asyncio.wait_for(queue.get(), timeout=remaining)
            event = _parse_event(raw)
            if event is None:
                continue
            etype = event.get("type", "")
            if etype == "response.text.delta" and event.get("delta"):
                parts.append(event["delta"])
            elif etype == "response.done":
                break
            elif etype == "error":
                raise RuntimeError(f"Upstream error: {json.dumps(event)}")
    finally:
        dc.remove_all_listeners("message")
    return "".join(parts)


async def _establish_webrtc_session(gateway_host, gateway_port, auth_headers):
    """Complete SDP signaling + ICE/DTLS handshake, return (pc, dc).

    Uses aiortc to create a real WebRTC peer connection to the SMG gateway,
    waits for the data channel to open, and returns the peer connection and
    data channel objects.
    """
    from aiortc import RTCPeerConnection, RTCSessionDescription

    pc = RTCPeerConnection()

    # Create a data channel (matches the "oai-events" channel SMG expects)
    dc = pc.createDataChannel("oai-events")

    # Add audio transceiver to match expected SDP structure
    pc.addTransceiver("audio", direction="sendrecv")

    # Create and set local offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send offer to SMG signaling endpoint (async to avoid blocking the aiortc event loop)
    calls_url = f"http://{gateway_host}:{gateway_port}/v1/realtime/calls?model={REALTIME_MODEL}"
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.post(
            calls_url,
            content=pc.localDescription.sdp,
            headers={**auth_headers, "Content-Type": "application/sdp"},
            timeout=30,
        )
    assert resp.status_code == 201, f"SDP signaling failed: {resp.status_code} {resp.text[:500]}"

    # Set remote answer
    answer = RTCSessionDescription(sdp=resp.text, type="answer")
    await pc.setRemoteDescription(answer)

    # Wait for data channel to open
    dc_open = asyncio.Event()
    original_dc = dc

    @dc.on("open")
    def _on_open():
        dc_open.set()

    # If SMG creates the data channel (server-initiated), capture it
    @pc.on("datachannel")
    def _on_datachannel(channel):
        nonlocal original_dc, dc_open
        if channel.label == "oai-events":
            original_dc = channel
            dc_open.set()

    await asyncio.wait_for(dc_open.wait(), timeout=RECV_TIMEOUT)
    return pc, original_dc


async def _open_text_session(gateway_host, gateway_port, auth_headers):
    """Establish WebRTC connection, wait for session.created, configure text mode.

    Returns (pc, dc) with the session already in text-only mode.
    """
    pc, dc = await _establish_webrtc_session(gateway_host, gateway_port, auth_headers)
    await _recv_dc_event(dc, event_type="session.created")
    _send_dc_event(dc, "session.update", session={"modalities": ["text"]})
    await _recv_dc_event(dc, event_type="session.updated")
    return pc, dc


def _send_dc_event(dc, event_type: str, **payload):
    """Send a JSON event over the data channel."""
    dc.send(json.dumps({"type": event_type, **payload}))


def _send_user_message(dc, text: str):
    """Send a user text message and trigger a text response."""
    _send_dc_event(
        dc,
        "conversation.item.create",
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    )
    _send_dc_event(dc, "response.create", response={"modalities": ["text"]})


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.flaky(reruns=4)
@pytest.mark.xfail(reason="DTLS handshake flaky in CI (aiortc↔str0m)", strict=False)
class TestRealtimeWebRtcDataChannel:
    """E2E tests for client/server events over the WebRTC data channel.

    These tests establish a real WebRTC peer connection via aiortc,
    then exchange Realtime API events over the data channel — mirroring
    the WebSocket event tests in test_realtime_ws.py.
    """

    def test_session_created_on_connect(self, dc_gateway, auth_headers):
        """Connecting should receive a session.created event on the data channel."""

        async def _run():
            pc, dc = await _establish_webrtc_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                event = await _recv_dc_event(dc, event_type="session.created")
                assert event["type"] == "session.created"
                assert "session" in event
                assert event["session"].get("model") is not None
                logger.info("DC session.created: id=%s", event["session"].get("id"))
            finally:
                await pc.close()

        asyncio.run(_run())

    def test_session_update(self, dc_gateway, auth_headers):
        """Sending session.update over data channel should receive session.updated."""

        async def _run():
            pc, dc = await _establish_webrtc_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                await _recv_dc_event(dc, event_type="session.created")
                _send_dc_event(dc, "session.update", session={"modalities": ["text"]})
                event = await _recv_dc_event(dc, event_type="session.updated")
                assert event["type"] == "session.updated"
                assert event["session"].get("modalities") == ["text"]
                logger.info("DC session.updated successfully")
            finally:
                await pc.close()

        asyncio.run(_run())

    def test_text_response(self, dc_gateway, auth_headers):
        """Full text round-trip over data channel: user message -> text response."""

        async def _run():
            pc, dc = await _open_text_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                _send_user_message(dc, "Say hello in one short sentence.")
                text = await _collect_dc_response_text(dc)
                assert len(text) > 0, "Expected non-empty text response"
                logger.info("DC text response: %s", text[:100])
            finally:
                await pc.close()

        asyncio.run(_run())

    def test_conversation_item_created_event(self, dc_gateway, auth_headers):
        """Sending conversation.item.create should echo conversation.item.created."""

        async def _run():
            pc, dc = await _open_text_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                _send_dc_event(
                    dc,
                    "conversation.item.create",
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hi"}],
                    },
                )
                event = await _recv_dc_event(dc, event_type="conversation.item.created")
                assert event["type"] == "conversation.item.created"
                assert event["item"]["role"] == "user"
                logger.info("DC conversation.item.created: id=%s", event["item"].get("id"))
            finally:
                await pc.close()

        asyncio.run(_run())

    def test_invalid_event_returns_error(self, dc_gateway, auth_headers):
        """Sending an unknown event type should return an error event."""

        async def _run():
            pc, dc = await _establish_webrtc_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                await _recv_dc_event(dc, event_type="session.created")
                _send_dc_event(dc, "totally.bogus.event")
                event = await _recv_dc_event(dc, event_type="error")
                assert event["type"] == "error"
                logger.info("DC error: %s", event.get("error", {}).get("message", ""))
            finally:
                await pc.close()

        asyncio.run(_run())

    def test_response_done_format(self, dc_gateway, auth_headers):
        """Validate response.done event schema over data channel."""

        async def _run():
            pc, dc = await _open_text_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                _send_user_message(dc, "Say hi.")
                event = await _recv_dc_event(dc, event_type="response.done")
                assert "event_id" in event
                resp = event["response"]
                assert resp.get("status") == "completed"
                assert isinstance(resp.get("output"), list)
                assert len(resp["output"]) > 0
                item = resp["output"][0]
                assert item.get("role") == "assistant"
                assert isinstance(item.get("content"), list)
                assert len(item["content"]) > 0
                assert isinstance(item["content"][0].get("text"), str)
                usage = resp.get("usage")
                assert isinstance(usage, dict)
                assert usage.get("total_tokens", 0) > 0
                logger.info("DC response.done schema OK: tokens=%d", usage["total_tokens"])
            finally:
                await pc.close()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Upstream error forwarding tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestRealtimeWebRtcErrorForwarding:
    """Verify that upstream HTTP errors are forwarded (not masked as 502)."""

    def test_invalid_api_key_forwards_upstream_401(self, calls_url):
        """Request with invalid API key should forward upstream 401, not return 502."""
        bad_headers = {"Authorization": "Bearer sk-invalid-key-for-testing"}
        resp = _post_sdp(calls_url, bad_headers, model=REALTIME_MODEL, timeout=15)
        # Upstream (OpenAI) should return 401; SMG should forward it, not mask as 502
        assert resp.status_code == 401, (
            f"Expected upstream 401 to be forwarded, got {resp.status_code}: {resp.text[:500]}"
        )

    def test_invalid_model_forwards_upstream_error(self, calls_url, auth_headers):
        """Request with non-existent model should forward upstream error status."""
        resp = _post_sdp(
            calls_url,
            auth_headers,
            model="nonexistent-model-that-does-not-exist-12345",
            timeout=15,
        )
        # Upstream should reject the model; SMG should forward the status (likely 400 or 404)
        assert resp.status_code != 502, (
            f"Expected upstream error to be forwarded, but got 502: {resp.text[:500]}"
        )
        assert 400 <= resp.status_code < 500, (
            f"Expected 4xx from upstream, got {resp.status_code}: {resp.text[:500]}"
        )


# ---------------------------------------------------------------------------
# Bridge cleanup tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.flaky(reruns=2)
class TestRealtimeWebRtcBridgeCleanup:
    """Verify bridge resources are cleaned up after disconnect."""

    def test_reconnect_after_disconnect(self, dc_gateway, auth_headers):
        """After closing a WebRTC session, a new session should succeed.

        Verifies that bridge cleanup releases resources (sockets, registry
        entries) so subsequent connections are not blocked.
        """

        async def _run():
            # Establish first session
            pc1, dc1 = await _establish_webrtc_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            event = await _recv_dc_event(dc1, event_type="session.created")
            assert event["type"] == "session.created"
            logger.info("First session established")

            # Close first session
            await pc1.close()
            # Brief pause for bridge cleanup
            await asyncio.sleep(1)

            # Establish second session — should succeed if cleanup worked
            pc2, dc2 = await _establish_webrtc_session(
                dc_gateway.host,
                dc_gateway.port,
                auth_headers,
            )
            try:
                event = await _recv_dc_event(dc2, event_type="session.created")
                assert event["type"] == "session.created"
                logger.info("Second session after disconnect succeeded")
            finally:
                await pc2.close()

        asyncio.run(_run())

    def test_multiple_sequential_sessions(self, gateway, auth_headers):
        """Open and close 3 sessions sequentially to verify no resource leak."""

        async def _run():
            for i in range(3):
                pc, dc = await _establish_webrtc_session(
                    gateway.host,
                    gateway.port,
                    auth_headers,
                )
                event = await _recv_dc_event(dc, event_type="session.created")
                assert event["type"] == "session.created"
                logger.info("Sequential session %d established", i + 1)
                await pc.close()
                await asyncio.sleep(0.5)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# STUN integration tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.flaky(reruns=2)
class TestRealtimeWebRtcStun:
    """Verify WebRTC signaling works with STUN server configured."""

    @pytest.fixture(scope="class")
    def stun_gateway(self):
        """Launch a gateway with an explicit STUN server configured."""
        from infra import launch_cloud_gateway

        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not set")

        gw = launch_cloud_gateway(
            "openai",
            history_backend="memory",
            extra_args=["--webrtc-stun-server", "stun.l.google.com:19302"],
        )
        yield gw
        gw.shutdown()

    def test_signaling_with_stun_server(self, stun_gateway, auth_headers):
        """SDP signaling should succeed when a STUN server is configured."""
        calls_url = f"http://{stun_gateway.host}:{stun_gateway.port}/v1/realtime/calls"
        resp = _post_sdp(calls_url, auth_headers, model=REALTIME_MODEL)
        assert resp.status_code == 201, (
            f"Expected 201 with STUN configured, got {resp.status_code}: {resp.text[:500]}"
        )
        assert resp.text.startswith("v=0"), "SDP answer should be valid"
        logger.info("Signaling with STUN server succeeded")

    def test_data_channel_with_stun_server(self, stun_gateway, auth_headers):
        """Full data channel round-trip should work with STUN server configured."""

        async def _run():
            pc, dc = await _establish_webrtc_session(
                stun_gateway.host,
                stun_gateway.port,
                auth_headers,
            )
            try:
                event = await _recv_dc_event(dc, event_type="session.created")
                assert event["type"] == "session.created"
                logger.info("Data channel with STUN server: session.created received")
            finally:
                await pc.close()

        asyncio.run(_run())
