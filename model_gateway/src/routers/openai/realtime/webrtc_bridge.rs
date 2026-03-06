//! WebRTC-to-WebRTC relay bridge.
//!
//! Analogous to [`super::proxy`] for WebSocket, this module implements a
//! bidirectional bridge between a client-facing and an upstream-facing WebRTC
//! peer connection.  SMG terminates both connections and relays data-channel
//! messages plus audio RTP packets, giving it full visibility into the traffic.

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};

use openai_protocol::realtime_events::{ClientEvent, ServerEvent};
use str0m::{
    change::{SdpAnswer, SdpOffer},
    channel::{ChannelData, ChannelId},
    media::{Direction, MediaKind, Mid},
    net::{Protocol, Receive},
    rtp::RtpPacket,
    Candidate, Event, Input, Output, Rtc, RtcConfig,
};
use tokio::net::UdpSocket;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use super::registry::{ConnectionState, RealtimeRegistry};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`WebRtcBridge::setup`] that the caller can
/// `tokio::spawn` to run the relay loop.
pub struct WebRtcBridge {
    client_rtc: Rtc,
    client_socket: UdpSocket,
    /// The candidate address advertised in ICE (resolved IP + port).
    /// Used as `destination` in `Receive::new` so str0m matches packets.
    client_candidate_addr: SocketAddr,

    upstream_rtc: Rtc,
    upstream_socket: UdpSocket,
    /// The candidate address advertised in ICE (resolved IP + port).
    upstream_candidate_addr: SocketAddr,

    /// Data channel id on the *client* peer (set when ChannelOpen fires).
    client_channel: Option<ChannelId>,
    /// Data channel id on the *upstream* peer (created during SDP setup).
    upstream_channel: Option<ChannelId>,

    /// Audio mid on the *upstream* peer — used to look up a TX stream for
    /// forwarding client audio.
    upstream_audio_mid: Option<Mid>,
    /// Audio mid on the *client* peer — used to look up a TX stream for
    /// forwarding upstream audio.
    client_audio_mid: Option<Mid>,

    /// Upstream data-channel messages received before the client channel opens.
    /// Flushed once `client_channel` becomes `Some`.
    pending_to_client: Vec<(bool, Vec<u8>)>,

    call_id: String,
    cancel_token: CancellationToken,
}

impl WebRtcBridge {
    /// Create both peer connections, perform SDP exchange with upstream, and
    /// return `(bridge, client_sdp_answer)`.
    ///
    /// The caller should then register the call, spawn `bridge.run()`, and
    /// return the SDP answer to the client.
    #[expect(
        clippy::too_many_arguments,
        reason = "setup requires all connection parameters"
    )]
    pub async fn setup(
        client_sdp_offer_str: &str,
        upstream_url: &str,
        auth_header: &str,
        session_config: Option<serde_json::Value>,
        call_id: String,
        cancel_token: CancellationToken,
        http_client: &reqwest::Client,
        bind_addr: IpAddr,
        stun_server: Option<SocketAddr>,
    ) -> anyhow::Result<(Self, String)> {
        // -- 1. Bind two UDP sockets (ephemeral ports) -----------------------
        // In production, `bind_addr` is typically `0.0.0.0` and both sockets
        // share the same bind/candidate IP (the server's routable address).
        //
        // For local development the user may set `--webrtc-bind-addr 127.0.0.1`
        // so the browser (same machine) can reach the client-facing peer.
        // A loopback address can't reach external servers, so the upstream
        // socket falls back to `0.0.0.0` in that case.
        let upstream_bind = if bind_addr.is_loopback() {
            IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)
        } else {
            bind_addr
        };

        let client_candidate_ip = resolve_candidate_ip(bind_addr)?;
        let upstream_candidate_ip = resolve_candidate_ip(upstream_bind)?;

        let client_socket = UdpSocket::bind(SocketAddr::new(bind_addr, 0)).await?;
        let upstream_socket = UdpSocket::bind(SocketAddr::new(upstream_bind, 0)).await?;

        let client_local = client_socket.local_addr()?;
        let upstream_local = upstream_socket.local_addr()?;

        let client_candidate = SocketAddr::new(client_candidate_ip, client_local.port());
        let upstream_candidate = SocketAddr::new(upstream_candidate_ip, upstream_local.port());

        // -- 1b. Gather server-reflexive candidate for *upstream* via STUN ---
        // Client-facing peer uses ICE-lite (host candidates only), so no STUN
        // needed there.
        let upstream_srflx = if let Some(stun) = stun_server {
            let u = stun_gather_srflx(&upstream_socket, stun).await;
            if u.is_some() {
                info!(call_id, upstream_srflx = ?u, "STUN gathering complete");
            } else {
                warn!(call_id, %stun, "STUN gathering failed for upstream socket");
            }
            u
        } else {
            None
        };

        debug!(
            call_id,
            client_addr = %client_local,
            upstream_addr = %upstream_local,
            client_candidate_ip = %client_candidate_ip,
            upstream_candidate_ip = %upstream_candidate_ip,
            "Bound UDP sockets for WebRTC bridge"
        );

        // -- 2. Create upstream Rtc (offerer) --------------------------------
        let now = Instant::now();
        let mut upstream_rtc = RtcConfig::new().set_rtp_mode(true).build(now);
        upstream_rtc.add_local_candidate(Candidate::host(upstream_candidate, Protocol::Udp)?);
        if let Some(srflx) = upstream_srflx {
            upstream_rtc.add_local_candidate(Candidate::server_reflexive(
                srflx,
                upstream_candidate,
                Protocol::Udp,
            )?);
        }

        // Add audio transceiver + data channel
        let mut sdp_api = upstream_rtc.sdp_api();
        let upstream_audio_mid =
            sdp_api.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None);
        let upstream_channel_id = sdp_api.add_channel("oai-events".to_string());
        let (upstream_offer, pending) = sdp_api
            .apply()
            .ok_or_else(|| anyhow::anyhow!("SDP apply produced no offer"))?;

        // -- 3. Send offer to OpenAI, get answer ----------------------------
        let upstream_answer = send_sdp_to_upstream(
            http_client,
            upstream_url,
            auth_header,
            &upstream_offer.to_sdp_string(),
            session_config,
        )
        .await?;

        upstream_rtc
            .sdp_api()
            .accept_answer(pending, upstream_answer)?;

        // -- 4. Create client Rtc (answerer) --------------------------------
        // ICE-lite: SMG acts as a server — only responds to the browser's
        // connectivity checks.  This avoids needing to resolve mDNS
        // candidates the browser may advertise, and is the standard mode
        // for SFU/relay servers.
        let mut client_rtc = RtcConfig::new()
            .set_rtp_mode(true)
            .set_ice_lite(true)
            .build(Instant::now());
        client_rtc.add_local_candidate(Candidate::host(client_candidate, Protocol::Udp)?);
        // ICE-lite: only host candidates are valid per spec. The browser (full
        // ICE agent) will send connectivity checks to our host candidate.
        // srflx/relay candidates are not used with ICE-lite.

        let client_offer = SdpOffer::from_sdp_string(client_sdp_offer_str)?;
        let client_answer = client_rtc.sdp_api().accept_offer(client_offer)?;

        // Discover the audio mid the client negotiated
        let client_audio_mid = find_audio_mid(&client_rtc);

        let answer_sdp = client_answer.to_sdp_string();
        info!(
            call_id,
            upstream_audio_mid = ?upstream_audio_mid,
            client_audio_mid = ?client_audio_mid,
            "WebRTC bridge SDP exchange complete"
        );
        trace!(call_id, sdp_answer = %answer_sdp, "Client SDP answer");

        Ok((
            Self {
                client_rtc,
                client_socket,
                client_candidate_addr: client_candidate,
                upstream_rtc,
                upstream_socket,
                upstream_candidate_addr: upstream_candidate,
                client_channel: None,
                pending_to_client: Vec::new(),
                upstream_channel: Some(upstream_channel_id),
                upstream_audio_mid: Some(upstream_audio_mid),
                client_audio_mid,
                call_id,
                cancel_token,
            },
            answer_sdp,
        ))
    }

    /// Run the bidirectional relay until cancelled or disconnected.
    pub async fn run(mut self, registry: Arc<RealtimeRegistry>) {
        registry.set_call_state(&self.call_id, ConnectionState::Connected);

        let mut buf_client = vec![0u8; 2000];
        let mut buf_upstream = vec![0u8; 2000];

        loop {
            // Fixed 50ms poll interval — str0m's poll_output drains events so
            // we can't peek without consuming; a short fixed interval is fine
            // for realtime audio latency.
            let timeout = Duration::from_millis(50);

            tokio::select! {
                result = self.client_socket.recv_from(&mut buf_client) => {
                    match result {
                        Ok((n, source)) => {
                            trace!(call_id = self.call_id, %source, n, "Client UDP packet received");
                            let now = Instant::now();
                            let dest = self.client_candidate_addr;
                            match Receive::new(Protocol::Udp, source, dest, &buf_client[..n]) {
                                Ok(recv) => {
                                    if let Err(e) = self.client_rtc.handle_input(Input::Receive(now, recv)) {
                                        warn!(call_id = self.call_id, error = %e, %source, "client_rtc rejected input");
                                    }
                                }
                                Err(e) => {
                                    debug!(call_id = self.call_id, error = %e, %source, n, "Client Receive::new failed (not STUN/DTLS/RTP?)");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(call_id = self.call_id, error = %e, "Client UDP recv error");
                        }
                    }
                }

                result = self.upstream_socket.recv_from(&mut buf_upstream) => {
                    match result {
                        Ok((n, source)) => {
                            let now = Instant::now();
                            let dest = self.upstream_candidate_addr;
                            match Receive::new(Protocol::Udp, source, dest, &buf_upstream[..n]) {
                                Ok(recv) => {
                                    if let Err(e) = self.upstream_rtc.handle_input(Input::Receive(now, recv)) {
                                        warn!(call_id = self.call_id, error = %e, %source, "upstream_rtc rejected input");
                                    }
                                }
                                Err(e) => {
                                    debug!(call_id = self.call_id, error = %e, %source, n, "Upstream Receive::new failed");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(call_id = self.call_id, error = %e, "Upstream UDP recv error");
                        }
                    }
                }

                () = tokio::time::sleep(timeout) => {
                    let now = Instant::now();
                    let _ = self.client_rtc.handle_input(Input::Timeout(now));
                    let _ = self.upstream_rtc.handle_input(Input::Timeout(now));
                }

                () = self.cancel_token.cancelled() => {
                    debug!(call_id = self.call_id, "WebRTC bridge cancelled");
                    break;
                }
            }

            // Drain outputs from both peers
            self.process_client_outputs().await;
            self.process_upstream_outputs().await;

            // Exit if either peer is dead
            if !self.client_rtc.is_alive() || !self.upstream_rtc.is_alive() {
                info!(
                    call_id = self.call_id,
                    client_alive = self.client_rtc.is_alive(),
                    upstream_alive = self.upstream_rtc.is_alive(),
                    "WebRTC bridge peer disconnected"
                );
                break;
            }
        }

        // Graceful disconnect
        self.client_rtc.disconnect();
        self.upstream_rtc.disconnect();
        registry.set_call_state(&self.call_id, ConnectionState::Disconnected);
        debug!(call_id = self.call_id, "WebRTC bridge ended");
    }
}

// ---------------------------------------------------------------------------
// Internal: output processing
// ---------------------------------------------------------------------------

impl WebRtcBridge {
    /// Drain all pending outputs from the client Rtc.
    async fn process_client_outputs(&mut self) {
        loop {
            match self.client_rtc.poll_output() {
                Ok(Output::Transmit(t)) => {
                    trace!(
                        call_id = self.call_id,
                        dest = %t.destination,
                        len = t.contents.len(),
                        "Client Rtc → transmit"
                    );
                    let _ = self.client_socket.send_to(&t.contents, t.destination).await;
                }
                Ok(Output::Event(event)) => {
                    self.handle_client_event(event);
                }
                Ok(Output::Timeout(_)) | Err(_) => break,
            }
        }
    }

    /// Drain all pending outputs from the upstream Rtc.
    async fn process_upstream_outputs(&mut self) {
        loop {
            match self.upstream_rtc.poll_output() {
                Ok(Output::Transmit(t)) => {
                    trace!(
                        call_id = self.call_id,
                        dest = %t.destination,
                        len = t.contents.len(),
                        "Upstream Rtc → transmit"
                    );
                    let _ = self
                        .upstream_socket
                        .send_to(&t.contents, t.destination)
                        .await;
                }
                Ok(Output::Event(event)) => {
                    self.handle_upstream_event(event);
                }
                Ok(Output::Timeout(_)) | Err(_) => break,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Client event → relay to upstream
    // -----------------------------------------------------------------------

    fn handle_client_event(&mut self, event: Event) {
        match event {
            Event::Connected => {
                info!(call_id = self.call_id, "Client peer connected");
            }

            Event::IceConnectionStateChange(state) => {
                info!(call_id = self.call_id, ?state, "Client ICE state");
            }

            Event::ChannelOpen(id, label) => {
                debug!(
                    call_id = self.call_id,
                    ?id,
                    label,
                    "Client data channel opened"
                );
                self.client_channel = Some(id);
                // Flush any upstream events that arrived before the client channel opened
                self.flush_pending_to_client();
            }

            Event::ChannelData(data) => {
                log_client_to_upstream(&self.call_id, &data);
                // Forward to upstream data channel
                if let Some(ch_id) = self.upstream_channel {
                    if let Some(mut ch) = self.upstream_rtc.channel(ch_id) {
                        let _ = ch.write(data.binary, &data.data);
                    }
                }
            }

            Event::ChannelClose(id) => {
                debug!(call_id = self.call_id, ?id, "Client data channel closed");
            }

            Event::RtpPacket(pkt) => {
                trace!(call_id = self.call_id, "Client→Upstream RTP");
                self.forward_rtp_to_upstream(&pkt);
            }

            Event::MediaAdded(added) => {
                debug!(
                    call_id = self.call_id,
                    mid = ?added.mid,
                    kind = ?added.kind,
                    "Client media added"
                );
            }

            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Upstream event → relay to client
    // -----------------------------------------------------------------------

    fn handle_upstream_event(&mut self, event: Event) {
        match event {
            Event::Connected => {
                info!(call_id = self.call_id, "Upstream peer connected");
            }

            Event::IceConnectionStateChange(state) => {
                info!(call_id = self.call_id, ?state, "Upstream ICE state");
            }

            Event::ChannelOpen(id, label) => {
                debug!(
                    call_id = self.call_id,
                    ?id,
                    label,
                    "Upstream data channel opened"
                );
                self.upstream_channel = Some(id);
            }

            Event::ChannelData(data) => {
                log_upstream_to_client(&self.call_id, &data);
                // Forward to client data channel, or buffer if not open yet
                if let Some(ch_id) = self.client_channel {
                    if let Some(mut ch) = self.client_rtc.channel(ch_id) {
                        let _ = ch.write(data.binary, &data.data);
                    }
                } else {
                    trace!(
                        call_id = self.call_id,
                        "Buffering upstream event (client channel not open)"
                    );
                    self.pending_to_client
                        .push((data.binary, data.data.to_vec()));
                }
            }

            Event::ChannelClose(id) => {
                debug!(call_id = self.call_id, ?id, "Upstream data channel closed");
            }

            Event::RtpPacket(pkt) => {
                trace!(call_id = self.call_id, "Upstream→Client RTP");
                self.forward_rtp_to_client(&pkt);
            }

            Event::MediaAdded(added) => {
                debug!(
                    call_id = self.call_id,
                    mid = ?added.mid,
                    kind = ?added.kind,
                    "Upstream media added"
                );
            }

            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Pending event flush
    // -----------------------------------------------------------------------

    /// Send all buffered upstream events to the now-open client data channel.
    fn flush_pending_to_client(&mut self) {
        let Some(ch_id) = self.client_channel else {
            return;
        };
        let pending = std::mem::take(&mut self.pending_to_client);
        if pending.is_empty() {
            return;
        }
        info!(
            call_id = self.call_id,
            count = pending.len(),
            "Flushing buffered upstream events to client"
        );
        if let Some(mut ch) = self.client_rtc.channel(ch_id) {
            for (binary, data) in pending {
                let _ = ch.write(binary, &data);
            }
        }
    }

    // -----------------------------------------------------------------------
    // RTP forwarding
    // -----------------------------------------------------------------------

    /// Forward an RTP packet from the client to the upstream peer.
    fn forward_rtp_to_upstream(&mut self, pkt: &RtpPacket) {
        if let Some(mid) = self.upstream_audio_mid {
            if let Some(tx) = self.upstream_rtc.direct_api().stream_tx_by_mid(mid, None) {
                let _ = tx.write_rtp(
                    pkt.header.payload_type,
                    pkt.seq_no,
                    pkt.header.timestamp,
                    pkt.timestamp,
                    pkt.header.marker,
                    pkt.header.ext_vals.clone(),
                    true,
                    pkt.payload.clone(),
                );
            }
        }
    }

    /// Forward an RTP packet from upstream to the client peer.
    fn forward_rtp_to_client(&mut self, pkt: &RtpPacket) {
        if let Some(mid) = self.client_audio_mid {
            if let Some(tx) = self.client_rtc.direct_api().stream_tx_by_mid(mid, None) {
                let _ = tx.write_rtp(
                    pkt.header.payload_type,
                    pkt.seq_no,
                    pkt.header.timestamp,
                    pkt.timestamp,
                    pkt.header.marker,
                    pkt.header.ext_vals.clone(),
                    true,
                    pkt.payload.clone(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Logging helpers — mirror proxy.rs patterns
// ---------------------------------------------------------------------------

fn log_client_to_upstream(call_id: &str, data: &ChannelData) {
    if data.binary {
        trace!(call_id, bytes = data.data.len(), "Client→Upstream binary");
        return;
    }
    if let Ok(text) = std::str::from_utf8(&data.data) {
        if let Ok(event) = serde_json::from_str::<ClientEvent>(text) {
            let et: &str = event.event_type();
            match et {
                "input_audio_buffer.append" => {
                    trace!(
                        call_id,
                        event_type = "input_audio_buffer.append",
                        "Client→Upstream"
                    );
                }
                _ => {
                    debug!(call_id, event_type = et, "Client→Upstream");
                }
            }
        } else {
            debug!(call_id, "Client→Upstream (unparsed)");
        }
    }
}

fn log_upstream_to_client(call_id: &str, data: &ChannelData) {
    if data.binary {
        trace!(call_id, bytes = data.data.len(), "Upstream→Client binary");
        return;
    }
    if let Ok(text) = std::str::from_utf8(&data.data) {
        if let Ok(event) = serde_json::from_str::<ServerEvent>(text) {
            let et: &str = event.event_type();
            match et {
                "response.output_audio.delta"
                | "response.output_text.delta"
                | "response.output_audio_transcript.delta"
                | "response.function_call_arguments.delta" => {
                    trace!(call_id, event_type = et, "Upstream→Client");
                }
                "session.created"
                | "session.updated"
                | "response.created"
                | "response.done"
                | "response.function_call_arguments.done"
                | "error" => {
                    info!(call_id, event_type = et, "Upstream→Client");
                }
                _ => {
                    debug!(call_id, event_type = et, "Upstream→Client");
                }
            }
        } else {
            debug!(call_id, "Upstream→Client (unparsed)");
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream SDP exchange
// ---------------------------------------------------------------------------

/// Send SMG's SDP offer to upstream (OpenAI) and parse the returned SDP
/// answer.  Supports multipart (with session config) and direct SDP.
async fn send_sdp_to_upstream(
    client: &reqwest::Client,
    upstream_url: &str,
    auth_header: &str,
    sdp_offer: &str,
    session_config: Option<serde_json::Value>,
) -> anyhow::Result<SdpAnswer> {
    let resp = if let Some(session) = session_config {
        // Multipart: SDP offer + session config JSON
        let form = reqwest::multipart::Form::new()
            .part(
                "sdp",
                reqwest::multipart::Part::text(sdp_offer.to_string())
                    .mime_str("application/sdp")?,
            )
            .part(
                "session",
                reqwest::multipart::Part::text(session.to_string()).mime_str("application/json")?,
            );
        client
            .post(upstream_url)
            .header("Authorization", auth_header)
            .multipart(form)
            .send()
            .await?
    } else {
        // Direct SDP
        client
            .post(upstream_url)
            .header("Authorization", auth_header)
            .header("Content-Type", "application/sdp")
            .body(sdp_offer.to_string())
            .send()
            .await?
    };

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Upstream SDP exchange failed: status={status}, body={body}");
    }

    let answer_text = resp.text().await?;
    let answer = SdpAnswer::from_sdp_string(&answer_text)?;
    Ok(answer)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the effective IP for ICE candidates.
///
/// When `addr` is unspecified (`0.0.0.0` / `::`), performs a non-sending UDP
/// "connect" to a public address to let the OS routing table pick the default
/// outbound interface.  No traffic is sent.
fn resolve_candidate_ip(addr: IpAddr) -> anyhow::Result<IpAddr> {
    if !addr.is_unspecified() {
        return Ok(addr);
    }
    let sock = std::net::UdpSocket::bind("0.0.0.0:0")?;
    sock.connect("8.8.8.8:80")?;
    Ok(sock.local_addr()?.ip())
}

/// Find the first audio Mid in an Rtc instance (after SDP negotiation).
fn find_audio_mid(rtc: &Rtc) -> Option<Mid> {
    // After accept_offer, the Rtc has media entries. We probe common mid
    // values — SDP m-lines typically use "0", "1", "2" etc.
    for i in 0..8u32 {
        let mid_str = i.to_string();
        let mid = Mid::from(mid_str.as_str());
        if let Some(media) = rtc.media(mid) {
            if media.kind() == MediaKind::Audio {
                return Some(mid);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// STUN binding — minimal client for server-reflexive candidate gathering
// ---------------------------------------------------------------------------

/// Perform a STUN Binding Request (RFC 5389) on `socket` to discover its
/// server-reflexive (public) address.  Returns `None` on timeout or parse
/// failure — callers should proceed with host-only candidates in that case.
async fn stun_gather_srflx(socket: &UdpSocket, stun_server: SocketAddr) -> Option<SocketAddr> {
    // 20-byte STUN Binding Request
    let mut req = [0u8; 20];
    req[0..2].copy_from_slice(&0x0001u16.to_be_bytes()); // Binding Request
                                                         // Length = 0 (no attributes)
    req[4..8].copy_from_slice(&0x2112_A442u32.to_be_bytes()); // Magic Cookie
                                                              // Transaction ID: first 12 bytes of a UUIDv7
    let txn = uuid::Uuid::now_v7();
    req[8..20].copy_from_slice(&txn.as_bytes()[..12]);

    let local = socket.local_addr().ok();
    if let Err(e) = socket.send_to(&req, stun_server).await {
        warn!(?local, %stun_server, error = %e, "STUN send failed");
        return None;
    }
    debug!(?local, %stun_server, "STUN Binding Request sent");

    let mut buf = [0u8; 512];
    let result = tokio::time::timeout(Duration::from_secs(3), socket.recv_from(&mut buf)).await;
    let (n, from) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            warn!(?local, error = %e, "STUN recv error");
            return None;
        }
        Err(_) => {
            warn!(?local, %stun_server, "STUN response timed out (3s)");
            return None;
        }
    };

    let addr = parse_stun_xor_mapped_address(&buf[..n], &req[8..20]);
    if addr.is_some() {
        info!(?local, %from, srflx = ?addr, "STUN srflx discovered");
    } else {
        warn!(?local, %from, n, "STUN response unparseable");
    }
    addr
}

/// Parse the XOR-MAPPED-ADDRESS attribute from a STUN Binding Success
/// Response.  Returns the decoded `SocketAddr` or `None`.
fn parse_stun_xor_mapped_address(resp: &[u8], txn_id: &[u8]) -> Option<SocketAddr> {
    if resp.len() < 20 {
        return None;
    }
    // 0x0101 = Binding Success Response
    if resp[0] != 0x01 || resp[1] != 0x01 {
        return None;
    }
    if &resp[8..20] != txn_id {
        return None;
    }

    let msg_len = u16::from_be_bytes([resp[2], resp[3]]) as usize;
    let end = (20 + msg_len).min(resp.len());
    let mut off = 20;

    while off + 4 <= end {
        let attr_type = u16::from_be_bytes([resp[off], resp[off + 1]]);
        let attr_len = u16::from_be_bytes([resp[off + 2], resp[off + 3]]) as usize;
        off += 4;
        if off + attr_len > end {
            break;
        }

        // XOR-MAPPED-ADDRESS = 0x0020
        if attr_type == 0x0020 && attr_len >= 8 {
            let family = resp[off + 1];
            if family == 0x01 {
                // IPv4
                let port = u16::from_be_bytes([resp[off + 2], resp[off + 3]]) ^ 0x2112;
                let ip = std::net::Ipv4Addr::new(
                    resp[off + 4] ^ 0x21,
                    resp[off + 5] ^ 0x12,
                    resp[off + 6] ^ 0xA4,
                    resp[off + 7] ^ 0x42,
                );
                return Some(SocketAddr::new(IpAddr::V4(ip), port));
            }
        }

        // Attributes are padded to 4-byte boundaries
        off += (attr_len + 3) & !3;
    }
    None
}
