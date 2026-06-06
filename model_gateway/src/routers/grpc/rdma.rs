//! EPD RDMA pixel transport (M2): the gateway exports each image's serialized
//! pixel buffer over NIXL (UCX/RoCE) so the encode worker PULLs it (one-sided
//! READ), instead of shipping ~10-50 MB inline in the Encode gRPC frame. This is
//! the leg that, on the inline path, burns ~31 gateway CPU cores serializing and
//! starves the encode worker to ~2% GPU util (the measured EPD bottleneck).
//!
//! Gated behind `SMG_MM_PIXEL_RDMA`; every NIXL call is reached only when the gate
//! is on (a missing `libnixl_capi.so` aborts on the first stub call, not a no-op),
//! and any export failure falls back to the inline payload so EPD never hard-fails.
//!
//! Contract (empirically validated against NIXL 1.x): the descriptor shipped in
//! `RemoteTensorHandle.descriptor` is `[remote_addr: u64 LE][agent_metadata...]`.
//! The metadata is NIXL-portable (`get_local_md`), the addr is explicit (the Rust
//! `serialize()` is bincode, NOT cross-language). The puller does
//! `add_remote_agent(md)` + `get_xfer_descs([(addr, nbytes, 0)], "DRAM")` + READ,
//! and tags the transfer notif with the `bootstrap_room`; this reaper consumes that
//! notif to deregister + free the pinned buffer (a TTL sweep is the lost-notif net).

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NotificationMap,
    RegistrationHandle,
};
use parking_lot::Mutex;
use tracing::{debug, error, warn};

/// `SMG_MM_PIXEL_RDMA` gate (mirrors `pixel_wire_dtype()`'s OnceLock pattern).
pub(crate) fn rdma_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("SMG_MM_PIXEL_RDMA").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// How long a pinned buffer may live without a free-notif before the reaper
/// force-evicts it (lost notif / dead worker). Well past ViT (~30ms) + READ (~20ms).
const MR_TTL: Duration = Duration::from_secs(30);
/// Reaper poll interval.
const REAPER_TICK: Duration = Duration::from_millis(20);

/// Zero-copy NIXL descriptor over the existing pixel byte buffer (host DRAM).
/// Holds the `Arc<Vec<u8>>` so the registered region stays mapped for the READ.
#[derive(Debug)]
struct HostDramRegion(Arc<Vec<u8>>);

impl MemoryRegion for HostDramRegion {
    // The NIXL trait declares this `unsafe`; the body is a safe Vec::as_ptr. The
    // crate denies unsafe_code globally, so allow it just for this trait impl.
    #[allow(unsafe_code)]
    unsafe fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
    fn size(&self) -> usize {
        self.0.len()
    }
}

impl NixlDescriptor for HostDramRegion {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }
    fn device_id(&self) -> u64 {
        0
    }
}

/// Persistent gateway NIXL agent (one per process). NIXL's `Agent` serializes its
/// own state behind an internal RwLock, but we also serialize register/md/notif
/// through this Mutex to keep the contract simple.
struct GatewayRdma {
    agent: Agent,
    _backend: Backend,
}

/// A pinned, registered pixel buffer kept alive until the encode worker's READ
/// completes (signalled by a notif tagged with the bootstrap_room) or the TTL.
struct MrEntry {
    _buf: Arc<Vec<u8>>,
    _handle: RegistrationHandle,
    at: Instant,
}

static AGENT: OnceLock<Option<Mutex<GatewayRdma>>> = OnceLock::new();
static MR_REGISTRY: OnceLock<DashMap<i32, MrEntry>> = OnceLock::new();

fn registry() -> &'static DashMap<i32, MrEntry> {
    MR_REGISTRY.get_or_init(DashMap::new)
}

/// Lazily build the persistent agent (UCX backend) on first use; spawn the reaper.
/// Returns None (and callers fall back to inline) if NIXL init fails.
fn agent() -> Option<&'static Mutex<GatewayRdma>> {
    AGENT
        .get_or_init(|| {
            if !rdma_enabled() {
                return None;
            }
            match init_agent() {
                Ok(g) => {
                    spawn_reaper();
                    debug!("EPD RDMA: gateway NIXL agent + UCX backend up");
                    Some(Mutex::new(g))
                }
                Err(e) => {
                    error!(error = ?e, "EPD RDMA: agent init failed; falling back to inline pixels");
                    None
                }
            }
        })
        .as_ref()
}

fn init_agent() -> Result<GatewayRdma, nixl_sys::NixlError> {
    // enable_listen_thread + a fixed listen_port so the encode worker (initiator)
    // can do the bidirectional NIXL metadata exchange (fetch_remote_metadata +
    // send_local_metadata) against this gateway (target). Without the listener the
    // worker's connect hits the ephemeral worker port -> "Connection refused"
    // cross-node (same-host worked over shm). enable_prog_thread (default) keeps the
    // UCX worker progressing so one-sided READs complete without our intervention.
    let cfg = AgentConfig {
        enable_listen_thread: true,
        listen_port: i32::from(gw_listen_port()),
        ..Default::default()
    };
    let agent = Agent::new_configured(GATEWAY_AGENT_NAME, &cfg)?;
    let (_mems, params) = agent.get_plugin_params("UCX")?;
    let backend = agent.create_backend("UCX", &params)?;
    Ok(GatewayRdma {
        agent,
        _backend: backend,
    })
}

/// Fixed agent name the encode worker passes to fetch_remote_metadata.
const GATEWAY_AGENT_NAME: &str = "smg-gateway-encode";

/// The gateway's RDMA listener IP (its RoCE address, e.g. 172.16.1.80) that the
/// encode worker dials for the metadata exchange. Empty -> RDMA disabled (inline).
fn gw_listen_ip() -> &'static str {
    static IP: OnceLock<String> = OnceLock::new();
    IP.get_or_init(|| std::env::var("SMG_RDMA_LISTEN_IP").unwrap_or_default())
}

/// The gateway's NIXL listener port (default 18515).
fn gw_listen_port() -> u16 {
    static PORT: OnceLock<u16> = OnceLock::new();
    *PORT.get_or_init(|| {
        std::env::var("SMG_RDMA_LISTEN_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(18515)
    })
}

/// Background reaper: drain free-notifs (tag = bootstrap_room) to deregister the
/// matching pinned buffer, plus a TTL sweep so a lost notif can never leak a MR.
fn spawn_reaper() {
    std::thread::Builder::new()
        .name("epd-rdma-reaper".into())
        .spawn(|| loop {
            std::thread::sleep(REAPER_TICK);
            let Some(g) = AGENT.get().and_then(|o| o.as_ref()) else {
                continue;
            };
            if let Ok(mut notifs) = NotificationMap::new() {
                {
                    let guard = g.lock();
                    let _ = guard.agent.get_notifications(&mut notifs, None);
                }
                if let Ok(map) = notifs.take_notifs() {
                    for (_agent, tags) in map {
                        for tag in tags {
                            if let Ok(room) = tag.parse::<i32>() {
                                registry().remove(&room);
                            }
                        }
                    }
                }
            }
            registry().retain(|_room, e| e.at.elapsed() < MR_TTL);
        })
        .ok();
}

/// Register `bytes` (the serialized pixel buffer for one image) with NIXL, keyed by
/// `room` (its bootstrap_room), and return the wire descriptor for the puller:
/// `[addr u64 LE][port u16 LE][listener_ip utf8]`. The worker connects to the
/// gateway's listener (ip:port) for the bidirectional metadata exchange, then reads
/// `nbytes` (the proto field) from `addr`. The buffer stays pinned in MR_REGISTRY
/// until the worker's free-notif or the TTL. On any failure returns `Err(bytes)` so
/// the caller re-attaches them as the inline payload (no behaviour change).
pub(crate) fn export_pixel_buffer(room: i32, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
    if gw_listen_ip().is_empty() {
        // No listener IP configured -> cannot do the cross-node metadata exchange.
        return Err(bytes);
    }
    let Some(g) = agent() else {
        return Err(bytes);
    };
    // The Vec's heap data address is stable across the move into the Arc; capture it
    // as the remote READ source address.
    let buf = Arc::new(bytes);
    let addr = buf.as_ptr() as u64;
    let region = HostDramRegion(buf.clone());
    {
        let guard = g.lock();
        let handle = match guard.agent.register_memory(&region, None) {
            Ok(h) => h,
            Err(e) => {
                warn!(error = ?e, room, "EPD RDMA: register_memory failed; inline fallback");
                drop(guard);
                drop(region);
                return Err(Arc::try_unwrap(buf).unwrap_or_default());
            }
        };
        registry().insert(
            room,
            MrEntry {
                _buf: buf,
                _handle: handle,
                at: Instant::now(),
            },
        );
    }
    drop(region);

    let ip = gw_listen_ip().as_bytes();
    let port = gw_listen_port();
    let mut descriptor = Vec::with_capacity(10 + ip.len());
    descriptor.extend_from_slice(&addr.to_le_bytes());
    descriptor.extend_from_slice(&port.to_le_bytes());
    descriptor.extend_from_slice(ip);
    debug!(
        room,
        addr,
        "EPD RDMA: exported pixel MR (listener {}:{})",
        gw_listen_ip(),
        port
    );
    Ok(descriptor)
}
