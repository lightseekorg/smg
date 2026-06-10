//! EPD RDMA pixel transport: the gateway exports each image's serialized pixel
//! buffer over NIXL (UCX/RoCE) so the encode worker PULLs it (one-sided READ),
//! instead of shipping ~10-50 MB inline in the Encode gRPC frame. This is the leg
//! that, on the inline path, burns ~31 gateway CPU cores serializing and starves
//! the encode worker to ~2% GPU util (the measured EPD bottleneck).
//!
//! Gated behind `SMG_MM_PIXEL_RDMA`; every NIXL call is reached only when the gate
//! is on (a missing `libnixl_capi.so` aborts on the first stub call, not a no-op),
//! and any export failure falls back to the inline payload so EPD never hard-fails.
//!
//! v2 (pre-registered pool): one host-DRAM arena is registered with NIXL ONCE at
//! init and sub-divided into fixed slots. Per image the hot path only LEASES a free
//! slot, memcpy's the pixels in, and ships `[slot_addr u64 LE][port u16 LE][ip utf8]`
//! -- NO per-image register_memory and NO growing agent metadata. The worker fetches
//! the gateway's (now fixed) metadata ONCE, then READs slot offsets into its own
//! pre-registered landing pool. The transfer notif is tagged with the bootstrap_room;
//! this reaper consumes it to return the slot to the free list (a TTL sweep is the
//! lost-notif net). This removes the v1 per-image control+registration overhead that
//! made the RDMA path slower than inline.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NotificationMap, OptArgs,
    RegistrationHandle,
};
use parking_lot::Mutex;
use tracing::{debug, error};

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

/// How long a leased slot may live without a free-notif before the reaper
/// force-reclaims it (lost notif / dead worker). Well past ViT (~30ms) + READ (~20ms).
const SLOT_TTL: Duration = Duration::from_secs(30);
/// Reaper poll interval.
const REAPER_TICK: Duration = Duration::from_millis(20);
/// Default arena geometry: 64 slots x 32 MiB = 2 GiB host DRAM. A slot must hold one
/// image's bf16 pixel buffer (~10 MiB at 1080p, ~40 MiB at 4k -> raise SLOT_BYTES).
const DEFAULT_POOL_SLOTS: usize = 64;
const DEFAULT_SLOT_BYTES: usize = 32 * 1024 * 1024;

/// NIXL descriptor over the pre-registered arena (host DRAM). Registered once; the
/// backing allocation is leaked (process-lifetime) so `base` is stable forever.
#[derive(Debug)]
struct ArenaRegion {
    base: usize,
    size: usize,
}

impl MemoryRegion for ArenaRegion {
    // The NIXL trait declares this `unsafe`; the body just reinterprets a usize the
    // crate denies unsafe_code globally, so allow it for this trait impl.
    #[allow(unsafe_code)]
    unsafe fn as_ptr(&self) -> *const u8 {
        self.base as *const u8
    }
    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for ArenaRegion {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }
    fn device_id(&self) -> u64 {
        0
    }
}

/// A leased slot, kept until the worker's READ-notif (tagged with bootstrap_room)
/// or the TTL sweep returns it to the free list.
struct OccSlot {
    slot: u32,
    at: Instant,
}

/// Pre-registered host-DRAM arena, sub-divided into fixed slots. One persistent NIXL
/// registration (`_handle`) covers the whole arena; each image's pixels are memcpy'd
/// into a leased slot. A slot is exclusively owned between lease and free, and the
/// write happens-before the descriptor ship which happens-before the worker's READ,
/// so no slot is ever written and read concurrently. The hot path touches no NIXL
/// agent state (only the free-list lock + a lockless memcpy into owned raw memory).
struct SlotArena {
    /// Raw base address of the leaked arena allocation (usize => Send+Sync).
    base: usize,
    slot_bytes: usize,
    n_slots: usize,
    /// The single persistent registration; kept alive for the arena's life.
    _handle: RegistrationHandle,
    /// Available slot indices.
    free: Mutex<Vec<u32>>,
    /// Leased slots keyed by bootstrap_room (free-on-notif + TTL reclaim).
    occupied: DashMap<i32, OccSlot>,
}

impl SlotArena {
    /// Lease a free slot and copy `bytes` into it. Returns `(slot, slot_addr)`, or
    /// `None` if the image exceeds `slot_bytes` or no slot is free (caller -> inline).
    fn lease_and_write(&self, bytes: &[u8]) -> Option<(u32, u64)> {
        if bytes.len() > self.slot_bytes {
            return None;
        }
        let slot = self.free.lock().pop()?;
        let addr = self.base + slot as usize * self.slot_bytes;
        // SAFETY: `slot` is exclusively leased (popped from `free`, returned only via
        // free_slot); [addr, addr+len) is within the arena (len <= slot_bytes) and
        // disjoint from every other leased slot; the worker reads it only after we
        // ship the descriptor below. The arena is raw, leaked memory (no aliasing
        // typed reference). So there is no data race and no overlap.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len());
        }
        Some((slot, addr as u64))
    }

    /// Return the slot leased for `room` to the free list (idempotent).
    fn free_room(&self, room: i32) {
        if let Some((_room, occ)) = self.occupied.remove(&room) {
            self.free.lock().push(occ.slot);
        }
    }
}

/// Persistent gateway NIXL agent (one per process). The agent is touched only at
/// init (register the arena) and by the reaper (drain notifs); the hot path is
/// agent-free, so it stays behind a coarse Mutex without contending the fast path.
struct GatewayRdma {
    agent: Agent,
    // The UCX backend. We must pass it explicitly to register_memory (OptArgs) so the
    // arena gets a UCX/rc rkey: register_memory(None) leaves the registration not bound
    // to UCX, which is fine for TCP (copy) but makes a one-sided rc READ hang (no rkey).
    backend: Backend,
}

static AGENT: OnceLock<Option<Mutex<GatewayRdma>>> = OnceLock::new();
static ARENA: OnceLock<Option<SlotArena>> = OnceLock::new();

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

/// Lazily build + register the slot arena (once) on first export. Requires the agent.
fn arena() -> Option<&'static SlotArena> {
    ARENA
        .get_or_init(|| {
            let g = agent()?;
            match build_arena(g) {
                Ok(a) => {
                    debug!(
                        slots = a.n_slots,
                        slot_bytes = a.slot_bytes,
                        "EPD RDMA: pixel arena registered"
                    );
                    Some(a)
                }
                Err(e) => {
                    error!(error = ?e, "EPD RDMA: arena registration failed; inline fallback");
                    None
                }
            }
        })
        .as_ref()
}

fn build_arena(g: &Mutex<GatewayRdma>) -> Result<SlotArena, nixl_sys::NixlError> {
    let n_slots = pool_slots();
    let slot_bytes = slot_bytes();
    let total = n_slots.saturating_mul(slot_bytes);
    // Leak the arena for the process lifetime: `base` must stay registered + stable,
    // and the gateway agent never shuts down. Holding it as raw memory (not a typed
    // Box) is what makes the per-slot memcpy through `base` sound.
    let boxed = vec![0u8; total].into_boxed_slice();
    let base = Box::into_raw(boxed) as *mut u8 as usize;
    let region = ArenaRegion { base, size: total };
    let handle = {
        let guard = g.lock();
        // Bind the registration to the UCX backend so it gets an rc rkey (needed for
        // the worker's one-sided RDMA READ; without it the READ hangs over rc).
        let mut opt = OptArgs::new()?;
        opt.add_backend(&guard.backend)?;
        guard.agent.register_memory(&region, Some(&opt))?
    };
    Ok(SlotArena {
        base,
        slot_bytes,
        n_slots,
        _handle: handle,
        free: Mutex::new((0..n_slots as u32).collect()),
        occupied: DashMap::new(),
    })
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
    Ok(GatewayRdma { agent, backend })
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

/// Arena slot count (`SMG_RDMA_POOL_SLOTS`, default 64).
fn pool_slots() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SMG_RDMA_POOL_SLOTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_POOL_SLOTS)
    })
}

/// Per-slot byte capacity (`SMG_RDMA_SLOT_BYTES`, default 32 MiB).
fn slot_bytes() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SMG_RDMA_SLOT_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_SLOT_BYTES)
    })
}

/// Background reaper: drain free-notifs (tag = bootstrap_room) to return the leased
/// slot to the free list, plus a TTL sweep so a lost notif can never leak a slot.
fn spawn_reaper() {
    std::thread::Builder::new()
        .name("epd-rdma-reaper".into())
        .spawn(|| loop {
            std::thread::sleep(REAPER_TICK);
            let (Some(g), Some(a)) = (
                AGENT.get().and_then(|o| o.as_ref()),
                ARENA.get().and_then(|o| o.as_ref()),
            ) else {
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
                                a.free_room(room);
                            }
                        }
                    }
                }
            }
            // TTL sweep: reclaim slots whose READ-notif never arrived.
            let now = Instant::now();
            let stale: Vec<i32> = a
                .occupied
                .iter()
                .filter(|e| now.duration_since(e.value().at) >= SLOT_TTL)
                .map(|e| *e.key())
                .collect();
            for room in stale {
                a.free_room(room);
            }
        })
        .ok();
}

/// Stage `bytes` (the serialized pixel buffer for one image) into a pre-registered
/// arena slot keyed by `room` (its bootstrap_room), and return the wire descriptor
/// for the puller: `[slot_addr u64 LE][port u16 LE][listener_ip utf8]`. The worker
/// fetches the gateway's (fixed) metadata once, connects to the listener (ip:port),
/// then READs `nbytes` (the proto field) from `slot_addr`. The slot is returned to
/// the free list on the worker's free-notif or the TTL. On any failure (no listener
/// IP, NIXL init, oversized image, or no free slot) returns `Err(bytes)` so the
/// caller re-attaches them as the inline payload (no behaviour change).
pub(crate) fn export_pixel_buffer(room: i32, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
    if gw_listen_ip().is_empty() {
        // No listener IP configured -> cannot do the cross-node metadata exchange.
        return Err(bytes);
    }
    let Some(arena) = arena() else {
        return Err(bytes);
    };
    let Some((slot, addr)) = arena.lease_and_write(&bytes) else {
        // Image too big for a slot, or the pool is momentarily exhausted.
        return Err(bytes);
    };
    arena.occupied.insert(
        room,
        OccSlot {
            slot,
            at: Instant::now(),
        },
    );

    let ip = gw_listen_ip().as_bytes();
    let port = gw_listen_port();
    let mut descriptor = Vec::with_capacity(10 + ip.len());
    descriptor.extend_from_slice(&addr.to_le_bytes());
    descriptor.extend_from_slice(&port.to_le_bytes());
    descriptor.extend_from_slice(ip);
    debug!(
        room,
        addr,
        slot,
        "EPD RDMA: staged pixel slot (listener {}:{})",
        gw_listen_ip(),
        port
    );
    Ok(descriptor)
}
