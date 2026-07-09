//! EPD RDMA pixel transport: the gateway exports each image's serialized pixel
//! buffer over NIXL (UCX/RoCE) so the encode worker PULLs it (one-sided READ),
//! instead of shipping the large pixel buffer inline in the Encode gRPC frame.
//! On the inline path, serializing it burns gateway CPU and starves the encode
//! worker's GPU -- the EPD bottleneck this transport removes.
//!
//! Gated behind `SMG_MM_PIXEL_RDMA`; every NIXL call is reached only when the gate
//! is on (a missing `libnixl_capi.so` aborts on the first stub call, not a no-op),
//! and any export failure falls back to the inline payload so EPD never hard-fails.
//!
//! v2 (pre-registered pool): one host-DRAM arena is registered with NIXL ONCE at
//! init and sub-divided into fixed slots. Per image the hot path only LEASES a free
//! slot, frames the pixels as `[gen u64][pixels][gen u64]`, and ships
//! `[magic 8B][slot_addr u64 LE][gen u64 LE][room i64 LE][port u16 LE][ip utf8]`
//! -- NO per-image register_memory and NO growing agent metadata. The worker fetches
//! the gateway's (now fixed) metadata ONCE, then READs slot offsets into its own
//! pre-registered landing pool. The transfer notif is tagged with the bootstrap_room;
//! this reaper consumes it to return the slot to the free list (a TTL sweep is the
//! lost-notif net). This removes the v1 per-image control+registration overhead that
//! made the RDMA path slower than inline.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NotificationMap, OptArgs,
    RegistrationHandle,
};
use parking_lot::Mutex;
use tracing::{debug, error};

use crate::routers::grpc::multimodal::mm_default_transport_is_rdma;

/// Whether the RDMA pixel lane is active. Selected by the first-class
/// `TransportMode::Rdma` (`--multimodal-tensor-transport rdma` /
/// `SMG_MM_TENSOR_TRANSPORT=rdma`), with the legacy `SMG_MM_PIXEL_RDMA` env as a
/// backward-compatible fallback. Cached in a process-wide `OnceLock`.
pub(crate) fn rdma_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        mm_default_transport_is_rdma()
            || matches!(
                std::env::var("SMG_MM_PIXEL_RDMA").as_deref(),
                Ok("1") | Ok("true")
            )
    })
}

/// Reaper poll interval.
const REAPER_TICK: Duration = Duration::from_millis(20);
/// Default arena geometry: 64 slots x 32 MiB = 2 GiB host DRAM. A slot must hold one
/// image's bf16 pixel buffer; raise SLOT_BYTES if larger images overflow it.
const DEFAULT_POOL_SLOTS: usize = 64;
const DEFAULT_SLOT_BYTES: usize = 32 * 1024 * 1024;

/// Per-lease generation framing. Each leased slot is written as
/// `[gen u64 LE][payload][gen u64 LE]` and the descriptor carries the same `gen`, so
/// the worker can detect a slot it READ after the gateway recycled and reused it (the
/// residual the TTL widening still leaves if the TTL is ever misconfigured / the two
/// sides' env drift): a recycled slot carries a strictly newer gen than the worker's
/// descriptor, and a READ torn by a concurrent reuse sees header != trailer. Both are
/// caught by requiring `header == trailer == descriptor.gen`, which makes correctness
/// independent of the TTL value (TTL stays purely a capacity knob). Worker-side parse
/// mirrors this in grpc_servicer encoder_servicer.py (`GEN`/`FRAME`).
const GEN_BYTES: usize = 8;
/// Header + trailer generation stamps bracketing the payload.
const FRAME_OVERHEAD: usize = 2 * GEN_BYTES;
/// Descriptor prefix for the version that carries `bootstrap_room` inline.
const DESCRIPTOR_MAGIC: &[u8; 8] = b"SMGRDMA1";

/// Read a seconds-valued env knob, falling back to `default`.
fn env_secs(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The worst-case wall time the encode worker may hold a shipped descriptor before
/// and during its one-sided READ: it waits up to `SMG_RDMA_LANDING_WAIT_S` for a free
/// landing slot, then READs for up to `SMG_RDMA_READ_TIMEOUT_S`. These mirror the
/// encode servicer's own knobs (same env names, same defaults) so the two sides
/// cannot drift. The gateway must not reclaim a slot inside this window.
fn worker_max_hold() -> Duration {
    Duration::from_secs(
        env_secs("SMG_RDMA_LANDING_WAIT_S", 120) + env_secs("SMG_RDMA_READ_TIMEOUT_S", 60),
    )
}

/// Fixed slack added to the derived `worker_max_hold()` when deriving the slot
/// TTL. A const rather than an env knob: it would only be a second tuning layer on
/// an already-derived value, it only ever widens the lost-notif leak window (a
/// capacity nit, never correctness -- the per-lease gen-framing in `GEN_BYTES` makes
/// a recycled-under-read slot detectable independent of the TTL), and 30s dwarfs any
/// Encode-RPC delivery jitter. `SMG_RDMA_SLOT_TTL_S` remains the explicit full-TTL
/// override.
const SLOT_TTL_SLACK: Duration = Duration::from_secs(30);

/// How long a leased slot may live without a free-notif before the reaper
/// force-reclaims it (lost notif / dead worker).
///
/// CORRECTNESS: this MUST exceed `worker_max_hold()` (plus slack for the Encode-RPC
/// delivery the gateway can't see), otherwise the TTL races the worker's still-valid
/// READ: the reaper returns the slot, the next image re-leases the SAME address, and
/// the late READ silently returns the WRONG image's pixels (no error, no NaN). Derived
/// by default (= worker_max_hold + the fixed `SLOT_TTL_SLACK`); `SMG_RDMA_SLOT_TTL_S`
/// overrides explicitly. The cost of the wider TTL is only that a genuinely lost notif
/// leaks one slot for longer -- a capacity nit, never a correctness one.
fn slot_ttl() -> Duration {
    static TTL: OnceLock<Duration> = OnceLock::new();
    *TTL.get_or_init(|| {
        if let Ok(v) = std::env::var("SMG_RDMA_SLOT_TTL_S") {
            if let Ok(secs) = v.parse::<u64>() {
                return Duration::from_secs(secs);
            }
        }
        worker_max_hold() + SLOT_TTL_SLACK
    })
}

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
    #[expect(
        unsafe_code,
        reason = "NIXL MemoryRegion trait method is unsafe; body only reinterprets the stored base address as a pointer"
    )]
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

/// Pure slot bookkeeping over the arena's raw memory: the free-list, the leased
/// (room -> slot) map, and the memcpy-into-slot. Deliberately holds NO NIXL handle so
/// the lease / reclaim / reuse policy (where the TTL-vs-READ race lives) is unit-
/// testable without a registered agent or any RDMA hardware.
struct SlotPool {
    /// Raw base address of the leaked arena allocation (usize => Send+Sync).
    base: usize,
    slot_bytes: usize,
    n_slots: usize,
    /// Available slot indices.
    free: Mutex<Vec<u32>>,
    /// Leased slots keyed by bootstrap_room (free-on-notif + TTL reclaim).
    occupied: DashMap<i64, OccSlot>,
    /// Monotonic per-lease generation stamp (see GEN_BYTES). Starts at 1 so the
    /// zero-initialized arena (a never-written slot reads gen 0) can never match a
    /// real descriptor's gen.
    gen: AtomicU64,
}

impl SlotPool {
    fn new(base: usize, slot_bytes: usize, n_slots: usize) -> SlotPool {
        SlotPool {
            base,
            slot_bytes,
            n_slots,
            free: Mutex::new((0..n_slots as u32).collect()),
            occupied: DashMap::new(),
            gen: AtomicU64::new(1),
        }
    }

    /// Lease a free slot for `room`, frame `bytes` as `[gen][payload][gen]` into it, and
    /// record the lease at `now`. Returns `(slot, slot_addr, gen)`, or `None` if the
    /// framed image exceeds `slot_bytes` or no slot is free (caller -> inline). Recording
    /// the lease here (rather than in the caller) keeps the slot and its TTL timestamp
    /// atomic w.r.t. the write. `gen` is shipped in the descriptor and re-checked by the
    /// worker against the slot's stamps to reject a recycled/torn READ.
    fn lease_and_write(&self, room: i64, bytes: &[u8], now: Instant) -> Option<(u32, u64, u64)> {
        // Reserve room for the generation stamps bracketing the payload.
        if bytes.len() + FRAME_OVERHEAD > self.slot_bytes {
            return None;
        }
        let slot = self.free.lock().pop()?;
        let addr = self.base + slot as usize * self.slot_bytes;
        // A fresh, monotonically increasing stamp for THIS lease; a later lease of the
        // same slot gets a strictly larger gen. Relaxed is fine: we only need
        // uniqueness, and the cross-process happens-before is the descriptor ship.
        let gen = self.gen.fetch_add(1, Ordering::Relaxed);
        let gen_le = gen.to_le_bytes();
        // SAFETY: `slot` is exclusively leased (popped from `free`, returned only via
        // free_room); [addr, addr + FRAME_OVERHEAD + len) is within the arena
        // (len + FRAME_OVERHEAD <= slot_bytes) and disjoint from every other leased
        // slot; the worker reads it only after we ship the descriptor below, and the
        // TTL keeps it leased past the worker's max READ window. The arena is raw,
        // leaked memory (no aliasing typed reference). So there is no data race and no
        // overlap. Write the header gen FIRST and the trailer gen LAST so a racing
        // reuse is visible at one stamp or the other (seqlock framing).
        #[expect(
            unsafe_code,
            reason = "seqlock gen-stamp write into the pre-registered NIXL slot arena"
        )]
        unsafe {
            let p = addr as *mut u8;
            std::ptr::copy_nonoverlapping(gen_le.as_ptr(), p, GEN_BYTES);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(GEN_BYTES), bytes.len());
            std::ptr::copy_nonoverlapping(
                gen_le.as_ptr(),
                p.add(GEN_BYTES + bytes.len()),
                GEN_BYTES,
            );
        }
        self.occupied.insert(room, OccSlot { slot, at: now });
        Some((slot, addr as u64, gen))
    }

    /// Return the slot leased for `room` to the free list (idempotent).
    fn free_room(&self, room: i64) {
        if let Some((_room, occ)) = self.occupied.remove(&room) {
            self.free.lock().push(occ.slot);
        }
    }

    /// Reclaim every slot whose lease is older than `ttl` as of `now` (the lost-notif
    /// net). Returns the count reclaimed. `now`/`ttl` are parameters so the reclaim
    /// policy is testable without sleeping.
    fn reap_stale(&self, now: Instant, ttl: Duration) -> usize {
        let stale: Vec<i64> = self
            .occupied
            .iter()
            .filter(|e| now.duration_since(e.value().at) >= ttl)
            .map(|e| *e.key())
            .collect();
        let n = stale.len();
        for room in stale {
            self.free_room(room);
        }
        n
    }
}

/// Pre-registered host-DRAM arena: a `SlotPool` plus the single persistent NIXL
/// registration that keeps the backing memory pinned for the worker's one-sided READ.
/// One registration (`_handle`) covers the whole arena; each image's pixels are
/// memcpy'd into a leased slot. A slot is exclusively owned between lease and free,
/// and the write happens-before the descriptor ship which happens-before the worker's
/// READ, so no slot is ever written and read concurrently. The hot path touches no
/// NIXL agent state (only the free-list lock + a lockless memcpy into owned memory).
struct SlotArena {
    pool: SlotPool,
    /// The single persistent registration; kept alive for the arena's life.
    _handle: RegistrationHandle,
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
                        slots = a.pool.n_slots,
                        slot_bytes = a.pool.slot_bytes,
                        ttl_s = slot_ttl().as_secs(),
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
        pool: SlotPool::new(base, slot_bytes, n_slots),
        _handle: handle,
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
                            if let Ok(room) = tag.parse::<i64>() {
                                a.pool.free_room(room);
                            }
                        }
                    }
                }
            }
            // TTL sweep: reclaim slots whose READ-notif never arrived. `slot_ttl()` is
            // derived to exceed the worker's max hold, so this never races a live READ.
            let _ = a.pool.reap_stale(Instant::now(), slot_ttl());
        })
        .ok();
}

/// Stage `bytes` (the serialized pixel buffer for one image) into a pre-registered
/// arena slot keyed by `room`, and return the wire descriptor for the puller:
/// `[magic 8B][slot_addr u64 LE][gen u64 LE][room i64 LE][port u16 LE]`
/// `[listener_ip utf8]`. The worker fetches the gateway's (fixed) metadata once,
/// connects to the listener (ip:port), then READs `nbytes + FRAME_OVERHEAD` from
/// `slot_addr` and re-checks the stamps against `gen`. The slot is returned to
/// the free list on the worker's free-notif or the TTL. On any failure (no listener
/// IP, NIXL init, oversized image, or no free slot) returns `Err(bytes)` so the
/// caller re-attaches them as the inline payload (no behaviour change).
pub(crate) fn export_pixel_buffer(room: i64, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
    if gw_listen_ip().is_empty() {
        // No listener IP configured -> cannot do the cross-node metadata exchange.
        return Err(bytes);
    }
    let Some(arena) = arena() else {
        return Err(bytes);
    };
    let Some((slot, addr, gen)) = arena.pool.lease_and_write(room, &bytes, Instant::now()) else {
        // Image too big for a slot, or the pool is momentarily exhausted.
        return Err(bytes);
    };

    let ip = gw_listen_ip().as_bytes();
    let port = gw_listen_port();
    let mut descriptor =
        Vec::with_capacity(DESCRIPTOR_MAGIC.len() + 8 + GEN_BYTES + 8 + 2 + ip.len());
    descriptor.extend_from_slice(DESCRIPTOR_MAGIC);
    descriptor.extend_from_slice(&addr.to_le_bytes());
    descriptor.extend_from_slice(&gen.to_le_bytes());
    descriptor.extend_from_slice(&room.to_le_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `SlotPool` over a real (leaked) heap arena -- no NIXL agent, no RDMA.
    /// Leaking matches production (the arena is process-lifetime) and keeps the raw
    /// `base` valid for the whole test.
    fn test_pool(n_slots: usize, slot_bytes: usize) -> SlotPool {
        let buf = vec![0u8; n_slots * slot_bytes].into_boxed_slice();
        let base = Box::into_raw(buf) as *mut u8 as usize;
        SlotPool::new(base, slot_bytes, n_slots)
    }

    /// Read back what physically sits at a slot address (what a one-sided READ to that
    /// address would return).
    fn bytes_at(addr: u64, len: usize) -> Vec<u8> {
        #[expect(
            unsafe_code,
            reason = "read bytes back from a NIXL slot address for gen verification"
        )]
        unsafe {
            std::slice::from_raw_parts(addr as *const u8, len).to_vec()
        }
    }

    /// Reproduces the bug: a too-short TTL reclaims a slot while the worker is still
    /// within its hold window, the next image re-leases the SAME address, and a late
    /// READ against the old descriptor returns the WRONG image's bytes.
    #[test]
    fn short_ttl_recycles_slot_under_a_live_descriptor() {
        let pool = test_pool(1, 64);
        let t0 = Instant::now();
        let img_a = b"AAAAAAAAAAAAAAAA";
        let (_slot_a, addr_a, _gen_a) = pool.lease_and_write(1, img_a, t0).expect("lease A");

        // Worker has NOT issued its READ yet (e.g. parked on landing-ring backpressure).
        // The reaper runs with a 30s TTL after 31s -> reclaims the still-referenced slot.
        let freed = pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
        assert_eq!(
            freed, 1,
            "30s TTL reclaims a slot still under a live descriptor"
        );

        // The next image re-leases the freed slot -> same physical address.
        let img_b = b"BBBBBBBBBBBBBBBB";
        let (_slot_b, addr_b, _gen_b) = pool
            .lease_and_write(2, img_b, t0 + Duration::from_secs(31))
            .expect("lease B");
        assert_eq!(addr_a, addr_b, "freed slot's address is reused");

        // The late READ for room 1 reads addr_a, whose payload (at +GEN_BYTES) now holds
        // room 2's pixels. The TTL no longer protects this; the gen stamps do (next test).
        assert_eq!(
            bytes_at(addr_a + GEN_BYTES as u64, img_b.len()),
            img_b,
            "stale READ returns the WRONG image's payload (physical cross-wire remains)"
        );
    }

    /// The gen guard: a recycled slot carries a fresh gen, so a worker validating the
    /// slot's header/trailer stamps against its (older) descriptor gen detects the
    /// cross-wire and fails the room -- independent of the TTL value.
    #[test]
    fn gen_guard_detects_recycled_slot() {
        let pool = test_pool(1, 64);
        let t0 = Instant::now();
        let img_a = b"AAAAAAAA";
        let (_a, addr_a, gen_a) = pool.lease_and_write(1, img_a, t0).expect("lease A");

        // Recycle under a too-short TTL (the residual the gen guard backstops).
        pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
        let img_b = b"BBBBBBBB";
        let (_b, addr_b, gen_b) = pool
            .lease_and_write(2, img_b, t0 + Duration::from_secs(31))
            .expect("lease B");
        assert_eq!(addr_a, addr_b, "slot address is physically reused");
        assert_ne!(gen_a, gen_b, "each lease gets a fresh, larger generation");

        // The slot's header+trailer stamps now read gen_b. A worker holding room 1's
        // descriptor (gen_a) compares against these and sees a mismatch -> reject.
        let hdr = u64::from_le_bytes(bytes_at(addr_a, GEN_BYTES).try_into().unwrap());
        let trl = u64::from_le_bytes(
            bytes_at(addr_a + (GEN_BYTES + img_b.len()) as u64, GEN_BYTES)
                .try_into()
                .unwrap(),
        );
        assert_eq!(hdr, gen_b);
        assert_eq!(trl, gen_b);
        assert_ne!(
            hdr, gen_a,
            "stamp != descriptor gen -> worker fails the room"
        );
    }

    /// A normal lease frames the payload as `[gen][payload][gen]` with both stamps equal
    /// to the returned (and shipped-in-descriptor) gen.
    #[test]
    fn gen_framing_brackets_the_payload() {
        let pool = test_pool(1, 64);
        let t0 = Instant::now();
        let (_s, addr, gen) = pool.lease_and_write(7, b"PIXELS!!", t0).expect("lease");
        assert_eq!(
            u64::from_le_bytes(bytes_at(addr, GEN_BYTES).try_into().unwrap()),
            gen,
            "header stamp == gen"
        );
        assert_eq!(bytes_at(addr + GEN_BYTES as u64, 8), b"PIXELS!!", "payload");
        assert_eq!(
            u64::from_le_bytes(
                bytes_at(addr + (GEN_BYTES + 8) as u64, GEN_BYTES)
                    .try_into()
                    .unwrap()
            ),
            gen,
            "trailer stamp == gen"
        );
    }

    /// The fix invariant: the derived TTL strictly exceeds the worker's max hold, so
    /// the reaper can never reclaim a slot the worker could still be reading. Fails
    /// under the old flat `const SLOT_TTL = 30s`.
    #[test]
    fn slot_ttl_exceeds_worker_max_hold() {
        assert!(
            slot_ttl() > worker_max_hold(),
            "slot_ttl {:?} must exceed worker_max_hold {:?} or a late READ cross-wires",
            slot_ttl(),
            worker_max_hold()
        );
    }

    /// With the derived TTL, a reap at the worker's max hold does NOT reclaim, so the
    /// address cannot be reused under a live descriptor: a second image finds the pool
    /// full and the gateway falls back to inline (Err), never aliasing the slot.
    #[test]
    fn derived_ttl_keeps_slot_through_worker_hold() {
        let pool = test_pool(1, 64);
        let t0 = Instant::now();
        let (_slot_a, addr_a, _gen_a) = pool
            .lease_and_write(1, b"AAAAAAAAAAAAAAAA", t0)
            .expect("lease A");

        let freed = pool.reap_stale(t0 + worker_max_hold(), slot_ttl());
        assert_eq!(
            freed, 0,
            "derived TTL must not reclaim within the worker hold"
        );

        // Slot still leased -> no free slot -> caller goes inline, no address reuse.
        assert!(
            pool.lease_and_write(2, b"BBBBBBBBBBBBBBBB", t0 + worker_max_hold())
                .is_none(),
            "leased slot is not handed to a second image"
        );
        assert_eq!(
            bytes_at(addr_a + GEN_BYTES as u64, 16),
            b"AAAAAAAAAAAAAAAA",
            "the slot still holds room 1's pixels (payload at +GEN_BYTES) for its READ"
        );
    }

    /// The recycle is not a one-slot artifact: with two slots, both reclaim under a
    /// short TTL and a later lease reuses one of the freed addresses.
    #[test]
    fn multi_slot_addresses_recycle_after_reclaim() {
        let pool = test_pool(2, 64);
        let t0 = Instant::now();
        let (_a, addr_a, _ga) = pool.lease_and_write(1, b"A", t0).unwrap();
        let (_b, addr_b, _gb) = pool.lease_and_write(2, b"B", t0).unwrap();
        assert_ne!(addr_a, addr_b, "distinct leases get distinct addresses");

        let freed = pool.reap_stale(t0 + Duration::from_secs(31), Duration::from_secs(30));
        assert_eq!(freed, 2);

        let (_c, addr_c, _gc) = pool
            .lease_and_write(3, b"C", t0 + Duration::from_secs(31))
            .unwrap();
        assert!(
            addr_c == addr_a || addr_c == addr_b,
            "a reclaimed address is reused by the next lease"
        );
    }

    /// An oversized image never leases (caller -> inline), and a normal free returns
    /// the slot for reuse.
    #[test]
    fn oversize_rejected_and_free_returns_slot() {
        // Slot holds FRAME_OVERHEAD (16) of gen stamps + payload, so 24 bytes => 8 of
        // usable payload.
        let pool = test_pool(1, FRAME_OVERHEAD + 8);
        let t0 = Instant::now();
        assert!(
            pool.lease_and_write(1, &[0u8; 9], t0).is_none(),
            "payload + gen frame larger than a slot is rejected"
        );
        let (_s, _addr, _g) = pool
            .lease_and_write(1, &[1u8; 8], t0)
            .expect("fits exactly");
        assert!(
            pool.lease_and_write(2, &[2u8; 8], t0).is_none(),
            "pool now full"
        );
        pool.free_room(1);
        assert!(
            pool.lease_and_write(2, &[2u8; 8], t0).is_some(),
            "freed slot is reusable"
        );
    }
}
