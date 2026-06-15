//! Recycling pools for the large per-image vision buffers.
//!
//! The preprocess pipeline allocates tens of MB per image (the [C, H, W] f32
//! tensor alone is ~25 MB at 1080p; the batched patch buffer ~10 MB/image).
//! Freshly-allocated buffers of this size bypass the allocator's reuse paths
//! (glibc caps non-main-arena chunks at 64 MB and mmaps anything larger or
//! colder), so every image pays tens of thousands of minor page faults; the
//! fault path serializes process-wide and caps the data plane's effective
//! parallelism. Recycling keeps the pages mapped and hot.
//!
//! Two tiers: a lock-free thread-local pool serves same-thread take/give
//! (preprocess internals on blocking-pool threads), and a global overflow
//! pool bridges cross-thread lifecycles (buffers taken on a blocking thread
//! and recycled after serialization on an async worker). Both tiers are
//! capped to bound residency.

use std::cell::RefCell;
use std::sync::Mutex;

/// Max recycled buffers kept per thread per class; excess spills to the
/// global pool. The vision path holds at most a couple of live tensors per
/// request, so a small cap captures same-thread reuse.
const MAX_THREAD_POOLED: usize = 2;
/// Max recycled buffers in the global overflow pool per class. Sized for
/// ~tens of concurrent in-flight requests' worth of large buffers.
const MAX_GLOBAL_POOLED: usize = 64;

thread_local! {
    static F32_LOCAL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}
static F32_GLOBAL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

macro_rules! pool_impl {
    ($take_cap:ident, $give:ident, $ty:ty, $local:ident, $global:ident) => {
        /// Take an empty `Vec` with at least `cap` capacity, reusing pooled storage.
        pub fn $take_cap(cap: usize) -> Vec<$ty> {
            let mut v = $local
                .with(|p| p.borrow_mut().pop())
                .or_else(|| $global.lock().ok().and_then(|mut g| g.pop()))
                .unwrap_or_default();
            v.clear();
            v.reserve(cap);
            v
        }

        /// Return a buffer for reuse by later takes (possibly on other threads).
        pub fn $give(v: Vec<$ty>) {
            if v.capacity() == 0 {
                return;
            }
            let spill = $local.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < MAX_THREAD_POOLED {
                    p.push(v);
                    None
                } else {
                    Some(v)
                }
            });
            if let Some(v) = spill {
                if let Ok(mut g) = $global.lock() {
                    if g.len() < MAX_GLOBAL_POOLED {
                        g.push(v);
                    }
                }
            }
        }
    };
}

pool_impl!(take_f32_cap, give_f32, f32, F32_LOCAL, F32_GLOBAL);

/// Take a zero-filled `Vec<f32>` of exactly `len`, reusing pooled storage.
pub fn take_f32(len: usize) -> Vec<f32> {
    let mut v = take_f32_cap(len);
    v.resize(len, 0.0);
    v
}
