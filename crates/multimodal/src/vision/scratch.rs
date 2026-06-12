//! Thread-local recycling pools for the large per-image vision buffers.
//!
//! The preprocess pipeline allocates tens of MB per image (the [C, H, W] f32
//! tensor alone is ~25 MB at 1080p). Freshly-allocated buffers of this size
//! bypass the allocator's reuse paths (glibc caps non-main-arena chunks at
//! 64 MB and mmaps anything larger or colder), so every image pays tens of
//! thousands of minor page faults; the fault path serializes process-wide and
//! caps the data plane's effective parallelism. Recycling the buffers keeps
//! the pages mapped and hot.
//!
//! Pools are thread-local (the preprocess work runs on blocking-pool threads),
//! so take/give pairs are lock-free. A small per-thread cap bounds residency.

use std::cell::RefCell;

/// Max recycled buffers kept per thread per class; excess is dropped. The
/// vision path holds at most a couple of live tensors per request, so a small
/// cap captures the reuse while bounding per-thread RSS.
const MAX_POOLED: usize = 4;

thread_local! {
    static F32_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// Take a zero-filled `Vec<f32>` of exactly `len`, reusing pooled capacity.
pub(crate) fn take_f32(len: usize) -> Vec<f32> {
    let mut v = F32_POOL
        .with(|p| p.borrow_mut().pop())
        .unwrap_or_default();
    v.clear();
    v.resize(len, 0.0);
    v
}

/// Return a buffer to the pool for reuse by later `take_f32` calls.
pub(crate) fn give_f32(v: Vec<f32>) {
    if v.capacity() == 0 {
        return;
    }
    F32_POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < MAX_POOLED {
            p.push(v);
        }
    });
}
