/// Sentinel value reserved by tree-delta payloads for whole-worker eviction.
pub const GLOBAL_EVICTION_HASH: u64 = 0;

/// Compute a compact 8-byte hash of a text prefix path.
///
/// Returns a non-zero hash; 0 is reserved for [`GLOBAL_EVICTION_HASH`].
#[expect(
    clippy::unwrap_used,
    reason = "blake3 always returns 32 bytes; [..8] into [u8; 8] cannot fail"
)]
pub fn hash_node_path(path: &str) -> u64 {
    let hash = blake3::hash(path.as_bytes());
    let h = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    if h == GLOBAL_EVICTION_HASH {
        1
    } else {
        h
    }
}

/// Compute a compact 8-byte hash from token IDs.
///
/// Returns a non-zero hash; 0 is reserved for [`GLOBAL_EVICTION_HASH`].
#[expect(
    clippy::unwrap_used,
    reason = "blake3 always returns 32 bytes; [..8] into [u8; 8] cannot fail"
)]
pub fn hash_token_path(tokens: &[u32]) -> u64 {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let hash = blake3::hash(&bytes);
    let h = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    if h == GLOBAL_EVICTION_HASH {
        1
    } else {
        h
    }
}
