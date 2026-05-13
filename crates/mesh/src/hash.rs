//! Stable 8-byte content hashes used to identify radix-tree nodes
//! on the wire.
//!
//! Both `hash_node_path` (string trees) and `hash_token_path`
//! (token trees) take the low 8 bytes of a Blake3 digest, with a
//! single carve-out for [`GLOBAL_EVICTION_HASH`]: a real prefix
//! that hashes to `0` is remapped to `1` so the sentinel space
//! stays disjoint from real entries.

/// Sentinel hash meaning "evict this tenant from ALL nodes" in
/// the tenant-eviction wire format. Real prefixes never hash to
/// this value: [`hash_node_path`] and [`hash_token_path`] remap
/// `0` to `1` if Blake3 lands there.
pub const GLOBAL_EVICTION_HASH: u64 = 0;

/// Compute a compact 8-byte hash of a prefix path. Returns a
/// non-zero hash; `0` is reserved for [`GLOBAL_EVICTION_HASH`].
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

/// Compute a compact 8-byte hash of a token-id sequence. Returns
/// a non-zero hash; `0` is reserved for [`GLOBAL_EVICTION_HASH`].
/// Streams into the hasher to avoid an intermediate `Vec<u8>`
/// (~128 KB at 32K tokens).
#[expect(
    clippy::unwrap_used,
    reason = "blake3 always returns 32 bytes; [..8] into [u8; 8] cannot fail"
)]
pub fn hash_token_path(tokens: &[u32]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for t in tokens {
        hasher.update(&t.to_le_bytes());
    }
    let hash = hasher.finalize();
    let h = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    if h == GLOBAL_EVICTION_HASH {
        1
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_node_path_never_returns_sentinel() {
        // A real prefix that would otherwise land on 0 must remap
        // to 1. We can't construct that input without grinding,
        // so we just assert the function never produces the
        // sentinel for representative inputs.
        for s in ["", "a", "hello world", "/api/v1/chat/completions"] {
            assert_ne!(hash_node_path(s), GLOBAL_EVICTION_HASH);
        }
    }

    #[test]
    fn hash_token_path_never_returns_sentinel() {
        for tokens in [&[][..], &[1][..], &[1, 2, 3][..], &[0; 32][..]] {
            assert_ne!(hash_token_path(tokens), GLOBAL_EVICTION_HASH);
        }
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash_node_path("a"), hash_node_path("a"));
        assert_eq!(hash_token_path(&[1, 2, 3]), hash_token_path(&[1, 2, 3]));
    }

    #[test]
    fn hash_token_path_matches_concat_then_hash() {
        // Lock in wire-compatibility with the prior allocating version.
        for tokens in [&[][..], &[42][..], &[1, 2, 3, 4][..], &[0u32; 64][..]] {
            let concat: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let expected_full = blake3::hash(&concat);
            let expected = u64::from_le_bytes(expected_full.as_bytes()[..8].try_into().unwrap());
            let expected = if expected == GLOBAL_EVICTION_HASH {
                1
            } else {
                expected
            };
            assert_eq!(hash_token_path(tokens), expected);
        }
    }
}
