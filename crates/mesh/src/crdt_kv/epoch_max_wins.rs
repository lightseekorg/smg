//! Epoch-aware max-wins merge for rate-limit counter values.
//!
//! Plain max-wins undoes window resets: A resets to 0, B still has
//! 100, max(0, 100) reverts the reset. This merge compares epoch
//! first, then max-count within the same epoch — a reset (higher
//! epoch, count = 0) always beats a higher count at an older epoch.
//!
//! Wire format: 16 bytes, `u64` big-endian epoch in bytes 0..8,
//! `i64` big-endian count in bytes 8..16. Fixed-size + big-endian so
//! the mesh crate can compare values without an application
//! callback. Signed count leaves room for future sentinels.
//!
//! Malformed input (length ≠ 16): if one side decodes, it wins. If
//! both fail, keep `local` per the `MergeStrategy::EpochMaxWins`
//! contract in `kv.rs` — a no-op on the store. This sacrifices
//! commutativity for the malformed/malformed case, but rate-limit
//! counters write on every increment and reset every window, so a
//! well-formed write restores clean state before the non-convergence
//! matters.

use std::cmp::Ordering;

/// Fixed wire size: 8-byte big-endian epoch + 8-byte big-endian count.
pub const EPOCH_MAX_WINS_ENCODED_LEN: usize = 16;

/// Parsed value returned owned so callers don't need to keep the
/// source slice alive across the merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochCount {
    pub epoch: u64,
    pub count: i64,
}

/// Encode `(epoch, count)` to the 16-byte big-endian wire format.
#[must_use]
pub fn encode(epoch: u64, count: i64) -> [u8; EPOCH_MAX_WINS_ENCODED_LEN] {
    let mut buf = [0u8; EPOCH_MAX_WINS_ENCODED_LEN];
    buf[0..8].copy_from_slice(&epoch.to_be_bytes());
    buf[8..16].copy_from_slice(&count.to_be_bytes());
    buf
}

/// Decode 16 bytes. `None` on any other length (caller treats as
/// malformed).
#[must_use]
pub fn decode(bytes: &[u8]) -> Option<EpochCount> {
    if bytes.len() != EPOCH_MAX_WINS_ENCODED_LEN {
        return None;
    }
    let epoch = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
    let count = i64::from_be_bytes(bytes[8..16].try_into().ok()?);
    Some(EpochCount { epoch, count })
}

/// Merge two rate-limit values per the epoch-max-wins rule.
///
/// Both decode: higher epoch wins; on equal epochs, max count wins.
/// One decodes: the well-formed side wins. Neither decodes: keep
/// `local` (no-op, per the `EpochMaxWins` contract in `kv.rs`).
///
/// Returned `Vec<u8>` so the caller can write it straight back.
#[must_use]
pub fn merge(local: &[u8], remote: &[u8]) -> Vec<u8> {
    match (decode(local), decode(remote)) {
        (Some(l), Some(r)) => {
            let winner = match l.epoch.cmp(&r.epoch) {
                Ordering::Greater => l,
                Ordering::Less => r,
                Ordering::Equal => EpochCount {
                    epoch: l.epoch,
                    count: l.count.max(r.count),
                },
            };
            encode(winner.epoch, winner.count).to_vec()
        }
        (Some(_), None) => local.to_vec(),
        (None, Some(_)) => remote.to_vec(),
        (None, None) => local.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        for (epoch, count) in [
            (0_u64, 0_i64),
            (1, 1),
            (5, 30),
            (u64::MAX, i64::MAX),
            (u64::MAX, i64::MIN),
            (42, -1),
        ] {
            let buf = encode(epoch, count);
            assert_eq!(buf.len(), EPOCH_MAX_WINS_ENCODED_LEN);
            let decoded = decode(&buf).expect("encoded buffer is 16 bytes");
            assert_eq!(decoded, EpochCount { epoch, count });
        }
    }

    #[test]
    fn decode_rejects_wrong_lengths() {
        assert_eq!(decode(&[]), None);
        assert_eq!(decode(&[0u8; 15]), None);
        assert_eq!(decode(&[0u8; 17]), None);
        // Just inside is fine, one byte off is not.
        assert!(decode(&[0u8; 16]).is_some());
    }

    #[test]
    fn same_epoch_max_count_wins() {
        // Normal counting within a window; highest observed count is
        // the cluster-wide truth. Also asserts commutativity.
        let local = encode(5, 30);
        let remote = encode(5, 42);
        let merged = merge(&local, &remote);
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 5,
                count: 42
            }
        );
        assert_eq!(merge(&remote, &local), merged);
    }

    #[test]
    fn higher_epoch_wins_even_with_lower_count() {
        // Reset must propagate: epoch 6 count 0 beats epoch 5 count 30.
        let merged = merge(&encode(5, 30), &encode(6, 0));
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 6, count: 0 });
    }

    #[test]
    fn lower_epoch_loses_to_local_newer_window() {
        // Stale remote from old window is dropped; local window-6
        // state survives.
        let merged = merge(&encode(6, 10), &encode(5, 100));
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 6,
                count: 10
            }
        );
    }

    #[test]
    fn near_simultaneous_reset_both_at_zero() {
        // Both sides at epoch 5 count 0. max(0, 0) = 0.
        let merged = merge(&encode(5, 0), &encode(5, 0));
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 5, count: 0 });
    }

    #[test]
    fn malformed_remote_keeps_local() {
        // Corrupt remote must not overwrite healthy local.
        let local = encode(5, 30);
        let merged = merge(&local, &[0xFFu8; 15]);
        assert_eq!(merged, local.to_vec());
    }

    #[test]
    fn malformed_local_is_replaced_by_remote() {
        // Healthy remote recovers a corrupt local.
        let remote = encode(5, 30);
        let merged = merge(&[], &remote);
        assert_eq!(merged, remote.to_vec());
    }

    #[test]
    fn both_malformed_keeps_local_no_panic() {
        // Per EpochMaxWins contract, both-malformed is a no-op that
        // keeps local. Non-commutative by design — see module docs.
        let corrupt_local = vec![1u8, 2, 3];
        let merged = merge(&corrupt_local, &[0xFFu8; 17]);
        assert_eq!(merged, corrupt_local);
    }

    #[test]
    fn signed_count_preserves_sign() {
        // Negative counts round-trip; the merge must not silently
        // reinterpret as unsigned.
        let merged = merge(&encode(5, -10), &encode(5, -5));
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 5,
                count: -5
            }
        );
    }

    #[test]
    fn merge_is_idempotent() {
        // merge(v, v) == v — gossip re-delivery must not drift.
        let value = encode(42, 7);
        assert_eq!(merge(&value, &value), value.to_vec());
    }

    #[test]
    fn merge_is_associative_on_three_values() {
        // ((a ⊕ b) ⊕ c) == (a ⊕ (b ⊕ c)). Required for eventual
        // consistency under reordering.
        let a = encode(5, 10);
        let b = encode(6, 3);
        let c = encode(6, 9);
        let ab_then_c = merge(&merge(&a, &b), &c);
        let a_then_bc = merge(&a, &merge(&b, &c));
        assert_eq!(ab_then_c, a_then_bc);
        assert_eq!(
            decode(&ab_then_c).unwrap(),
            EpochCount { epoch: 6, count: 9 }
        );
    }
}
