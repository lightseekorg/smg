//! Epoch-aware max-wins merge for rate-limit counter values.
//!
//! Rate-limit counters can't merge with plain max-wins because
//! window resets would be silently undone: one node resets its
//! counter to 0 while another still carries the pre-reset count of
//! 100 — plain max(0, 100) = 100 reverts the reset cluster-wide.
//! The merge here compares an explicit epoch (window number) first
//! and only falls back to max-count within the same epoch, so a
//! reset (higher epoch, count = 0) always beats a higher count at an
//! older epoch.
//!
//! Wire format is a fixed 16-byte payload chosen so the mesh crate
//! can interpret and compare values without an application callback:
//!
//! ```text
//! bytes 0..8  : epoch  (u64, big-endian)
//! bytes 8..16 : count  (i64, big-endian)
//! ```
//!
//! Big-endian is the single public encoding so both sides of the
//! gossip stream agree byte-for-byte; a floating mix of host
//! endiannesses would corrupt the compare. `count` is signed because
//! the caller may choose to encode deltas or reserved sentinels
//! (e.g. eviction markers) in the same slot; the merge itself only
//! uses max-compare on the value.
//!
//! Malformed input (any length ≠ 16) is kept-if-well-formed-else-
//! local: a single corrupt gossip message must never crash the merge
//! loop or retroactively erase a healthy value. If both sides are
//! malformed, the local copy survives — there's nothing better the
//! merger can do locally, and a later healthy message from either
//! side resolves the state.

// All items below are intentionally unused in non-test compilation
// of this PR: the RateLimitSyncAdapter in the follow-up PR registers
// them against the `rl:*` prefix. Isolating the merge function first
// lets it be reviewed without the wider adapter change. Gated on
// `not(test)` so the attribute only applies where dead_code actually
// fires (tests do exercise every item below).
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by RateLimitSyncAdapter in a follow-up PR"
    )
)]

use std::cmp::Ordering;

/// Fixed wire size: 8 bytes big-endian epoch + 8 bytes big-endian count.
pub const EPOCH_MAX_WINS_ENCODED_LEN: usize = 16;

/// Parsed value from the 16-byte wire format. Returned as an owned
/// pair rather than a struct to keep this module free of application
/// types — adapters layer their own newtype on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochCount {
    pub epoch: u64,
    pub count: i64,
}

/// Encode `(epoch, count)` into the fixed 16-byte big-endian
/// representation used on the wire.
#[must_use]
pub fn encode(epoch: u64, count: i64) -> [u8; EPOCH_MAX_WINS_ENCODED_LEN] {
    let mut buf = [0u8; EPOCH_MAX_WINS_ENCODED_LEN];
    buf[0..8].copy_from_slice(&epoch.to_be_bytes());
    buf[8..16].copy_from_slice(&count.to_be_bytes());
    buf
}

/// Decode 16 bytes of big-endian `(epoch, count)`. Returns `None` if
/// the slice is not exactly 16 bytes — callers must treat that as
/// "malformed" and defer to the merge's keep-well-formed behaviour.
#[must_use]
pub fn decode(bytes: &[u8]) -> Option<EpochCount> {
    if bytes.len() != EPOCH_MAX_WINS_ENCODED_LEN {
        return None;
    }
    let epoch = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
    let count = i64::from_be_bytes(bytes[8..16].try_into().ok()?);
    Some(EpochCount { epoch, count })
}

/// Merge two rate-limit values using the epoch-max-wins rule.
///
/// - If only one side decodes to a valid 16-byte value, that one
///   wins (the other was corrupt / truncated; keeping a garbage
///   payload would propagate corruption to every future merge).
/// - If both sides decode, compare epochs first; higher epoch wins
///   outright. On equal epochs, `max(local.count, remote.count)`
///   wins — within a single window, the highest observed count is
///   the authoritative one.
/// - If both sides fail to decode, return `local` unchanged; at
///   worst this preserves whatever the node already had so traffic
///   keeps flowing until a healthy message arrives.
///
/// Returns an owned `Vec<u8>` rather than borrowing because the
/// caller usually needs to write the result back into storage and
/// re-emit it on the next gossip round.
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
        // Both nodes counting within window 5. Higher count is the
        // cluster-wide truth; the other side has simply not yet seen
        // recent requests.
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

        // Symmetric: same call with sides swapped yields the same
        // winner — merge is commutative.
        let merged_rev = merge(&remote, &local);
        assert_eq!(merged_rev, merged);
    }

    #[test]
    fn higher_epoch_wins_even_with_lower_count() {
        // Node A is partway through window 5 with 30 requests. Node
        // B has reset to window 6 with 0. B's reset must propagate
        // even though its count is smaller — otherwise resets silently
        // unwind.
        let local = encode(5, 30);
        let remote = encode(6, 0);
        let merged = merge(&local, &remote);
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 6, count: 0 });
    }

    #[test]
    fn lower_epoch_loses_to_local_newer_window() {
        // Local already advanced to window 6, remote gossip arrives
        // from the old window 5. The remote is stale — keep local.
        let local = encode(6, 10);
        let remote = encode(5, 100);
        let merged = merge(&local, &remote);
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
        // Both nodes entered window 5 at roughly the same moment;
        // neither has counted yet. max(0, 0) = 0.
        let local = encode(5, 0);
        let remote = encode(5, 0);
        let merged = merge(&local, &remote);
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 5, count: 0 });
    }

    #[test]
    fn malformed_remote_keeps_local() {
        // Remote carrier was truncated (e.g. protocol downgrade / MTU
        // issue). Local well-formed value must not be replaced with
        // garbage — doing so would poison every subsequent merge.
        let local = encode(5, 30);
        let corrupt_remote = &[0xFFu8; 15];
        let merged = merge(&local, corrupt_remote);
        assert_eq!(merged, local.to_vec());
    }

    #[test]
    fn malformed_local_is_replaced_by_remote() {
        // Whatever persisted locally was corrupt (crash mid-write,
        // partial disk page, etc). A well-formed remote message is a
        // chance to recover cleanly.
        let corrupt_local = vec![];
        let remote = encode(5, 30);
        let merged = merge(&corrupt_local, &remote);
        assert_eq!(merged, remote.to_vec());
    }

    #[test]
    fn both_malformed_returns_local_no_panic() {
        // Neither side has a value we can trust. Keep local so the
        // cluster doesn't churn on garbage; a later healthy message
        // from either side will repair the state.
        let corrupt_local = vec![1u8, 2, 3];
        let corrupt_remote = &[0xFFu8; 17];
        let merged = merge(&corrupt_local, corrupt_remote);
        assert_eq!(merged, corrupt_local);
    }

    #[test]
    fn signed_count_preserves_sign() {
        // The count is an i64 on the wire. Negative values are valid
        // inputs as far as this merge is concerned (the spec reserves
        // signed semantics for future use; this merge must not silently
        // reinterpret the bit pattern as unsigned).
        let local = encode(5, -10);
        let remote = encode(5, -5);
        // -5 > -10, so -5 wins on max-count within the same epoch.
        let merged = merge(&local, &remote);
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
        // Merging a value with itself is a no-op. Gossip protocols
        // re-deliver the same payload frequently; the merge must not
        // drift under repeated self-merges.
        let value = encode(42, 7);
        let merged = merge(&value, &value);
        assert_eq!(merged, value.to_vec());
    }

    #[test]
    fn merge_is_associative_on_three_values() {
        // ((a ⊕ b) ⊕ c) == (a ⊕ (b ⊕ c)). Required for eventual
        // consistency: peers observing the same set of updates in
        // different orders must converge to the same value.
        let a = encode(5, 10);
        let b = encode(6, 3); // higher epoch
        let c = encode(6, 9); // same epoch as b, higher count

        let ab_then_c = merge(&merge(&a, &b), &c);
        let a_then_bc = merge(&a, &merge(&b, &c));
        assert_eq!(ab_then_c, a_then_bc);
        // The fixed-point: epoch 6 with max count 9.
        assert_eq!(
            decode(&ab_then_c).unwrap(),
            EpochCount { epoch: 6, count: 9 }
        );
    }
}
