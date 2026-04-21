//! Handoff transport between the central round collector and
//! per-peer SyncStream sender tasks for stream-mode traffic.
//!
//! Broadcast drain entries ride a retention ring so an active sender
//! still observes past rounds up to the ring depth even if its task
//! tick was delayed longer than the collection interval. Rounds that
//! roll off the back of the ring before a sender observed them
//! surface as an explicit gap log on the next poll instead of a
//! silent overwrite.
//!
//! Targeted entries fan out into bounded per-peer MPSC queues keyed
//! by the target peer id. A sender task subscribes once (on peer-id
//! learn, either from the inbound stream header or the first payload
//! message) and owns its own queue. Full queue or missing subscriber
//! on publish → drop + log at the dispatch boundary: explicit
//! at-most-once where the previous replace-the-Arc model was
//! silently lossy under any timing skew between the collector and
//! sender tasks.
//!
//! Both pipes stamp each round with the same monotonic `round_id`
//! so per-peer senders can align emission order: broadcast-K must
//! precede targeted-K for the same round. Targeted rounds older than
//! the peer's current broadcast position are dropped — if the
//! matching broadcast round was evicted from the ring, emitting the
//! targeted portion alone would violate the ordering invariant.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use bytes::Bytes;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing as log;

/// Broadcast ring depth. 4 rounds at a 1 Hz collection tick absorb
/// transient sender stalls up to ~4 s while capping steady-state
/// memory to `depth × average_round_size`.
pub const DEFAULT_BROADCAST_RING_DEPTH: usize = 4;

/// Per-peer targeted queue depth. 4 rounds of targeted payloads per
/// peer; full queue → drop with explicit log (at-most-once).
pub const DEFAULT_TARGETED_QUEUE_DEPTH: usize = 4;

/// One round of drained broadcast entries. Shared via `Arc` so a
/// single allocation is reused across every subscribed peer — the
/// "one payload copy + O(peers) metadata" property from the design
/// spec.
#[derive(Debug)]
pub struct BroadcastRound {
    pub round_id: u64,
    pub entries: Vec<(String, Bytes)>,
}

/// One round of drained targeted entries for a specific peer.
/// Exactly one `TargetedRound` is enqueued per peer per collection
/// round (never per entry), so queue depth is bounded in *rounds*
/// rather than entry count — one large round costs the same slot as
/// a small one.
#[derive(Debug)]
pub struct TargetedRound {
    pub round_id: u64,
    pub entries: Vec<(String, Bytes)>,
}

/// Shared dispatch transport between the central collector and
/// per-peer sender tasks.
#[derive(Debug)]
pub struct StreamDispatch {
    next_round_id: AtomicU64,
    broadcast: RwLock<BroadcastRing>,
    targeted: RwLock<HashMap<String, mpsc::Sender<Arc<TargetedRound>>>>,
    targeted_depth: usize,
}

#[derive(Debug)]
struct BroadcastRing {
    depth: usize,
    rounds: VecDeque<Arc<BroadcastRound>>,
}

impl BroadcastRing {
    fn push(&mut self, round: Arc<BroadcastRound>) {
        if self.rounds.len() >= self.depth {
            self.rounds.pop_front();
        }
        self.rounds.push_back(round);
    }

    fn newest_id(&self) -> Option<u64> {
        self.rounds.back().map(|r| r.round_id)
    }

    fn oldest_id(&self) -> Option<u64> {
        self.rounds.front().map(|r| r.round_id)
    }

    /// Collect retained rounds with `round_id >= since`. If the
    /// oldest retained round is newer than `since`, returns
    /// `Some(oldest)` as the gap-adjusted start so the caller can
    /// log the jump and advance its cursor accordingly.
    fn rounds_since(&self, since: u64) -> (Vec<Arc<BroadcastRound>>, Option<u64>) {
        let Some(oldest) = self.oldest_id() else {
            return (Vec::new(), None);
        };
        let (start, gap_advance) = if oldest > since {
            (oldest, Some(oldest))
        } else {
            (since, None)
        };
        let rounds: Vec<_> = self
            .rounds
            .iter()
            .skip_while(|r| r.round_id < start)
            .cloned()
            .collect();
        (rounds, gap_advance)
    }
}

/// Result of polling the broadcast ring. `skipped` counts rounds
/// the caller lost (ring rolled past their cursor).
#[derive(Debug)]
pub struct BroadcastPoll {
    pub rounds: Vec<Arc<BroadcastRound>>,
    pub new_cursor: u64,
    pub skipped: u64,
}

impl StreamDispatch {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BROADCAST_RING_DEPTH, DEFAULT_TARGETED_QUEUE_DEPTH)
    }

    pub fn with_capacity(broadcast_depth: usize, targeted_depth: usize) -> Self {
        Self {
            next_round_id: AtomicU64::new(1),
            broadcast: RwLock::new(BroadcastRing {
                depth: broadcast_depth,
                rounds: VecDeque::with_capacity(broadcast_depth),
            }),
            targeted: RwLock::new(HashMap::new()),
            targeted_depth,
        }
    }

    /// Publish one drained collection round into the dispatch
    /// transport. `drain_entries` become a single broadcast round
    /// retained in the ring; `targeted_entries` are grouped by
    /// target peer into one `TargetedRound` per peer and enqueued
    /// into each target's per-peer queue (dropped + logged if the
    /// peer has no subscriber or its queue is full).
    ///
    /// Returns the `round_id` stamped on this round, so callers can
    /// log or correlate it.
    pub fn publish_round(
        &self,
        drain_entries: Vec<(String, Bytes)>,
        targeted_entries: Vec<(String, String, Bytes)>,
    ) -> u64 {
        let round_id = self.next_round_id.fetch_add(1, Ordering::Relaxed);

        let broadcast_round = Arc::new(BroadcastRound {
            round_id,
            entries: drain_entries,
        });
        self.broadcast.write().push(broadcast_round);

        if targeted_entries.is_empty() {
            return round_id;
        }

        let mut per_peer: HashMap<String, Vec<(String, Bytes)>> = HashMap::new();
        for (target, key, value) in targeted_entries {
            per_peer.entry(target).or_default().push((key, value));
        }

        // Hold the write guard while publishing so we can evict
        // subscribers whose receivers were dropped (Closed) without a
        // separate round-trip through the lock. subscribe / unsubscribe
        // on other senders wait briefly but are themselves rare.
        let mut subscribers = self.targeted.write();
        for (target, entries) in per_peer {
            let round = Arc::new(TargetedRound { round_id, entries });
            let Some(tx) = subscribers.get(&target) else {
                log::debug!(
                    peer = %target,
                    round_id,
                    "targeted round dropped: peer has no subscriber"
                );
                continue;
            };
            match tx.try_send(round) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    log::debug!(
                        peer = %target,
                        round_id,
                        "targeted round dropped: peer queue full"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    log::debug!(
                        peer = %target,
                        round_id,
                        "targeted round dropped: receiver gone, evicting subscriber"
                    );
                    subscribers.remove(&target);
                }
            }
        }

        round_id
    }

    /// Register a peer for targeted delivery. Returns the receiver
    /// half; the caller owns it for the lifetime of its sender task.
    /// A second subscription for the same peer replaces the first
    /// (the prior receiver sees `None` on its next `recv`, which
    /// the prior task treats as a shutdown signal).
    pub fn subscribe_targeted(&self, peer_id: &str) -> mpsc::Receiver<Arc<TargetedRound>> {
        let (tx, rx) = mpsc::channel(self.targeted_depth);
        self.targeted.write().insert(peer_id.to_string(), tx);
        rx
    }

    /// Drop a peer's targeted subscription (e.g. on stream teardown).
    /// No-op if the peer wasn't subscribed. Normally not needed —
    /// `publish_round` already evicts entries whose receiver has been
    /// dropped on the next targeted round attempt — but exposed for
    /// tests and for explicit cleanup.
    #[cfg(test)]
    pub fn unsubscribe_targeted(&self, peer_id: &str) {
        self.targeted.write().remove(peer_id);
    }

    /// Starting broadcast cursor for a new subscriber. Set to one
    /// past the newest retained round so a fresh peer only observes
    /// rounds published after they subscribed — retained rounds
    /// exist to absorb sender delays for already-subscribed peers,
    /// not to replay history to newcomers (streams are at-most-once
    /// and ephemeral by design).
    pub fn initial_broadcast_cursor(&self) -> u64 {
        self.broadcast
            .read()
            .newest_id()
            .map(|id| id + 1)
            .unwrap_or(1)
    }

    /// Poll the broadcast ring for retained rounds at or after
    /// `next_round_id`. If the ring's oldest retained round is
    /// newer than the caller's cursor, the caller lost rounds —
    /// `skipped` reports how many and `new_cursor` jumps to the
    /// oldest retained id. Callers log the gap and advance; retained
    /// rounds then emit in `round_id` order.
    pub fn poll_broadcast_rounds(&self, next_round_id: u64) -> BroadcastPoll {
        let ring = self.broadcast.read();
        let (rounds, gap_advance) = ring.rounds_since(next_round_id);
        let skipped = gap_advance.map_or(0, |a| a.saturating_sub(next_round_id));
        let new_cursor = rounds
            .last()
            .map(|r| r.round_id + 1)
            .or_else(|| gap_advance.or_else(|| ring.newest_id().map(|id| id + 1)))
            .unwrap_or(next_round_id);
        BroadcastPoll {
            rounds,
            new_cursor,
            skipped,
        }
    }
}

impl Default for StreamDispatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(s: &str) -> Bytes {
        Bytes::copy_from_slice(s.as_bytes())
    }

    #[test]
    fn test_publish_stamps_monotonic_round_ids() {
        let d = StreamDispatch::new();
        let r1 = d.publish_round(vec![], vec![]);
        let r2 = d.publish_round(vec![], vec![]);
        let r3 = d.publish_round(vec![], vec![]);
        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_eq!(r3, 3);
    }

    #[test]
    fn test_new_subscriber_skips_pre_subscription_rounds() {
        let d = StreamDispatch::new();
        d.publish_round(vec![("k".into(), bs("v"))], vec![]);
        d.publish_round(vec![("k".into(), bs("v"))], vec![]);
        // Fresh subscriber starts at newest+1: no replay.
        let cursor = d.initial_broadcast_cursor();
        let poll = d.poll_broadcast_rounds(cursor);
        assert!(poll.rounds.is_empty());
        assert_eq!(poll.skipped, 0);
    }

    #[test]
    fn test_poll_returns_retained_rounds_in_order() {
        let d = StreamDispatch::new();
        let cursor = d.initial_broadcast_cursor();
        d.publish_round(vec![("a".into(), bs("1"))], vec![]);
        d.publish_round(vec![("b".into(), bs("2"))], vec![]);
        d.publish_round(vec![("c".into(), bs("3"))], vec![]);
        let poll = d.poll_broadcast_rounds(cursor);
        assert_eq!(poll.rounds.len(), 3);
        assert_eq!(poll.rounds[0].entries[0].0, "a");
        assert_eq!(poll.rounds[1].entries[0].0, "b");
        assert_eq!(poll.rounds[2].entries[0].0, "c");
        assert_eq!(poll.skipped, 0);
        assert_eq!(poll.new_cursor, poll.rounds[2].round_id + 1);
    }

    #[test]
    fn test_gap_detection_on_ring_overflow() {
        // Depth 2 → publish 5 rounds → ring retains 4..5.
        let d = StreamDispatch::with_capacity(2, 4);
        let cursor = d.initial_broadcast_cursor();
        for i in 0..5 {
            d.publish_round(vec![(format!("k{i}"), bs("v"))], vec![]);
        }
        let poll = d.poll_broadcast_rounds(cursor);
        assert_eq!(
            poll.rounds.len(),
            2,
            "only depth=2 rounds retained: ids 4 and 5"
        );
        assert!(poll.skipped > 0, "gap must be reported");
        assert_eq!(
            poll.rounds[0].round_id,
            cursor + poll.skipped,
            "first emitted round equals cursor advanced past the gap"
        );
        assert_eq!(
            poll.new_cursor,
            poll.rounds.last().unwrap().round_id + 1,
            "new_cursor parks one past the last emitted round"
        );
    }

    #[test]
    fn test_advancing_cursor_past_newest_reports_no_skip() {
        let d = StreamDispatch::new();
        let cursor = d.initial_broadcast_cursor();
        d.publish_round(vec![("k".into(), bs("v"))], vec![]);
        let poll = d.poll_broadcast_rounds(cursor);
        assert_eq!(poll.rounds.len(), 1);
        // Second poll at new_cursor should get nothing and no skip.
        let poll2 = d.poll_broadcast_rounds(poll.new_cursor);
        assert!(poll2.rounds.is_empty());
        assert_eq!(poll2.skipped, 0);
    }

    #[tokio::test]
    async fn test_targeted_round_reaches_subscribed_peer_only() {
        let d = StreamDispatch::new();
        let mut rx_a = d.subscribe_targeted("A");
        let mut rx_b = d.subscribe_targeted("B");
        d.publish_round(
            vec![],
            vec![
                ("A".into(), "k1".into(), bs("for-a")),
                ("B".into(), "k2".into(), bs("for-b")),
                ("A".into(), "k3".into(), bs("also-for-a")),
            ],
        );

        let ra = rx_a.recv().await.unwrap();
        assert_eq!(ra.entries.len(), 2, "both A-entries grouped in one round");
        assert_eq!(ra.entries[0].1, bs("for-a"));

        let rb = rx_b.recv().await.unwrap();
        assert_eq!(rb.entries.len(), 1);
        assert_eq!(rb.entries[0].1, bs("for-b"));

        assert_eq!(ra.round_id, rb.round_id, "same round_id stamped on both");
    }

    #[tokio::test]
    async fn test_targeted_without_subscriber_is_dropped() {
        let d = StreamDispatch::new();
        // No subscription for "nobody"; publish_round logs + drops.
        let round_id = d.publish_round(vec![], vec![("nobody".into(), "k".into(), bs("orphan"))]);
        assert_eq!(round_id, 1);
    }

    #[tokio::test]
    async fn test_targeted_queue_full_drops_without_blocking() {
        let d = StreamDispatch::with_capacity(4, 1);
        let mut rx = d.subscribe_targeted("A");
        // Fill the queue with a round that won't be drained.
        d.publish_round(vec![], vec![("A".into(), "k1".into(), bs("v1"))]);
        // Next round for A — queue full, should be dropped + logged.
        d.publish_round(vec![], vec![("A".into(), "k2".into(), bs("v2"))]);

        let first = rx.recv().await.unwrap();
        assert_eq!(first.entries[0].0, "k1");

        // Ensure a third publish (now with capacity) lands.
        d.publish_round(vec![], vec![("A".into(), "k3".into(), bs("v3"))]);
        let third = rx.recv().await.unwrap();
        assert_eq!(third.entries[0].0, "k3");
    }

    #[tokio::test]
    async fn test_resubscribe_closes_prior_subscription() {
        let d = StreamDispatch::new();
        let mut rx_old = d.subscribe_targeted("A");
        d.publish_round(vec![], vec![("A".into(), "k1".into(), bs("v1"))]);
        let _ = rx_old.recv().await.unwrap();

        let mut rx_new = d.subscribe_targeted("A");
        // Old receiver now sees channel closed (sender dropped on
        // hashmap replace).
        assert!(rx_old.recv().await.is_none());

        d.publish_round(vec![], vec![("A".into(), "k2".into(), bs("v2"))]);
        let r = rx_new.recv().await.unwrap();
        assert_eq!(r.entries[0].0, "k2");
    }

    #[tokio::test]
    async fn test_unsubscribe_removes_peer() {
        let d = StreamDispatch::new();
        let _rx = d.subscribe_targeted("A");
        d.unsubscribe_targeted("A");
        // Publishing after unsubscribe drops the targeted round.
        d.publish_round(vec![], vec![("A".into(), "k".into(), bs("v"))]);
    }

    #[test]
    fn test_broadcast_and_targeted_share_round_id() {
        let d = StreamDispatch::new();
        let cursor = d.initial_broadcast_cursor();
        let _rx = d.subscribe_targeted("A");
        let round_id = d.publish_round(
            vec![("b".into(), bs("bcast"))],
            vec![("A".into(), "t".into(), bs("tgt"))],
        );
        let poll = d.poll_broadcast_rounds(cursor);
        assert_eq!(poll.rounds.len(), 1);
        assert_eq!(poll.rounds[0].round_id, round_id);
    }
}
