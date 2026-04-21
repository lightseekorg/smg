use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use bytes::Bytes;
use parking_lot::{Mutex, RwLock};
use tokio::sync::mpsc;
use tracing as log;

use crate::{
    chunking::{
        build_stream_batches, chunk_value, next_generation, DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    },
    kv::RoundBatch,
    service::gossip::StreamBatch,
};

pub const DEFAULT_BROADCAST_ROUND_RETENTION: usize = 4;
pub const DEFAULT_TARGETED_QUEUE_DEPTH: usize = 4;

#[derive(Debug)]
pub struct BroadcastRound {
    /// Monotonic broadcast-only sequence used for per-peer gap detection.
    pub broadcast_round_id: u64,
    /// Monotonic stream-dispatch sequence for this collection round.
    #[expect(
        dead_code,
        reason = "reserved for cross-queue ordering and diagnostics"
    )]
    pub dispatch_round_id: u64,
    pub entries: Vec<(String, Bytes)>,
}

#[derive(Debug)]
pub struct TargetedRound {
    /// Monotonic stream-dispatch sequence for this collection round.
    pub dispatch_round_id: u64,
    pub entries: Vec<(String, Bytes)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastGap {
    pub missing_rounds: u64,
    pub resume_from_broadcast_round_id: u64,
}

#[derive(Debug, Default)]
pub struct BroadcastPoll {
    pub rounds: Vec<Arc<BroadcastRound>>,
    pub gap: Option<BroadcastGap>,
}

pub struct TargetedPeerSubscription {
    dispatch: Arc<StreamDispatch>,
    peer_id: String,
    registration_id: u64,
    pub receiver: mpsc::Receiver<Arc<TargetedRound>>,
}

type TargetedPeerSender = (u64, mpsc::Sender<Arc<TargetedRound>>);
type TargetedPeerRegistry = HashMap<String, TargetedPeerSender>;

impl Drop for TargetedPeerSubscription {
    fn drop(&mut self) {
        self.dispatch
            .unregister_targeted_peer(&self.peer_id, self.registration_id);
    }
}

#[derive(Debug)]
pub struct StreamDispatch {
    next_dispatch_round_id: AtomicU64,
    next_broadcast_round_id: AtomicU64,
    next_targeted_registration_id: AtomicU64,
    broadcast_round_retention: usize,
    broadcast_rounds: RwLock<VecDeque<Arc<BroadcastRound>>>,
    targeted_queue_depth: usize,
    targeted_peers: Mutex<TargetedPeerRegistry>,
}

impl Default for StreamDispatch {
    fn default() -> Self {
        Self::new(
            DEFAULT_BROADCAST_ROUND_RETENTION,
            DEFAULT_TARGETED_QUEUE_DEPTH,
        )
    }
}

impl StreamDispatch {
    pub fn new(broadcast_round_retention: usize, targeted_queue_depth: usize) -> Self {
        assert!(
            broadcast_round_retention > 0,
            "broadcast_round_retention must be non-zero"
        );
        assert!(
            targeted_queue_depth > 0,
            "targeted_queue_depth must be non-zero"
        );
        Self {
            next_dispatch_round_id: AtomicU64::new(1),
            next_broadcast_round_id: AtomicU64::new(1),
            next_targeted_registration_id: AtomicU64::new(1),
            broadcast_round_retention,
            broadcast_rounds: RwLock::new(VecDeque::new()),
            targeted_queue_depth,
            targeted_peers: Mutex::new(HashMap::new()),
        }
    }

    pub fn initial_broadcast_round_id(&self) -> u64 {
        let rounds = self.broadcast_rounds.read();
        rounds
            .front()
            .map(|round| round.broadcast_round_id)
            .unwrap_or_else(|| self.next_broadcast_round_id.load(Ordering::Relaxed))
    }

    pub fn poll_broadcast_rounds(&self, next_broadcast_round_id: u64) -> BroadcastPoll {
        let rounds = self.broadcast_rounds.read();
        let Some(oldest) = rounds.front() else {
            return BroadcastPoll::default();
        };

        let mut poll = BroadcastPoll::default();
        let resume_from = if next_broadcast_round_id < oldest.broadcast_round_id {
            poll.gap = Some(BroadcastGap {
                missing_rounds: oldest.broadcast_round_id - next_broadcast_round_id,
                resume_from_broadcast_round_id: oldest.broadcast_round_id,
            });
            oldest.broadcast_round_id
        } else {
            next_broadcast_round_id
        };

        poll.rounds = rounds
            .iter()
            .filter(|round| round.broadcast_round_id >= resume_from)
            .cloned()
            .collect();
        poll
    }

    pub fn subscribe_targeted(self: &Arc<Self>, peer_id: &str) -> TargetedPeerSubscription {
        let registration_id = self
            .next_targeted_registration_id
            .fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel(self.targeted_queue_depth);
        self.targeted_peers
            .lock()
            .insert(peer_id.to_string(), (registration_id, tx));
        TargetedPeerSubscription {
            dispatch: self.clone(),
            peer_id: peer_id.to_string(),
            registration_id,
            receiver: rx,
        }
    }

    pub fn publish_round(&self, batch: RoundBatch) {
        if batch.drain_entries.is_empty() && batch.targeted_entries.is_empty() {
            return;
        }

        let dispatch_round_id = self.next_dispatch_round_id.fetch_add(1, Ordering::Relaxed);

        if !batch.drain_entries.is_empty() {
            let broadcast_round_id = self.next_broadcast_round_id.fetch_add(1, Ordering::Relaxed);
            let round = Arc::new(BroadcastRound {
                broadcast_round_id,
                dispatch_round_id,
                entries: batch
                    .drain_entries
                    .into_iter()
                    .map(|(key, value)| (key, Bytes::from(value)))
                    .collect(),
            });
            let mut rounds = self.broadcast_rounds.write();
            rounds.push_back(round);
            while rounds.len() > self.broadcast_round_retention {
                rounds.pop_front();
            }
        }

        if batch.targeted_entries.is_empty() {
            return;
        }

        let mut entries_by_peer: HashMap<String, Vec<(String, Bytes)>> = HashMap::new();
        for (peer_id, key, value) in batch.targeted_entries {
            entries_by_peer
                .entry(peer_id)
                .or_default()
                .push((key, value));
        }

        let targeted_peers = self.targeted_peers.lock();
        for (peer_id, entries) in entries_by_peer {
            let Some((_, tx)) = targeted_peers.get(&peer_id) else {
                log::debug!(
                    peer = %peer_id,
                    dispatch_round_id,
                    "dropping targeted stream round without an active peer queue"
                );
                continue;
            };

            let round = Arc::new(TargetedRound {
                dispatch_round_id,
                entries,
            });
            match tx.try_send(round) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    log::debug!(
                        peer = %peer_id,
                        dispatch_round_id,
                        "dropping targeted stream round on queue backpressure"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    log::debug!(
                        peer = %peer_id,
                        dispatch_round_id,
                        "dropping targeted stream round because the peer queue is closed"
                    );
                }
            }
        }
    }

    fn unregister_targeted_peer(&self, peer_id: &str, registration_id: u64) {
        let mut peers = self.targeted_peers.lock();
        if peers
            .get(peer_id)
            .is_some_and(|(current_id, _)| *current_id == registration_id)
        {
            peers.remove(peer_id);
        }
    }
}

pub fn encode_stream_batches(entries: &[(String, Bytes)]) -> Vec<StreamBatch> {
    let mut chunked_entries = Vec::new();
    for (key, value) in entries {
        chunked_entries.extend(chunk_value(
            key.clone(),
            next_generation(),
            value.clone(),
            MAX_STREAM_CHUNK_BYTES,
        ));
    }
    build_stream_batches(
        chunked_entries,
        DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    )
}
