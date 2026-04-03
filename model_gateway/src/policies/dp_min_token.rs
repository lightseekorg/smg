//! dp minimum tokens policy
//！
//! Engine support external DP dispatch
//！ this policy select dp_rank with the minimum number of tokens

use std::sync::Arc;

use super::DPRankLoadPolicy;
use crate::core::{Worker, WorkerLoadManager};

#[derive(Debug)]
pub struct MinimumTokensPolicy {
    worker_load_manager: Option<Arc<WorkerLoadManager>>,
}

impl MinimumTokensPolicy {
    pub fn new(worker_load_manager: Option<Arc<WorkerLoadManager>>) -> Self {
        Self {
            worker_load_manager,
        }
    }
}

impl DPRankLoadPolicy for MinimumTokensPolicy {
    fn select_dp_rank(&self, worker: &dyn Worker, estimated_cost: isize) -> Option<isize> {
        if let Some(worker_load) = self.worker_load_manager.as_ref() {
            return worker_load.select_and_increment_lowest_dp_load(worker, estimated_cost);
        }
        None
    }
}
