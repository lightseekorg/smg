//! Worker load
//!
//! Record and manage the DP group load of workers.
use std::{collections::HashMap, fmt::Debug, sync::RwLock};

use tracing::debug;

use crate::core::Worker;

#[derive(Debug, Default)]
pub struct WorkerLoadManager {
    // <worker, <dp_rank, loads>>
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl WorkerLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("WorkerLoadManager update_dp_loads map:{:?}", loads);
        let mut cached = self
            .dp_cached_loads
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    pub fn select_and_increment_lowest_dp_load(
        &self,
        worker: &dyn Worker,
        increment: isize,
    ) -> Option<isize> {
        let mut cached = self
            .dp_cached_loads
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let loads = cached.get_mut(worker.url())?;
        let (&dp_rank, _) = loads.iter().min_by_key(|&(rank, load)| (*load, *rank))?;
        if let Some(v) = loads.get_mut(&dp_rank) {
            *v += increment;
        }
        Some(dp_rank)
    }

    pub fn remove_workers(&self, urls: &[String]) {
        let mut cached = self
            .dp_cached_loads
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for url in urls {
            cached.remove(url);
        }
    }
}

#[cfg(test)]
mod dp_load_manager_tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = WorkerLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read().unwrap();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();

        // insert worker1_load
        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        // insert worker2.load
        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        // update
        manager.update_dp_loads(&loads);

        // assert
        let cached = manager.dp_cached_loads.read().unwrap();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_multiple() {
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_key")
            .build();

        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();
        let mut worker_load = HashMap::new();
        worker_load.insert(0, 10);
        worker_load.insert(1, 3);
        worker_load.insert(2, 7);
        loads.insert(worker.url().to_string(), worker_load);
        manager.update_dp_loads(&loads);

        let selected = manager.select_and_increment_lowest_dp_load(&worker, 4);

        assert_eq!(selected, Some(1));
        let cached = manager.dp_cached_loads.read().unwrap();
        assert_eq!(*cached.get(worker.url()).unwrap().get(&1).unwrap(), 3 + 4);
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_none_worker() {
        let worker = BasicWorkerBuilder::new("http://nonexist:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test")
            .build();

        let manager = WorkerLoadManager::new();
        let result = manager.select_and_increment_lowest_dp_load(&worker, 1);
        assert_eq!(result, None);
    }
}
