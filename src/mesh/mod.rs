//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This module re-exports the smg-mesh crate and provides HTTP API endpoints.

// Re-export everything from smg-mesh crate
pub use smg_mesh::*;

// Local HTTP API routes (depends on app-specific types like AppState)
pub mod endpoints;

// Re-export endpoint functions for convenience
pub use endpoints::{
    get_app_config, get_cluster_status, get_global_rate_limit, get_global_rate_limit_stats,
    get_mesh_health, get_policy_state, get_policy_states, get_worker_state, get_worker_states,
    set_global_rate_limit, trigger_graceful_shutdown, update_app_config,
};
