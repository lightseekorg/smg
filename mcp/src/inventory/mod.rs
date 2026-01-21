//! Tool inventory and indexing.
//!
//! This module provides tool storage and lookup with support for:
//! - Qualified tool names (handling collisions across servers)
//! - Multi-tenant isolation
//! - Tool aliasing
//! - Cache management with TTL

pub mod args;
pub mod index;
pub mod types;

pub use args::ToolArgs;
pub use index::{IndexCounts, ToolInventory};
pub use types::{AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry};
