//! Tool inventory and indexing.
//!
//! This module provides tool storage and lookup with support for:
//! - Qualified tool names (handling collisions across servers)
//! - Tool aliasing
//! - Category-based filtering

pub mod args;
pub mod index;
pub mod types;

pub use args::ToolArgs;
pub use index::{IndexCounts, ToolInventory};
pub use types::{AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry};
