//! Internal tests module
//!
//! This module contains comprehensive integration and unit tests
//! that have full access to private crate internals.

#[cfg(test)]
mod chunking_integration;
#[cfg(test)]
mod comprehensive;
pub(crate) mod test_utils;
