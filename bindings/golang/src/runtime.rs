//! Shared runtime and global resources for FFI

use once_cell::sync::Lazy;
use smg::tool_parser::ParserFactory;
use tokio::runtime::Runtime;

/// Global tokio runtime for all async FFI operations
pub static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create tokio runtime for FFI"));

/// Global parser factory (initialized once)
pub static PARSER_FACTORY: Lazy<ParserFactory> = Lazy::new(ParserFactory::new);
