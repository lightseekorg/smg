//! Deliberate construction of the gateway's tokio runtime.
//!
//! Replaces the bare `tokio::runtime::Runtime::new()` call, which sized the
//! worker pool from an *unclamped* `TOKIO_WORKER_THREADS` (a 32-thread pool on
//! a 12-core host makes reactor starvation worse, not better) and left the
//! blocking pool at tokio's default of 512 threads, with nothing validated or
//! logged.
//!
//! Worker thread count — first match wins:
//! 1. `SMG_WORKER_THREADS`
//! 2. `TOKIO_WORKER_THREADS` (back-compat: tokio itself honored this when the
//!    runtime was built implicitly)
//! 3. default: `std::thread::available_parallelism()` (cgroup/cpuset-aware,
//!    unlike a raw physical core count)
//!
//! Any worker override is clamped to [`WORKER_THREADS_CLAMP_FACTOR`] x
//! `available_parallelism` with a warning; zero or non-numeric values are
//! rejected at startup.
//!
//! Max blocking threads — first match wins:
//! 1. `SMG_MAX_BLOCKING_THREADS` (not clamped: a deliberate escape hatch back
//!    toward tokio's old 512 default if the cap ever converts a thread storm
//!    into blocking-pool queueing)
//! 2. default: `available_parallelism` (instead of tokio's 512)
//!
//! Empty or whitespace-only values are treated as unset. Present values are
//! validated even when a higher-precedence variable outranks them, so a busted
//! deploy is surfaced rather than masked.
//!
//! The runtime is built before logging is initialized (`main` runs pre-logger;
//! tracing comes up inside `server::startup`), so the effective configuration
//! is stashed here and emitted by [`log_effective_config`] once a subscriber
//! exists.

use std::fmt;

use parking_lot::Mutex;
use tracing::{info, warn};

use crate::config::{ConfigError, ConfigResult};

/// Preferred override for the number of tokio worker threads.
pub const WORKER_THREADS_ENV: &str = "SMG_WORKER_THREADS";
/// Back-compat override for worker threads (honored by tokio itself before
/// SMG built its runtime explicitly). `SMG_WORKER_THREADS` wins if both are set.
pub const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
/// Override for the blocking pool cap.
pub const MAX_BLOCKING_THREADS_ENV: &str = "SMG_MAX_BLOCKING_THREADS";

/// Worker overrides are clamped to this multiple of `available_parallelism`.
pub const WORKER_THREADS_CLAMP_FACTOR: usize = 4;

/// Where a resolved thread count came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadCountSource {
    /// Derived from `std::thread::available_parallelism()`.
    Default,
    /// Set by the named environment variable.
    Env(&'static str),
    /// Set by the named environment variable, then clamped to the ceiling.
    EnvClamped(&'static str),
}

impl fmt::Display for ThreadCountSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "default (available_parallelism)"),
            Self::Env(var) => write!(f, "{var}"),
            Self::EnvClamped(var) => write!(f, "{var} (clamped)"),
        }
    }
}

/// Validated thread configuration for the gateway runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub worker_threads: usize,
    pub worker_threads_source: ThreadCountSource,
    pub max_blocking_threads: usize,
    pub max_blocking_threads_source: ThreadCountSource,
    pub available_parallelism: usize,
    /// Validation warnings (e.g. clamped overrides), emitted once logging is up.
    pub warnings: Vec<String>,
}

impl RuntimeConfig {
    /// Resolve the runtime configuration from the process environment.
    pub fn from_env() -> ConfigResult<Self> {
        let available = std::thread::available_parallelism()
            .map_err(|e| ConfigError::ValidationFailed {
                reason: format!("Failed to query available parallelism: {e}"),
            })?
            .get();
        resolve(available, |var| std::env::var(var).ok())
    }

    /// Build the multi-threaded runtime described by this configuration.
    pub fn create_runtime(&self) -> std::io::Result<tokio::runtime::Runtime> {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.worker_threads)
            .max_blocking_threads(self.max_blocking_threads)
            .thread_name("smg-worker")
            .enable_all()
            .build()
    }
}

impl fmt::Display for RuntimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "worker_threads={} (source: {}), max_blocking_threads={} (source: {}), \
             available_parallelism={}",
            self.worker_threads,
            self.worker_threads_source,
            self.max_blocking_threads,
            self.max_blocking_threads_source,
            self.available_parallelism
        )
    }
}

/// Pure resolution of the runtime configuration; `env` supplies environment
/// lookups so tests never have to mutate (and race on) the process environment.
fn resolve(
    available_parallelism: usize,
    env: impl Fn(&str) -> Option<String>,
) -> ConfigResult<RuntimeConfig> {
    let mut warnings = Vec::new();

    let worker_override = parse_thread_count(WORKER_THREADS_ENV, &env)?
        .or(parse_thread_count(TOKIO_WORKER_THREADS_ENV, &env)?);
    let (worker_threads, worker_threads_source) = match worker_override {
        Some((var, requested)) => {
            let ceiling = WORKER_THREADS_CLAMP_FACTOR * available_parallelism;
            if requested > ceiling {
                warnings.push(format!(
                    "{var}={requested} exceeds {ceiling} ({WORKER_THREADS_CLAMP_FACTOR}x \
                     available_parallelism={available_parallelism}); clamping worker threads \
                     to {ceiling}"
                ));
                (ceiling, ThreadCountSource::EnvClamped(var))
            } else {
                (requested, ThreadCountSource::Env(var))
            }
        }
        None => (available_parallelism, ThreadCountSource::Default),
    };

    let (max_blocking_threads, max_blocking_threads_source) =
        match parse_thread_count(MAX_BLOCKING_THREADS_ENV, &env)? {
            Some((var, requested)) => (requested, ThreadCountSource::Env(var)),
            None => (available_parallelism, ThreadCountSource::Default),
        };

    Ok(RuntimeConfig {
        worker_threads,
        worker_threads_source,
        max_blocking_threads,
        max_blocking_threads_source,
        available_parallelism,
        warnings,
    })
}

/// Parse one thread-count variable. Unset, empty, or whitespace-only values
/// resolve to `None`; zero and non-numeric values are rejected.
fn parse_thread_count(
    var: &'static str,
    env: impl Fn(&str) -> Option<String>,
) -> ConfigResult<Option<(&'static str, usize)>> {
    let Some(raw) = env(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match trimmed.parse::<usize>() {
        Ok(0) => Err(ConfigError::InvalidValue {
            field: var.to_string(),
            value: raw,
            reason: "thread count must be at least 1; unset the variable to use the default"
                .to_string(),
        }),
        Ok(count) => Ok(Some((var, count))),
        Err(_) => Err(ConfigError::InvalidValue {
            field: var.to_string(),
            value: raw,
            reason: "expected a positive integer".to_string(),
        }),
    }
}

/// Configuration stashed by `main` for logging once tracing is initialized.
static PENDING_STARTUP_LOG: Mutex<Option<RuntimeConfig>> = Mutex::new(None);

/// Stash the effective configuration so [`log_effective_config`] can emit it
/// after logging init. The runtime is built pre-logger, so logging from the
/// resolution site would be lost.
pub fn stash_startup_log(config: RuntimeConfig) {
    *PENDING_STARTUP_LOG.lock() = Some(config);
}

/// Emit the stashed runtime configuration. Called by `server::startup` right
/// after logging init; a no-op when nothing is stashed (e.g. embedders that
/// call `startup` on their own runtime) or on repeated calls.
pub fn log_effective_config() {
    let Some(config) = PENDING_STARTUP_LOG.lock().take() else {
        return;
    };
    for warning in &config.warnings {
        warn!("{warning}");
    }
    info!(
        worker_threads = config.worker_threads,
        worker_threads_source = %config.worker_threads_source,
        max_blocking_threads = config.max_blocking_threads,
        max_blocking_threads_source = %config.max_blocking_threads_source,
        available_parallelism = config.available_parallelism,
        "Tokio runtime configured"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Build an env lookup from a fixed map. Resolution is tested through this
    /// instead of `std::env::set_var` so tests cannot race on process state.
    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(var, value)| (var.to_string(), value.to_string()))
            .collect();
        move |var| map.get(var).cloned()
    }

    #[test]
    fn defaults_to_available_parallelism() {
        let config = resolve(12, env(&[])).unwrap();
        assert_eq!(config.worker_threads, 12);
        assert_eq!(config.worker_threads_source, ThreadCountSource::Default);
        assert_eq!(config.max_blocking_threads, 12);
        assert_eq!(
            config.max_blocking_threads_source,
            ThreadCountSource::Default
        );
        assert_eq!(config.available_parallelism, 12);
        assert!(config.warnings.is_empty());
    }

    #[test]
    fn smg_worker_threads_overrides_default() {
        let config = resolve(12, env(&[(WORKER_THREADS_ENV, "6")])).unwrap();
        assert_eq!(config.worker_threads, 6);
        assert_eq!(
            config.worker_threads_source,
            ThreadCountSource::Env(WORKER_THREADS_ENV)
        );
        assert!(config.warnings.is_empty());
    }

    #[test]
    fn tokio_worker_threads_honored_for_back_compat() {
        let config = resolve(12, env(&[(TOKIO_WORKER_THREADS_ENV, "8")])).unwrap();
        assert_eq!(config.worker_threads, 8);
        assert_eq!(
            config.worker_threads_source,
            ThreadCountSource::Env(TOKIO_WORKER_THREADS_ENV)
        );
    }

    #[test]
    fn smg_env_takes_precedence_over_tokio_env() {
        let config = resolve(
            12,
            env(&[(WORKER_THREADS_ENV, "6"), (TOKIO_WORKER_THREADS_ENV, "32")]),
        )
        .unwrap();
        assert_eq!(config.worker_threads, 6);
        assert_eq!(
            config.worker_threads_source,
            ThreadCountSource::Env(WORKER_THREADS_ENV)
        );
    }

    #[test]
    fn worker_override_above_ceiling_is_clamped_with_warning() {
        // The #1685 shape: an oversized TOKIO_WORKER_THREADS on a small host.
        let config = resolve(12, env(&[(TOKIO_WORKER_THREADS_ENV, "64")])).unwrap();
        assert_eq!(config.worker_threads, 48); // 4x12
        assert_eq!(
            config.worker_threads_source,
            ThreadCountSource::EnvClamped(TOKIO_WORKER_THREADS_ENV)
        );
        assert_eq!(config.warnings.len(), 1);
        assert!(config.warnings[0].contains("TOKIO_WORKER_THREADS=64"));
        assert!(config.warnings[0].contains("clamping"));
    }

    #[test]
    fn worker_override_at_ceiling_is_not_clamped() {
        let config = resolve(12, env(&[(WORKER_THREADS_ENV, "48")])).unwrap();
        assert_eq!(config.worker_threads, 48);
        assert_eq!(
            config.worker_threads_source,
            ThreadCountSource::Env(WORKER_THREADS_ENV)
        );
        assert!(config.warnings.is_empty());
    }

    #[test]
    fn zero_worker_threads_rejected() {
        let err = resolve(12, env(&[(WORKER_THREADS_ENV, "0")])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(WORKER_THREADS_ENV), "{message}");
        assert!(message.contains("at least 1"), "{message}");
    }

    #[test]
    fn zero_blocking_threads_rejected() {
        let err = resolve(12, env(&[(MAX_BLOCKING_THREADS_ENV, "0")])).unwrap_err();
        assert!(err.to_string().contains(MAX_BLOCKING_THREADS_ENV));
    }

    #[test]
    fn non_numeric_worker_threads_rejected() {
        let err = resolve(12, env(&[(TOKIO_WORKER_THREADS_ENV, "many")])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(TOKIO_WORKER_THREADS_ENV), "{message}");
        assert!(message.contains("positive integer"), "{message}");
    }

    #[test]
    fn invalid_lower_precedence_value_still_rejected() {
        let err = resolve(
            12,
            env(&[
                (WORKER_THREADS_ENV, "6"),
                (TOKIO_WORKER_THREADS_ENV, "junk"),
            ]),
        )
        .unwrap_err();
        assert!(err.to_string().contains(TOKIO_WORKER_THREADS_ENV));
    }

    #[test]
    fn empty_values_treated_as_unset() {
        let config = resolve(
            12,
            env(&[(WORKER_THREADS_ENV, ""), (MAX_BLOCKING_THREADS_ENV, "  ")]),
        )
        .unwrap();
        assert_eq!(config.worker_threads, 12);
        assert_eq!(config.worker_threads_source, ThreadCountSource::Default);
        assert_eq!(config.max_blocking_threads, 12);
    }

    #[test]
    fn whitespace_around_values_is_trimmed() {
        let config = resolve(12, env(&[(WORKER_THREADS_ENV, " 6 ")])).unwrap();
        assert_eq!(config.worker_threads, 6);
    }

    #[test]
    fn blocking_override_honored_without_clamp() {
        // Deliberate escape hatch: the blocking cap may be raised back toward
        // tokio's old default without a clamp fighting the operator.
        let config = resolve(12, env(&[(MAX_BLOCKING_THREADS_ENV, "512")])).unwrap();
        assert_eq!(config.max_blocking_threads, 512);
        assert_eq!(
            config.max_blocking_threads_source,
            ThreadCountSource::Env(MAX_BLOCKING_THREADS_ENV)
        );
        assert!(config.warnings.is_empty());
    }
}
