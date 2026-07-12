//! Multimodal tensor transport resolution.
//!
//! Decides how large multimodal tensors travel from the gateway to the worker:
//! the SHM-vs-inline transport mode, its size threshold, the encoder-input wire
//! dtype, and the `/dev/shm` namespace verification that makes the SHM path safe.
//!
//! Resolution precedence for the transport mode and SHM threshold:
//! per-worker `WorkerSpec` override → router config (seeded once at startup via
//! [`init_mm_transport_defaults`]) → `SMG_MM_*` env (with the legacy
//! `SMG_TOKENSPEED_MM_*` names as a fallback) → built-in default (`inline`,
//! 64 KiB).

use std::sync::{Arc, OnceLock};

use llm_multimodal::Modality;
use openai_protocol::worker::TransportMode;
use tracing::{info, warn};

use crate::routers::grpc::{context::WorkerSelection, proto_wrapper::mm_shm_dev_writable};

const DEFAULT_SHM_MIN_BYTES: usize = 64 * 1024;

/// Router-level transport defaults, resolved once at startup from `RouterConfig`
/// (falling back to env, then built-in defaults). Per-worker `WorkerSpec`
/// overrides take precedence over these at request time.
#[derive(Debug, Clone, Copy)]
struct MmTransportDefaults {
    mode: TransportMode,
    shm_min_bytes: usize,
}

static DEFAULTS: OnceLock<MmTransportDefaults> = OnceLock::new();

/// Seed the process-wide transport defaults from router config. Config values
/// win; unset values fall back to env, then the built-in defaults. Call once at
/// startup before serving; idempotent (first call wins).
pub(crate) fn init_mm_transport_defaults(
    mode: Option<TransportMode>,
    shm_min_bytes: Option<usize>,
) {
    let resolved = MmTransportDefaults {
        mode: mode
            .or_else(mm_tensor_transport_mode_from_env)
            .unwrap_or_default(),
        shm_min_bytes: shm_min_bytes
            .or_else(mm_shm_min_bytes_from_env)
            .unwrap_or(DEFAULT_SHM_MIN_BYTES),
    };
    let _ = DEFAULTS.set(resolved);
    log_transport_config_once(resolved);
}

/// The resolved router-level defaults. If [`init_mm_transport_defaults`] was
/// never called (e.g. in tests), resolve lazily from env + built-in defaults.
fn mm_transport_defaults() -> MmTransportDefaults {
    if let Some(defaults) = DEFAULTS.get() {
        return *defaults;
    }
    MmTransportDefaults {
        mode: mm_tensor_transport_mode_from_env().unwrap_or_default(),
        shm_min_bytes: mm_shm_min_bytes_from_env().unwrap_or(DEFAULT_SHM_MIN_BYTES),
    }
}

fn mm_tensor_transport_mode_from_env() -> Option<TransportMode> {
    static LEGACY_WARNED: OnceLock<()> = OnceLock::new();
    let raw = env_with_deprecated_alias(
        "SMG_MM_TENSOR_TRANSPORT",
        "SMG_TOKENSPEED_MM_TENSOR_TRANSPORT",
        &LEGACY_WARNED,
    )?;
    match TransportMode::parse(&raw) {
        Some(mode) => Some(mode),
        None => {
            log_unknown_transport_once(&raw);
            None
        }
    }
}

fn mm_shm_min_bytes_from_env() -> Option<usize> {
    static LEGACY_WARNED: OnceLock<()> = OnceLock::new();
    let raw = env_with_deprecated_alias(
        "SMG_MM_SHM_MIN_BYTES",
        "SMG_TOKENSPEED_MM_SHM_MIN_BYTES",
        &LEGACY_WARNED,
    )?;
    match raw.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            log_invalid_shm_min_bytes_once(&raw);
            None
        }
    }
}

/// Read the canonical env var, falling back to the deprecated alias. When the
/// value comes from the alias, log a one-time migration warning (guarded by
/// `warned`, one warning per variable).
fn env_with_deprecated_alias(
    canonical: &str,
    deprecated: &str,
    warned: &OnceLock<()>,
) -> Option<String> {
    if let Some(value) = read_env_nonempty(canonical) {
        return Some(value);
    }
    let value = read_env_nonempty(deprecated)?;
    warned.get_or_init(|| {
        warn!(
            deprecated,
            canonical,
            "Deprecated multimodal transport env var is set; migrate to the canonical name"
        );
    });
    Some(value)
}

fn read_env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Resolve whether large multimodal tensors should use the SHM transport for
/// this request: per-worker override → router default. `shm` forces SHM whenever
/// SMG can write `/dev/shm` (the operator asserts co-location); `auto` also
/// requires the receiving worker leg to be verified as sharing SMG's `/dev/shm`;
/// `inline` (the default) keeps the gRPC path.
pub(super) fn resolve_mm_shm_enabled(
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> bool {
    let mode =
        worker_transport_mode_override(workers).unwrap_or_else(|| mm_transport_defaults().mode);
    match mode {
        TransportMode::Shm => mm_shm_dev_writable(),
        TransportMode::Auto => {
            worker_shares_dev_shm(workers, skip_pixel_values) && mm_shm_dev_writable()
        }
        // `rdma` routes large tensors through the NIXL pixel lane, not SHM.
        TransportMode::Inline | TransportMode::Rdma => false,
    }
}

/// Resolve the SHM size threshold (bytes) for this request: per-worker override
/// → router default.
pub(super) fn resolve_mm_shm_min_bytes(workers: Option<&WorkerSelection>) -> usize {
    worker_shm_min_bytes_override(workers).unwrap_or_else(|| mm_transport_defaults().shm_min_bytes)
}

/// Whether the router-level transport mode selects the RDMA pixel lane.
///
/// The `mm_rdma` gate consults this so `rdma` is a first-class [`TransportMode`]
/// alongside `inline`/`shm`/`auto` (settable via `--multimodal-tensor-transport`
/// / `SMG_MM_TENSOR_TRANSPORT`). The legacy `SMG_MM_PIXEL_RDMA` env stays a
/// fallback inside the gate for backward compatibility. Only compiled with the
/// `mm-rdma` feature, since the no-op RDMA shim never consults it.
#[cfg(feature = "mm-rdma")]
pub(crate) fn mm_default_transport_is_rdma() -> bool {
    mm_transport_defaults().mode == TransportMode::Rdma
}

fn worker_transport_mode_override(workers: Option<&WorkerSelection>) -> Option<TransportMode> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_tensor_transport
}

fn worker_shm_min_bytes_override(workers: Option<&WorkerSelection>) -> Option<usize> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_shm_min_bytes
}

/// The worker whose per-worker overrides apply. Multimodal tensors are sent to
/// wherever the vision encoder runs: the encode worker in EPD (so its spec wins),
/// otherwise the single/prefill worker that does the encoding itself.
fn primary_worker(workers: Option<&WorkerSelection>) -> Option<&Arc<dyn crate::worker::Worker>> {
    match workers? {
        WorkerSelection::Single { worker } => Some(worker),
        WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            ..
        } => encode_assignments
            .as_ref()
            .and_then(|assignments| assignments.first())
            .map(|assignment| &assignment.worker)
            .or(Some(prefill)),
    }
}

pub(super) fn mm_encoder_input_dtype(
    modality: Modality,
    workers: Option<&WorkerSelection>,
) -> String {
    resolve_mm_encoder_input_dtype(
        mm_modality_encoder_input_dtype_from_env(modality),
        mm_default_encoder_input_dtype_from_env(),
        mm_encoder_input_dtype_from_worker(workers),
    )
}

fn resolve_mm_encoder_input_dtype(
    modality_override: Option<String>,
    default_override: Option<String>,
    worker_dtype: Option<String>,
) -> String {
    // Use one configurable wire policy across modalities, with an optional
    // per-modality override for encoder contracts that require another dtype.
    modality_override
        .or(default_override)
        .or(worker_dtype)
        .unwrap_or_else(|| "bfloat16".to_string())
}

fn mm_modality_encoder_input_dtype_from_env(modality: Modality) -> Option<String> {
    static IMAGE_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static VIDEO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static AUDIO_DTYPE: OnceLock<Option<String>> = OnceLock::new();

    match modality {
        Modality::Image | Modality::ImageEmbeds => {
            cached_env_dtype(&IMAGE_DTYPE, "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE")
        }
        Modality::Video => {
            cached_env_dtype(&VIDEO_DTYPE, "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE")
        }
        Modality::Audio => {
            cached_env_dtype(&AUDIO_DTYPE, "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE")
        }
    }
}

fn mm_default_encoder_input_dtype_from_env() -> Option<String> {
    static DEFAULT_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    cached_env_dtype(&DEFAULT_DTYPE, "SMG_TOKENSPEED_ENCODER_INPUT_DTYPE")
}

fn cached_env_dtype(cell: &'static OnceLock<Option<String>>, name: &str) -> Option<String> {
    cell.get_or_init(|| std::env::var(name).ok().filter(|dtype| !dtype.is_empty()))
        .clone()
}

fn mm_encoder_input_dtype_from_worker(workers: Option<&WorkerSelection>) -> Option<String> {
    primary_worker(workers)?
        .metadata()
        .spec
        .labels
        .get("multimodal_encoder_dtype")
        .filter(|dtype| !dtype.is_empty())
        .cloned()
}

fn log_transport_config_once(defaults: MmTransportDefaults) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            mode = %defaults.mode,
            shm_min_bytes = defaults.shm_min_bytes,
            dev_writable = mm_shm_dev_writable(),
            "Multimodal tensor transport configured"
        );
    });
}

fn log_unknown_transport_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Unknown multimodal tensor transport value; expected inline|shm|auto|rdma, using inline"
        );
    });
}

fn log_invalid_shm_min_bytes_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Invalid multimodal SHM min-bytes value; expected a non-negative integer, using default"
        );
    });
}

/// Whether the worker is *verified* to share SMG's `/dev/shm`, making the SHM
/// transport safe for this payload.
///
/// Rather than inferring locality from the worker URL (TCP loopback proves only
/// network locality, not a shared `/dev/shm`), the worker advertises its
/// `/dev/shm` filesystem identity (`<boot_id>:<st_dev of /dev/shm>`) via
/// `GetServerInfo`, which discovery stores in the worker's `shm_namespace_id`
/// label. Two processes share `/dev/shm` iff these tokens match: `boot_id` pins
/// the host, and `st_dev` is the tmpfs superblock device, identical whenever the
/// same tmpfs backs both `/dev/shm` mounts — including separate containers that
/// share it via `--ipc`/bind-mount (where mount-namespace inodes differ but the
/// underlying superblock is the same). We compare the worker's token to ours:
/// equal ⇒ shared. A missing/empty token or any mismatch is treated as
/// non-sharing, so `auto` safely falls back to inline.
fn worker_shares_dev_shm(workers: Option<&WorkerSelection>, skip_pixel_values: bool) -> bool {
    let Some(local) = local_shm_namespace_id() else {
        return false;
    };
    match workers {
        Some(WorkerSelection::Single { worker }) => worker_matches_shm_namespace(worker, local),
        Some(WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            decode,
            ..
        }) => {
            if !skip_pixel_values {
                if let Some(encode_assignments) = encode_assignments {
                    // EPD: encoder_input (pixels) ships gateway -> encode worker, so SHM
                    // is safe only if every encode worker assigned in this request shares
                    // the gateway's /dev/shm. A mixed local/remote fan-out must fall back
                    // to inline/RDMA rather than giving a remote worker an unreadable SHM handle.
                    return encode_assignments
                        .iter()
                        .all(|assignment| worker_matches_shm_namespace(&assignment.worker, local));
                }
            }
            worker_matches_shm_namespace(prefill, local)
                && worker_matches_shm_namespace(decode, local)
        }
        None => false,
    }
}

fn worker_matches_shm_namespace(worker: &Arc<dyn crate::worker::Worker>, local: &str) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get("shm_namespace_id")
        .is_some_and(|id| !id.is_empty() && id == local)
}

/// This process's `/dev/shm` filesystem identity: `<boot_id>:<st_dev of /dev/shm>`.
/// `boot_id` pins the host (it is not namespaced) and `st_dev` is the tmpfs
/// superblock device backing `/dev/shm`; together they identify the tmpfs so two
/// processes sharing it (even across containers via `--ipc`/bind-mount) produce
/// the same token. Computed once; `None` if it can't be determined (then `auto`
/// stays inline).
fn local_shm_namespace_id() -> Option<&'static str> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(compute_shm_namespace_id).as_deref()
}

#[cfg(unix)]
fn compute_shm_namespace_id() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let shm_dev = std::fs::metadata("/dev/shm").ok()?.dev();
    Some(format!("{}:{shm_dev}", boot_id.trim()))
}

#[cfg(not(unix))]
fn compute_shm_namespace_id() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_dtype_precedes_global_then_worker_with_bfloat16_fallback() {
        assert_eq!(
            resolve_mm_encoder_input_dtype(
                Some("float16".to_string()),
                Some("float32".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float16"
        );
        assert_eq!(
            resolve_mm_encoder_input_dtype(
                None,
                Some("float32".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float32"
        );
        assert_eq!(
            resolve_mm_encoder_input_dtype(None, None, Some("bfloat16".to_string()),),
            "bfloat16"
        );
        assert_eq!(resolve_mm_encoder_input_dtype(None, None, None), "bfloat16");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn local_shm_namespace_id_resolves_on_linux() {
        // /proc/.../boot_id and /dev/shm both exist on the Linux CI/runtime
        // image, so the token must resolve to `<boot_id>:<st_dev>`. If it ever
        // returned None, `auto` would silently never enable SHM.
        let id = local_shm_namespace_id().expect("shm namespace id should resolve on Linux");
        assert!(
            id.contains(':'),
            "token must be <boot_id>:<st_dev>, got {id:?}"
        );
        let dev = id.rsplit(':').next().unwrap();
        assert!(
            dev.parse::<u64>().is_ok(),
            "st_dev component must be numeric, got {id:?}"
        );
    }
}
