//! Multimodal tensor transport resolution.
//!
//! Decides how large multimodal tensors travel from the gateway to the worker:
//! the SHM-vs-inline transport mode, the encoder-input wire dtype, and the
//! `/dev/shm` namespace verification that makes the SHM path safe. All values
//! resolve from environment + worker labels today.

use std::sync::{Arc, OnceLock};

use llm_multimodal::Modality;
use tracing::{info, warn};

use crate::routers::grpc::{
    context::WorkerSelection,
    proto_wrapper::{
        tokenspeed_mm_shm_min_bytes, tokenspeed_mm_tensor_transport_mode,
        tokenspeed_shm_dev_writable,
    },
};

pub(super) fn tokenspeed_encoder_input_dtype(
    modality: Modality,
    workers: Option<&WorkerSelection>,
) -> String {
    if let Some(dtype) = tokenspeed_encoder_input_dtype_from_env(modality) {
        return dtype;
    }
    if let Some(dtype) = tokenspeed_encoder_input_dtype_from_worker(workers) {
        return dtype;
    }
    // Default to bf16 on the wire: the engine casts encoder_input to the model
    // dtype (bf16) at the ViT regardless, so this is numerically identical to f32
    // while halving the gateway->encode payload (the EPD throughput limiter).
    // Override per-modality via SMG_TOKENSPEED_*_ENCODER_INPUT_DTYPE.
    "bfloat16".to_string()
}

fn tokenspeed_encoder_input_dtype_from_env(modality: Modality) -> Option<String> {
    static IMAGE_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static VIDEO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static AUDIO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static DEFAULT_DTYPE: OnceLock<Option<String>> = OnceLock::new();

    let modality_dtype = match modality {
        Modality::Image | Modality::ImageEmbeds => {
            cached_env_dtype(&IMAGE_DTYPE, "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE")
        }
        Modality::Video => {
            cached_env_dtype(&VIDEO_DTYPE, "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE")
        }
        Modality::Audio => {
            cached_env_dtype(&AUDIO_DTYPE, "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE")
        }
    };
    modality_dtype
        .or_else(|| cached_env_dtype(&DEFAULT_DTYPE, "SMG_TOKENSPEED_ENCODER_INPUT_DTYPE"))
}

fn cached_env_dtype(cell: &'static OnceLock<Option<String>>, name: &str) -> Option<String> {
    cell.get_or_init(|| std::env::var(name).ok().filter(|dtype| !dtype.is_empty()))
        .clone()
}

fn tokenspeed_encoder_input_dtype_from_worker(workers: Option<&WorkerSelection>) -> Option<String> {
    let worker = match workers? {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Disaggregated { prefill, .. } => prefill,
    };
    worker
        .metadata()
        .spec
        .labels
        .get("multimodal_encoder_dtype")
        .filter(|dtype| !dtype.is_empty())
        .cloned()
}

/// Resolve whether large multimodal tensors should use the SHM transport for
/// this request. `shm` and `auto` require the receiving worker leg to share
/// SMG's `/dev/shm`; anything else (including unset or `inline`) keeps the
/// inline gRPC path.
pub(super) fn resolve_tokenspeed_shm_enabled(
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> bool {
    let mode = tokenspeed_mm_tensor_transport_mode();
    log_tokenspeed_transport_config_once(&mode);
    match mode.as_str() {
        // SHM only ever happens when SMG can actually write /dev/shm.
        "shm" | "auto" => {
            worker_shares_dev_shm(workers, skip_pixel_values) && tokenspeed_shm_dev_writable()
        }
        "" | "inline" => false,
        other => {
            log_unknown_tokenspeed_transport_once(other);
            false
        }
    }
}

fn log_tokenspeed_transport_config_once(mode: &str) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            mode,
            shm_min_bytes = tokenspeed_mm_shm_min_bytes(),
            dev_writable = tokenspeed_shm_dev_writable(),
            "TokenSpeed multimodal tensor transport configured"
        );
    });
}

fn log_unknown_tokenspeed_transport_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Unknown SMG_TOKENSPEED_MM_TENSOR_TRANSPORT value; expected inline|shm|auto, using inline"
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
