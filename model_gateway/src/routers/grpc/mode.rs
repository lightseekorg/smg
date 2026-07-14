//! The single source of truth for the gRPC disaggregation mode. Regular/PD/EPD
//! is one axis; every place that used to re-derive it (router structs, the
//! `router_type` label, the per-endpoint pipeline constructors) will map from
//! this enum.

use crate::config::types::{RouterConfig, RoutingMode};
use crate::routers::grpc::common::stages::WorkerSelectionMode;
use crate::routers::grpc::context::ExecutionPlanKind;
use crate::worker::ConnectionMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Regular,
    PrefillDecode,
    EncodePrefillDecode,
}

impl Mode {
    pub(crate) fn worker_selection(self) -> WorkerSelectionMode {
        match self {
            Mode::Regular => WorkerSelectionMode::Regular,
            Mode::PrefillDecode => WorkerSelectionMode::PrefillDecode,
            Mode::EncodePrefillDecode => WorkerSelectionMode::EncodePrefillDecode,
        }
    }
    pub(crate) fn plan_kind(self) -> ExecutionPlanKind {
        match self {
            Mode::Regular => ExecutionPlanKind::Single,
            Mode::PrefillDecode => ExecutionPlanKind::PrefillDecode,
            Mode::EncodePrefillDecode => ExecutionPlanKind::EncodePrefillDecode,
        }
    }
    /// SGLang PD bootstrap injection: PD only. EPD (TokenSpeed) uses the encode
    /// bootstrap info instead, so it must stay false to avoid double injection.
    pub(crate) fn inject_pd_metadata(self) -> bool {
        matches!(self, Mode::PrefillDecode)
    }
    /// Metrics/introspection label. Strings preserved from the pre-refactor routers.
    pub(crate) fn router_type(self) -> &'static str {
        match self {
            Mode::Regular => "grpc",
            Mode::PrefillDecode => "grpc_pd",
            Mode::EncodePrefillDecode => "grpc_epd",
        }
    }
}

/// Derive the gRPC mode from config. Returns `None` for non-gRPC backends
/// (OpenAI/Anthropic/Gemini/HTTP), served by other router types.
pub(crate) fn grpc_mode(cfg: &RouterConfig) -> Option<Mode> {
    if cfg.connection_mode != ConnectionMode::Grpc {
        return None;
    }
    match cfg.mode {
        RoutingMode::Regular { .. } => Some(Mode::Regular),
        RoutingMode::PrefillDecode { .. } => Some(Mode::PrefillDecode),
        RoutingMode::EncodePrefillDecode { .. } => Some(Mode::EncodePrefillDecode),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_maps_to_stage_params() {
        let cases = [
            (Mode::Regular, WorkerSelectionMode::Regular, ExecutionPlanKind::Single, false, "grpc"),
            (Mode::PrefillDecode, WorkerSelectionMode::PrefillDecode, ExecutionPlanKind::PrefillDecode, true, "grpc_pd"),
            (Mode::EncodePrefillDecode, WorkerSelectionMode::EncodePrefillDecode, ExecutionPlanKind::EncodePrefillDecode, false, "grpc_epd"),
        ];
        for (m, ws, pk, inject, rt) in cases {
            assert_eq!(m.worker_selection(), ws, "worker_selection {m:?}");
            assert_eq!(m.plan_kind(), pk, "plan_kind {m:?}");
            assert_eq!(m.inject_pd_metadata(), inject, "inject_pd_metadata {m:?}");
            assert_eq!(m.router_type(), rt, "router_type {m:?}");
        }
    }
}
