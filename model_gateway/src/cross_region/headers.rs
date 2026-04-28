use std::str::FromStr;

use http::{HeaderMap, StatusCode};
use serde::{Deserialize, Serialize};

use super::{
    CrossRegionError, CrossRegionResult, FailoverPolicy, ModalityPolicy, RoutingProfileContext,
};
use crate::config::CrossRegionFailoverMode;

/// Supported cross-region header contract version.
pub const SUPPORTED_CONTRACT_VERSION: u32 = 1;

/// Header carrying the cross-region header contract version.
pub const CONTRACT_VERSION_HEADER: &str = "x-contract-version";

/// Header used by DP-API and remote SMG to indicate routing state.
pub const REQUEST_MODE_HEADER: &str = "x-request-mode";

/// Header carrying the original DP-API entry region.
pub const ENTRY_REGION_HEADER: &str = "x-entry-region";

/// Header carrying the service that delegated or forwarded the request.
pub const SOURCE_SERVICE_HEADER: &str = "x-source-service";

/// Header carrying the request id used for tracing across regions.
pub const OPC_REQUEST_ID_HEADER: &str = "x-opc-request-id";

/// Header carrying comma-separated candidate regions allowed by the profile.
pub const ALLOWED_REGIONS_HEADER: &str = "x-allowed-regions";

/// Header carrying comma-separated candidate models allowed by the profile.
pub const ALLOWED_MODELS_HEADER: &str = "x-allowed-models";

/// Header carrying the routing-profile failover mode.
pub const FAILOVER_MODE_HEADER: &str = "x-failover-mode";

/// Header carrying the routing-profile retry cap.
pub const MAX_RETRY_HEADER: &str = "x-max-retry";

/// Header carrying the optional request input modality.
pub const INPUT_MODALITY_HEADER: &str = "x-input-modality";

/// Header carrying the optional request output modality.
pub const OUTPUT_MODALITY_HEADER: &str = "x-output-modality";

/// Header carrying the target region after a route has been committed.
pub const TARGET_REGION_HEADER: &str = "x-target-region";

/// Header carrying the committed route id.
pub const ROUTE_ID_HEADER: &str = "x-route-id";

/// Header carrying the committed route attempt number.
pub const ATTEMPT_HEADER: &str = "x-attempt";

/// Header value that means SMG still needs to calculate a target region.
pub const REQUEST_MODE_UNRESOLVED: &str = "UNRESOLVED";

/// Header value that means the entry SMG already committed the target region.
pub const REQUEST_MODE_SETTLED: &str = "SETTLED";

/// Request mode carried by the cross-region header contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RequestMode {
    #[default]
    Unresolved,
    Settled,
}

impl RequestMode {
    /// Return the canonical wire value for the request mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unresolved => REQUEST_MODE_UNRESOLVED,
            Self::Settled => REQUEST_MODE_SETTLED,
        }
    }
}

impl std::fmt::Display for RequestMode {
    /// Format the request mode using the canonical header value.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RequestMode {
    type Err = CrossRegionError;

    /// Parse request mode from a case-insensitive header value.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_uppercase().as_str() {
            REQUEST_MODE_UNRESOLVED => Ok(Self::Unresolved),
            REQUEST_MODE_SETTLED => Ok(Self::Settled),
            _ => Err(CrossRegionError::InvalidHeader {
                reason: format!(
                    "{REQUEST_MODE_HEADER} must be {REQUEST_MODE_UNRESOLVED} or {REQUEST_MODE_SETTLED}"
                ),
            }),
        }
    }
}

/// Settled-only route metadata forwarded from the entry SMG to the target SMG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettledRouteMetadata {
    pub target_region: String,
    pub route_id: String,
    pub attempt: u32,
}

/// Common headers required for both unresolved and settled cross-region requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossRegionCommonHeaders {
    pub contract_version: u32,
    pub source_service: String,
    pub opc_request_id: String,
    pub entry_region: String,
    pub request_mode: RequestMode,
}

/// Parsed unresolved request context carrying routing profile constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnresolvedRequestContext {
    pub common: CrossRegionCommonHeaders,
    pub profile: RoutingProfileContext,
}

/// Parsed settled request context carrying only committed route metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettledRequestContext {
    pub common: CrossRegionCommonHeaders,
    pub route: SettledRouteMetadata,
}

/// Parsed cross-region header view for later request-mode dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CrossRegionHeaders {
    Unresolved(UnresolvedRequestContext),
    Settled(SettledRequestContext),
}

impl CrossRegionHeaders {
    /// Build a minimal parsed header view from the request mode value.
    pub fn from_request_mode(value: &str) -> CrossRegionResult<Self> {
        let request_mode = value.parse()?;
        let common = CrossRegionCommonHeaders {
            contract_version: SUPPORTED_CONTRACT_VERSION,
            source_service: "unknown".to_string(),
            opc_request_id: "unknown".to_string(),
            entry_region: "unknown".to_string(),
            request_mode,
        };

        match request_mode {
            RequestMode::Unresolved => {
                let profile = RoutingProfileContext::new(
                    vec!["unknown".to_string()],
                    vec!["unknown".to_string()],
                    FailoverPolicy::new(CrossRegionFailoverMode::Manual, 0),
                    ModalityPolicy::default(),
                )?;
                Ok(Self::Unresolved(UnresolvedRequestContext {
                    common,
                    profile,
                }))
            }
            RequestMode::Settled => Ok(Self::Settled(SettledRequestContext {
                common,
                route: SettledRouteMetadata {
                    target_region: "unknown".to_string(),
                    route_id: "unknown".to_string(),
                    attempt: 0,
                },
            })),
        }
    }

    /// Parse and validate the full cross-region header contract.
    pub fn parse(headers: &HeaderMap, platform_max_retry: u32) -> CrossRegionResult<Self> {
        let common = parse_common_headers(headers)?;

        match common.request_mode {
            RequestMode::Unresolved => {
                reject_settled_headers_on_unresolved(headers)?;
                let profile = parse_unresolved_profile(headers, platform_max_retry)?;
                Ok(Self::Unresolved(UnresolvedRequestContext {
                    common,
                    profile,
                }))
            }
            RequestMode::Settled => {
                reject_profile_headers_on_settled(headers)?;
                let route = parse_settled_metadata(headers)?;
                Ok(Self::Settled(SettledRequestContext { common, route }))
            }
        }
    }

    /// Return the parsed request mode.
    pub fn request_mode(&self) -> RequestMode {
        self.common().request_mode
    }

    /// Return common headers shared by both request modes.
    pub fn common(&self) -> &CrossRegionCommonHeaders {
        match self {
            Self::Unresolved(context) => &context.common,
            Self::Settled(context) => &context.common,
        }
    }

    /// Return unresolved routing profile context when the request mode is unresolved.
    pub fn profile(&self) -> Option<&RoutingProfileContext> {
        match self {
            Self::Unresolved(context) => Some(&context.profile),
            Self::Settled(_) => None,
        }
    }

    /// Return settled route metadata when the request mode is settled.
    pub fn settled(&self) -> Option<&SettledRouteMetadata> {
        match self {
            Self::Unresolved(_) => None,
            Self::Settled(context) => Some(&context.route),
        }
    }
}

impl CrossRegionError {
    /// Map cross-region boundary errors to the HTTP status later endpoint code should return.
    pub fn http_status(&self) -> StatusCode {
        match self {
            Self::InvalidHeader { .. } | Self::InvalidProfile { .. } => StatusCode::BAD_REQUEST,
            Self::InvalidConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::PeerNotFound { .. } | Self::InvalidPeer { .. } => StatusCode::BAD_GATEWAY,
            Self::InvalidForwardingTarget { .. } | Self::NoCandidate { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
        }
    }
}

/// Parse headers that are common to both cross-region request modes.
fn parse_common_headers(headers: &HeaderMap) -> CrossRegionResult<CrossRegionCommonHeaders> {
    let contract_version = parse_contract_version(headers)?;
    let request_mode = parse_request_mode(headers)?;
    let source_service = required_header(headers, SOURCE_SERVICE_HEADER)?;
    let opc_request_id = required_header(headers, OPC_REQUEST_ID_HEADER)?;
    let entry_region = required_header(headers, ENTRY_REGION_HEADER)?;

    Ok(CrossRegionCommonHeaders {
        contract_version,
        source_service,
        opc_request_id,
        entry_region,
        request_mode,
    })
}

/// Parse and validate the cross-region contract version header.
fn parse_contract_version(headers: &HeaderMap) -> CrossRegionResult<u32> {
    let version = required_header(headers, CONTRACT_VERSION_HEADER)?;
    let parsed = version
        .parse::<u32>()
        .map_err(|_| invalid_header(CONTRACT_VERSION_HEADER, "must be an integer"))?;
    if parsed != SUPPORTED_CONTRACT_VERSION {
        return Err(invalid_header(
            CONTRACT_VERSION_HEADER,
            &format!("must be {SUPPORTED_CONTRACT_VERSION}"),
        ));
    }
    Ok(parsed)
}

/// Parse the request-mode header into the typed enum.
fn parse_request_mode(headers: &HeaderMap) -> CrossRegionResult<RequestMode> {
    required_header(headers, REQUEST_MODE_HEADER)?.parse()
}

/// Parse the failover-mode header into the shared config enum.
fn parse_failover_mode(headers: &HeaderMap) -> CrossRegionResult<CrossRegionFailoverMode> {
    required_header(headers, FAILOVER_MODE_HEADER)?
        .parse()
        .map_err(|e: String| invalid_header(FAILOVER_MODE_HEADER, &e))
}

/// Parse x-max-retry and cap it by the platform retry limit.
fn parse_max_retry(headers: &HeaderMap, platform_max_retry: u32) -> CrossRegionResult<u32> {
    let requested = required_header(headers, MAX_RETRY_HEADER)?
        .parse::<u32>()
        .map_err(|_| invalid_header(MAX_RETRY_HEADER, "must be an integer"))?;
    Ok(requested.min(platform_max_retry))
}

/// Parse and validate the Phase 1 model-id list header.
fn parse_model_ids(headers: &HeaderMap) -> CrossRegionResult<Vec<String>> {
    let models = parse_csv_header(headers, ALLOWED_MODELS_HEADER)?;
    if models.len() != 1 {
        return Err(invalid_header(
            ALLOWED_MODELS_HEADER,
            "Phase 1 requires exactly one allowed model",
        ));
    }
    Ok(models)
}

/// Parse profile-only headers for an unresolved request.
fn parse_unresolved_profile(
    headers: &HeaderMap,
    platform_max_retry: u32,
) -> CrossRegionResult<RoutingProfileContext> {
    let allowed_regions = parse_csv_header(headers, ALLOWED_REGIONS_HEADER)?;
    let model_ids = parse_model_ids(headers)?;
    let failover_mode = parse_failover_mode(headers)?;
    let max_retry = parse_max_retry(headers, platform_max_retry)?;
    let modality = ModalityPolicy {
        input: optional_header(headers, INPUT_MODALITY_HEADER)?,
        output: optional_header(headers, OUTPUT_MODALITY_HEADER)?,
    };

    RoutingProfileContext::new(
        allowed_regions,
        model_ids,
        FailoverPolicy::new(failover_mode, max_retry),
        modality,
    )
}

/// Parse comma-separated header values into trimmed non-empty strings.
fn parse_csv_header(headers: &HeaderMap, name: &'static str) -> CrossRegionResult<Vec<String>> {
    let raw = required_header(headers, name)?;
    let values = raw
        .split(',')
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();

    if values.is_empty() || values.iter().any(String::is_empty) {
        return Err(invalid_header(
            name,
            "must contain at least one non-empty comma-separated value",
        ));
    }
    Ok(values)
}

/// Parse settled-only route metadata without requiring routing profile headers.
fn parse_settled_metadata(headers: &HeaderMap) -> CrossRegionResult<SettledRouteMetadata> {
    let target_region = required_header(headers, TARGET_REGION_HEADER)?;
    let route_id = required_header(headers, ROUTE_ID_HEADER)?;
    let attempt = required_header(headers, ATTEMPT_HEADER)?
        .parse::<u32>()
        .map_err(|_| invalid_header(ATTEMPT_HEADER, "must be an integer"))?;

    Ok(SettledRouteMetadata {
        target_region,
        route_id,
        attempt,
    })
}

/// Reject settled-only metadata on unresolved requests to avoid ambiguous routing state.
fn reject_settled_headers_on_unresolved(headers: &HeaderMap) -> CrossRegionResult<()> {
    for header in [TARGET_REGION_HEADER, ROUTE_ID_HEADER, ATTEMPT_HEADER] {
        if headers.contains_key(header) {
            return Err(invalid_header(
                header,
                "must only be present when x-request-mode is SETTLED",
            ));
        }
    }
    Ok(())
}

/// Reject profile-only metadata on settled requests to avoid duplicate routing context.
fn reject_profile_headers_on_settled(headers: &HeaderMap) -> CrossRegionResult<()> {
    for header in [
        ALLOWED_REGIONS_HEADER,
        ALLOWED_MODELS_HEADER,
        FAILOVER_MODE_HEADER,
        MAX_RETRY_HEADER,
        INPUT_MODALITY_HEADER,
        OUTPUT_MODALITY_HEADER,
    ] {
        if headers.contains_key(header) {
            return Err(invalid_header(
                header,
                "must only be present when x-request-mode is UNRESOLVED",
            ));
        }
    }
    Ok(())
}

/// Return a required header value as a trimmed string.
fn required_header(headers: &HeaderMap, name: &'static str) -> CrossRegionResult<String> {
    let value = headers
        .get(name)
        .ok_or_else(|| invalid_header(name, "is required"))?
        .to_str()
        .map_err(|_| invalid_header(name, "must be valid ASCII/UTF-8"))?
        .trim()
        .to_string();

    if value.is_empty() {
        return Err(invalid_header(name, "must not be empty"));
    }
    Ok(value)
}

/// Return an optional header value as a trimmed string.
fn optional_header(headers: &HeaderMap, name: &'static str) -> CrossRegionResult<Option<String>> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| invalid_header(name, "must be valid ASCII/UTF-8"))?
        .trim()
        .to_string();
    if value.is_empty() {
        return Err(invalid_header(name, "must not be empty when present"));
    }
    Ok(Some(value))
}

/// Build a typed invalid-header error with a stable message.
fn invalid_header(name: &'static str, reason: &str) -> CrossRegionError {
    CrossRegionError::InvalidHeader {
        reason: format!("{name} {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use http::HeaderValue;

    use super::*;

    /// Build valid unresolved headers for parser tests.
    fn unresolved_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "1");
        insert(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_UNRESOLVED);
        insert(&mut headers, ENTRY_REGION_HEADER, "us-ashburn-1");
        insert(&mut headers, SOURCE_SERVICE_HEADER, "dp-api");
        insert(&mut headers, OPC_REQUEST_ID_HEADER, "opc-request-1");
        insert(
            &mut headers,
            ALLOWED_REGIONS_HEADER,
            "us-ashburn-1, us-chicago-1",
        );
        insert(&mut headers, ALLOWED_MODELS_HEADER, "cohere.command-r-plus");
        insert(&mut headers, FAILOVER_MODE_HEADER, "AUTO");
        insert(&mut headers, MAX_RETRY_HEADER, "5");
        insert(&mut headers, INPUT_MODALITY_HEADER, "text");
        insert(&mut headers, OUTPUT_MODALITY_HEADER, "text");
        headers
    }

    /// Build valid settled headers for parser tests.
    fn settled_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "1");
        insert(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED);
        insert(&mut headers, ENTRY_REGION_HEADER, "us-ashburn-1");
        insert(&mut headers, SOURCE_SERVICE_HEADER, "smg");
        insert(&mut headers, OPC_REQUEST_ID_HEADER, "opc-request-1");
        insert(&mut headers, TARGET_REGION_HEADER, "us-chicago-1");
        insert(&mut headers, ROUTE_ID_HEADER, "route-1");
        insert(&mut headers, ATTEMPT_HEADER, "1");
        headers
    }

    /// Add a static test header value.
    fn insert(headers: &mut HeaderMap, name: &'static str, value: &'static str) {
        headers.insert(name, HeaderValue::from_static(value));
    }

    #[test]
    fn request_mode_parses_canonical_values() {
        assert_eq!(
            REQUEST_MODE_UNRESOLVED.parse::<RequestMode>(),
            Ok(RequestMode::Unresolved)
        );
        assert_eq!("settled".parse::<RequestMode>(), Ok(RequestMode::Settled));
    }

    #[test]
    fn request_mode_rejects_unknown_values() {
        let error = "LOCAL".parse::<RequestMode>().expect_err("invalid mode");

        assert!(error.to_string().contains(REQUEST_MODE_HEADER));
    }

    #[test]
    fn request_mode_serializes_as_contract_value() {
        let json = serde_json::to_string(&RequestMode::Settled).expect("serialize mode");

        assert_eq!(json, "\"SETTLED\"");
    }

    #[test]
    fn valid_unresolved_headers_parse_into_routing_context() {
        let parsed =
            CrossRegionHeaders::parse(&unresolved_headers(), 3).expect("headers should parse");
        let CrossRegionHeaders::Unresolved(context) = parsed else {
            panic!("expected unresolved context");
        };

        assert_eq!(context.common.contract_version, SUPPORTED_CONTRACT_VERSION);
        assert_eq!(context.common.request_mode, RequestMode::Unresolved);
        assert_eq!(context.common.entry_region, "us-ashburn-1");
        assert_eq!(
            context.profile.allowed_regions,
            vec!["us-ashburn-1".to_string(), "us-chicago-1".to_string()]
        );
        assert_eq!(
            context.profile.model_ids,
            vec!["cohere.command-r-plus".to_string()]
        );
        assert_eq!(
            context.profile.single_model_id().expect("single model id"),
            "cohere.command-r-plus"
        );
        assert_eq!(
            context.profile.failover_policy.failover_mode,
            CrossRegionFailoverMode::Automatic
        );
        assert_eq!(context.profile.failover_policy.max_retry, 3);
        assert_eq!(context.profile.modality.input.as_deref(), Some("text"));
    }

    #[test]
    fn missing_contract_version_is_rejected() {
        let mut headers = unresolved_headers();
        headers.remove(CONTRACT_VERSION_HEADER);

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("missing version");

        assert!(error.to_string().contains(CONTRACT_VERSION_HEADER));
        assert_eq!(error.http_status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn invalid_contract_version_is_rejected() {
        let mut headers = unresolved_headers();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "2");

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("invalid version");

        assert!(error.to_string().contains("must be 1"));
    }

    #[test]
    fn missing_entry_region_is_rejected() {
        let mut headers = unresolved_headers();
        headers.remove(ENTRY_REGION_HEADER);

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("missing entry region");

        assert!(error.to_string().contains(ENTRY_REGION_HEADER));
    }

    #[test]
    fn missing_allowed_regions_is_rejected() {
        let mut headers = unresolved_headers();
        headers.remove(ALLOWED_REGIONS_HEADER);

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("missing allowed regions");

        assert!(error.to_string().contains(ALLOWED_REGIONS_HEADER));
    }

    #[test]
    fn multiple_allowed_models_are_rejected_for_phase1() {
        let mut headers = unresolved_headers();
        insert(
            &mut headers,
            ALLOWED_MODELS_HEADER,
            "cohere.command-r-plus, meta.llama-3",
        );

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("multiple models");

        assert!(error.to_string().contains("exactly one allowed model"));
    }

    #[test]
    fn empty_allowed_models_are_rejected_for_phase1() {
        let mut headers = unresolved_headers();
        insert(&mut headers, ALLOWED_MODELS_HEADER, " ");

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("empty model");

        assert!(error.to_string().contains(ALLOWED_MODELS_HEADER));
    }

    #[test]
    fn max_retry_is_capped_by_platform_limit() {
        let parsed =
            CrossRegionHeaders::parse(&unresolved_headers(), 2).expect("headers should parse");
        let profile = parsed.profile().expect("unresolved profile");

        assert_eq!(profile.failover_policy.max_retry, 2);
    }

    #[test]
    fn valid_settled_headers_parse_without_routing_profile() {
        let parsed =
            CrossRegionHeaders::parse(&settled_headers(), 3).expect("settled headers should parse");
        let CrossRegionHeaders::Settled(context) = parsed else {
            panic!("expected settled context");
        };

        assert_eq!(context.common.request_mode, RequestMode::Settled);
        assert_eq!(context.common.entry_region, "us-ashburn-1");
        assert_eq!(context.route.target_region, "us-chicago-1");
        assert_eq!(context.route.attempt, 1);
    }

    #[test]
    fn settled_missing_route_metadata_is_rejected() {
        let mut headers = settled_headers();
        headers.remove(ROUTE_ID_HEADER);

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("missing route id");

        assert!(error.to_string().contains(ROUTE_ID_HEADER));
    }

    #[test]
    fn invalid_request_mode_is_rejected() {
        let mut headers = unresolved_headers();
        insert(&mut headers, REQUEST_MODE_HEADER, "LOCAL");

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("invalid mode");

        assert!(error.to_string().contains(REQUEST_MODE_HEADER));
    }

    #[test]
    fn unresolved_request_rejects_settled_only_metadata() {
        let mut headers = unresolved_headers();
        insert(&mut headers, ROUTE_ID_HEADER, "route-1");

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("ambiguous headers");

        assert!(error.to_string().contains("SETTLED"));
    }

    #[test]
    fn settled_request_rejects_routing_profile_headers() {
        let mut headers = settled_headers();
        insert(&mut headers, ALLOWED_MODELS_HEADER, "cohere.command-r-plus");

        let error = CrossRegionHeaders::parse(&headers, 3).expect_err("profile header");

        assert!(error.to_string().contains("UNRESOLVED"));
    }

    #[test]
    fn context_helpers_return_mode_specific_views() {
        let unresolved =
            CrossRegionHeaders::parse(&unresolved_headers(), 3).expect("unresolved parses");
        let settled = CrossRegionHeaders::parse(&settled_headers(), 3).expect("settled parses");

        assert_eq!(unresolved.request_mode(), RequestMode::Unresolved);
        assert!(unresolved.profile().is_some());
        assert!(unresolved.settled().is_none());
        assert_eq!(settled.request_mode(), RequestMode::Settled);
        assert!(settled.profile().is_none());
        assert_eq!(
            settled
                .settled()
                .expect("settled route metadata")
                .target_region,
            "us-chicago-1"
        );
    }
}
