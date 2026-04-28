use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::{CrossRegionError, CrossRegionResult};

/// Header used by DP-API and remote SMG to indicate routing state.
pub const REQUEST_MODE_HEADER: &str = "x-request-mode";

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

/// Minimal parsed cross-region header view for later request context parsing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossRegionHeaders {
    pub request_mode: RequestMode,
}

impl CrossRegionHeaders {
    /// Build a parsed header view from the request mode value.
    pub fn from_request_mode(value: &str) -> CrossRegionResult<Self> {
        Ok(Self {
            request_mode: value.parse()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
