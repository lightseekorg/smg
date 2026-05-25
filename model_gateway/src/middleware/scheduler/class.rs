//! Priority class + `X-SMG-Priority` header parser.

/// HTTP request header that conveys the desired priority class.
///
/// Case-insensitive; unknown values degrade to [`Class::Default`].
pub const PRIORITY_HEADER: &str = "x-smg-priority";

/// Service class assigned to an inbound request.
///
/// Numeric values are load-bearing: `Ord` is derived so the tenant clamp is
/// `std::cmp::min(header_class, max_class)`, and `repr(u8)` lets the scheduler
/// pack per-class inflight counts into a single `AtomicU64`. Serde encoding is
/// lowercase so YAML files use `system`/`interactive`/`default`/`bulk`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Class {
    /// Background / batch jobs. Lowest priority; preemptible by everyone else.
    Bulk = 0,
    /// Unlabeled traffic. Middle of the road; what every request gets when no
    /// header is set.
    Default = 1,
    /// Latency-sensitive traffic (chat completions, autocomplete).
    Interactive = 2,
    /// Internal control-plane traffic. Highest priority; never preempted by
    /// external clients (the tenant clamp prevents any external tenant from
    /// landing here in practice).
    System = 3,
}

impl Class {
    /// All four variants in ascending priority order.
    pub const ALL: [Class; 4] = [Self::Bulk, Self::Default, Self::Interactive, Self::System];

    /// Parse a header value into a class. Case-insensitive, whitespace-tolerant.
    /// Unknown values (including the empty string) map to [`Class::Default`]
    /// — admission shouldn't fail because of a typo in a non-essential header.
    pub fn parse_header(value: &str) -> Class {
        match value.trim().to_ascii_lowercase().as_str() {
            "system" => Self::System,
            "interactive" => Self::Interactive,
            "bulk" => Self::Bulk,
            _ => Self::Default,
        }
    }

    /// Lowercase variant name. Used as a metrics label and structured-log field
    /// only; never serialized back over the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bulk => "bulk",
            Self::Default => "default",
            Self::Interactive => "interactive",
            Self::System => "system",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_known_values() {
        assert_eq!(Class::parse_header("system"), Class::System);
        assert_eq!(Class::parse_header("interactive"), Class::Interactive);
        assert_eq!(Class::parse_header("default"), Class::Default);
        assert_eq!(Class::parse_header("bulk"), Class::Bulk);
    }

    #[test]
    fn test_parse_header_is_case_insensitive() {
        assert_eq!(Class::parse_header("Bulk"), Class::Bulk);
        assert_eq!(Class::parse_header("INTERACTIVE"), Class::Interactive);
        assert_eq!(Class::parse_header("SyStEm"), Class::System);
    }

    #[test]
    fn test_parse_header_unknown_defaults_to_default() {
        assert_eq!(Class::parse_header("urgent"), Class::Default);
        assert_eq!(Class::parse_header(""), Class::Default);
        assert_eq!(Class::parse_header("123"), Class::Default);
    }

    #[test]
    fn test_parse_header_tolerates_whitespace() {
        assert_eq!(Class::parse_header("  bulk  "), Class::Bulk);
        assert_eq!(Class::parse_header("\tinteractive\n"), Class::Interactive);
    }

    #[test]
    fn test_ord_min_implements_tenant_clamp() {
        // Tenant clamp = min(header_class, max_class).
        assert_eq!(
            std::cmp::min(Class::Interactive, Class::Default),
            Class::Default
        );
        assert_eq!(
            std::cmp::min(Class::System, Class::Interactive),
            Class::Interactive
        );
        assert_eq!(std::cmp::min(Class::Bulk, Class::System), Class::Bulk);
    }

    #[test]
    fn test_all_is_ascending() {
        assert_eq!(
            Class::ALL,
            [
                Class::Bulk,
                Class::Default,
                Class::Interactive,
                Class::System
            ]
        );
        // Ord order matches numeric order.
        assert!(Class::Bulk < Class::Default);
        assert!(Class::Default < Class::Interactive);
        assert!(Class::Interactive < Class::System);
    }

    #[test]
    fn test_as_str_returns_lowercase_variant_name() {
        assert_eq!(Class::Bulk.as_str(), "bulk");
        assert_eq!(Class::Default.as_str(), "default");
        assert_eq!(Class::Interactive.as_str(), "interactive");
        assert_eq!(Class::System.as_str(), "system");
    }

    #[test]
    fn test_priority_header_constant() {
        assert_eq!(PRIORITY_HEADER, "x-smg-priority");
    }
}
