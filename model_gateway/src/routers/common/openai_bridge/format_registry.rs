//! Side-map of `QualifiedToolName → ResponseFormat`, populated when MCP
//! servers register and queried by router code at request time.

use std::sync::Arc;

use dashmap::DashMap;
use smg_mcp::{inventory::ALIAS_SERVER_KEY, McpServerConfig, QualifiedToolName};

use super::ResponseFormat;

/// Resolve an exposed tool name's `ResponseFormat` via the session's name map
/// and the registry. Returns `Passthrough` for unknown tools.
///
/// Lives next to `FormatRegistry` because it's a thin lookup helper that
/// composes session's name map with `FormatRegistry::lookup_by_names`,
/// avoiding the `Arc<str>` allocations of `QualifiedToolName::new`.
pub fn lookup_tool_format(
    session: &smg_mcp::McpToolSession<'_>,
    registry: &FormatRegistry,
    exposed_name: &str,
) -> ResponseFormat {
    let Some(qn) = session.qualified_name_for_exposed(exposed_name) else {
        return ResponseFormat::Passthrough;
    };
    registry.lookup_by_names(qn.server_key(), qn.tool_name())
}

#[derive(Default, Debug, Clone)]
pub struct FormatRegistry {
    formats: Arc<DashMap<QualifiedToolName, ResponseFormat>>,
}

impl FormatRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lookup(&self, qualified: &QualifiedToolName) -> ResponseFormat {
        self.formats
            .get(qualified)
            .map(|r| *r.value())
            .unwrap_or(ResponseFormat::Passthrough)
    }

    pub fn lookup_by_names(&self, server_key: &str, tool_name: &str) -> ResponseFormat {
        self.lookup(&QualifiedToolName::new(server_key, tool_name))
    }

    fn insert(&self, qualified: QualifiedToolName, format: ResponseFormat) {
        self.formats.insert(qualified, format);
    }

    /// Populate from a server config: per-tool overrides + builtin defaults.
    /// Safe to call repeatedly — entries are overwritten.
    ///
    /// Mirrors `McpOrchestrator::apply_tool_configs`:
    /// - When a tool has an `alias`, the format is attached **only** to the
    ///   alias entry (under `("alias", alias_name)`), matching the orchestrator's
    ///   `register_alias` qualified-name shape. The underlying `(server, tool)`
    ///   stays at the `Passthrough` default so direct calls aren't transformed.
    /// - When a tool has no alias but a non-default format, attach to
    ///   `(server, tool)` directly.
    pub fn populate_from_server_config(&self, config: &McpServerConfig) {
        if let Some(tools) = &config.tools {
            for (tool_name, tool_config) in tools {
                let format: ResponseFormat = tool_config.response_format.into();
                if format == ResponseFormat::Passthrough {
                    continue;
                }
                if let Some(alias) = &tool_config.alias {
                    self.insert(QualifiedToolName::new(ALIAS_SERVER_KEY, alias), format);
                } else {
                    self.insert(QualifiedToolName::new(&config.name, tool_name), format);
                }
            }
        }

        if let (Some(builtin_type), Some(tool_name)) =
            (&config.builtin_type, &config.builtin_tool_name)
        {
            let has_explicit = config
                .tools
                .as_ref()
                .is_some_and(|tools| tools.contains_key(tool_name));
            if !has_explicit {
                let format: ResponseFormat = builtin_type.response_format().into();
                self.insert(QualifiedToolName::new(&config.name, tool_name), format);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smg_mcp::{
        BuiltinToolType, McpServerConfig, McpTransport, ResponseFormatConfig, ToolConfig,
    };

    use super::*;

    fn server(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Streamable {
                url: "http://x".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: None,
            builtin_tool_name: None,
            internal: false,
        }
    }

    #[test]
    fn lookup_unknown_returns_passthrough() {
        let r = FormatRegistry::new();
        assert_eq!(
            r.lookup_by_names("any", "tool"),
            ResponseFormat::Passthrough
        );
    }

    #[test]
    fn alias_format_stored_under_alias_server_key() {
        // Mirrors orchestrator::register_alias which uses
        // QualifiedToolName::new("alias", alias_name).
        let mut tools = HashMap::new();
        tools.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: ResponseFormatConfig::WebSearchCall,
                arg_mapping: None,
            },
        );
        let mut cfg = server("brave");
        cfg.tools = Some(tools);

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
            "alias entry must use the literal `alias` server_key prefix"
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::Passthrough,
            "underlying tool entry must NOT receive the format when an alias exists"
        );
    }

    #[test]
    fn non_aliased_tool_stores_format_under_server_tool_pair() {
        let mut tools = HashMap::new();
        tools.insert(
            "search".to_string(),
            ToolConfig {
                alias: None,
                response_format: ResponseFormatConfig::WebSearchCall,
                arg_mapping: None,
            },
        );
        let mut cfg = server("brave");
        cfg.tools = Some(tools);

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("brave", "search"),
            ResponseFormat::WebSearchCall
        );
    }

    #[test]
    fn builtin_default_applies_when_no_explicit_tool_config() {
        let mut cfg = server("search");
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::WebSearchCall
        );
    }

    #[test]
    fn explicit_per_tool_override_wins_over_builtin_default() {
        let mut tools = HashMap::new();
        tools.insert(
            "do_search".to_string(),
            ToolConfig {
                alias: None,
                // Explicit override differs from the builtin default.
                response_format: ResponseFormatConfig::Passthrough,
                arg_mapping: None,
            },
        );
        let mut cfg = server("search");
        cfg.tools = Some(tools);
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        // Explicit Passthrough override means "no entry inserted"; lookup
        // falls back to Passthrough — same result, but importantly the
        // builtin default is NOT applied on top.
        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::Passthrough
        );
    }
}
