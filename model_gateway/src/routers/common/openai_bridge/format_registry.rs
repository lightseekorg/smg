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
/// composes the session's name map with `FormatRegistry::lookup`. Reuses the
/// `QualifiedToolName` returned by `qualified_name_for_exposed` rather than
/// rebuilding one, so we pay the two `Arc<str>` allocations once instead of
/// twice per call.
pub fn lookup_tool_format(
    session: &smg_mcp::McpToolSession<'_>,
    registry: &FormatRegistry,
    exposed_name: &str,
) -> ResponseFormat {
    let Some(qn) = session.qualified_name_for_exposed(exposed_name) else {
        return ResponseFormat::Passthrough;
    };
    registry.lookup(&qn)
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

    fn remove(&self, qualified: &QualifiedToolName) {
        self.formats.remove(qualified);
    }

    /// Populate from a server config: per-tool overrides + builtin defaults.
    ///
    /// Safe to call repeatedly. Each affected key is unconditionally rewritten
    /// (or removed for an explicit `Some(Passthrough)` downgrade) so a later
    /// config that demotes a tool from a hosted format back to `Passthrough`
    /// does not leave the previous entry behind in the map.
    ///
    /// Mirrors `McpOrchestrator::apply_tool_configs` *and* the dispatch shape
    /// of `McpToolSession`. Two production lookup paths must resolve to the
    /// same format:
    /// - via the alias key `("alias", alias_name)` — what the session exposes
    ///   when `collect_visible_mcp_tools` replaces the direct entry with its
    ///   alias entry (see `crates/mcp/src/core/session.rs:565-571`).
    /// - via the underlying `(server_key, tool_name)` — what direct dispatch
    ///   uses when no alias hides the tool.
    ///
    /// So when a tool has an alias the format is mirrored on **both** keys.
    /// Aliased builtin tools get the same treatment, and only when the
    /// per-tool stanza omits `response_format` entirely (`None`) is the
    /// builtin default applied; an explicit `Some(Passthrough)` is the
    /// documented escape hatch for opting out of the hosted shape.
    pub fn populate_from_server_config(&self, config: &McpServerConfig) {
        if let Some(tools) = &config.tools {
            for (tool_name, tool_config) in tools {
                let direct_key = QualifiedToolName::new(&config.name, tool_name);
                let alias_key = tool_config
                    .alias
                    .as_deref()
                    .map(|alias| QualifiedToolName::new(ALIAS_SERVER_KEY, alias));

                let Some(format_config) = tool_config.response_format else {
                    // `response_format` omitted: defer to the builtin-default
                    // pass below. Don't touch existing entries here so a
                    // direct-dispatch override placed by an earlier loop
                    // iteration survives.
                    continue;
                };
                let format: ResponseFormat = format_config.into();
                if format == ResponseFormat::Passthrough {
                    // Explicit downgrade — clear any prior hosted-format entry
                    // on every key the production lookup might consult.
                    self.remove(&direct_key);
                    if let Some(alias_key) = &alias_key {
                        self.remove(alias_key);
                    }
                    continue;
                }

                // Mirror the format on every key production might query.
                // Also write the direct key so that direct dispatch (which
                // does not go through the alias rewrite) still gets the
                // hosted shape.
                self.insert(direct_key, format);
                if let Some(alias_key) = alias_key {
                    self.insert(alias_key, format);
                }
            }
        }

        if let (Some(builtin_type), Some(tool_name)) =
            (&config.builtin_type, &config.builtin_tool_name)
        {
            let stanza = config.tools.as_ref().and_then(|tools| tools.get(tool_name));
            let has_explicit_format = stanza.is_some_and(|cfg| cfg.response_format.is_some());
            if !has_explicit_format {
                let format: ResponseFormat = builtin_type.response_format().into();
                self.insert(QualifiedToolName::new(&config.name, tool_name), format);
                // `collect_visible_mcp_tools` exposes the alias entry instead
                // of the direct entry, so the production lookup queries
                // `("alias", alias)`. Without this mirror an alias-only stanza
                // on a builtin tool silently degrades to `Passthrough`.
                if let Some(alias) = stanza.and_then(|cfg| cfg.alias.as_deref()) {
                    self.insert(QualifiedToolName::new(ALIAS_SERVER_KEY, alias), format);
                }
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
    fn alias_format_mirrored_on_both_keys() {
        // The session lookup path goes through `("alias", alias_name)` because
        // `collect_visible_mcp_tools` replaces the direct entry with its alias
        // entry. Direct dispatch (no alias rewrite) still queries
        // `(server_key, tool_name)`, so both keys must carry the format.
        let mut tools = HashMap::new();
        tools.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::WebSearchCall),
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
            ResponseFormat::WebSearchCall,
            "direct (server_key, tool_name) entry must also carry the format \
             so direct dispatch resolves the same shape as alias dispatch"
        );
    }

    #[test]
    fn non_aliased_tool_stores_format_under_server_tool_pair() {
        let mut tools = HashMap::new();
        tools.insert(
            "search".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::WebSearchCall),
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
                response_format: Some(ResponseFormatConfig::Passthrough),
                arg_mapping: None,
            },
        );
        let mut cfg = server("search");
        cfg.tools = Some(tools);
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        // Explicit Some(Passthrough) override means "no entry inserted" AND
        // the builtin default is NOT applied on top.
        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::Passthrough
        );
    }

    #[test]
    fn alias_only_stanza_preserves_builtin_default_on_both_keys() {
        // Regression: a per-tool stanza that only aliases a builtin tool
        // (or only sets arg_mapping) used to suppress the builtin default,
        // collapsing the hosted format to plain mcp_call. The crucial
        // production path is the alias key — `collect_visible_mcp_tools`
        // exposes the alias entry, so `lookup_tool_format(session, …,
        // "web_search")` resolves to `("alias", "web_search")`. If the
        // builtin default only landed at the direct key, dispatch silently
        // degrades to `Passthrough`.
        let mut tools = HashMap::new();
        tools.insert(
            "do_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: None,
                arg_mapping: None,
            },
        );
        let mut cfg = server("search");
        cfg.tools = Some(tools);
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        // Alias key — what production session lookup actually hits.
        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
            "alias-only stanza on a builtin must mirror the hosted format on \
             the alias key — that is the key production resolves through"
        );
        // Direct key — what direct dispatch hits.
        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::WebSearchCall,
            "direct (server, tool) lookup must still resolve the hosted format"
        );
    }

    #[test]
    fn explicit_passthrough_downgrade_clears_prior_hosted_entry() {
        // `populate_from_server_config` is called every time a server
        // (re)registers, so a config that flips a tool from a hosted format
        // back to `Passthrough` must clear the stale entry — otherwise the
        // registry keeps transforming outputs as the old hosted type.
        let r = FormatRegistry::new();

        let mut hosted = HashMap::new();
        hosted.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: None,
            },
        );
        let mut hosted_cfg = server("brave");
        hosted_cfg.tools = Some(hosted);
        r.populate_from_server_config(&hosted_cfg);
        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
            "precondition: alias key carries the hosted format"
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::WebSearchCall,
            "precondition: direct key carries the hosted format"
        );

        let mut downgraded = HashMap::new();
        downgraded.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::Passthrough),
                arg_mapping: None,
            },
        );
        let mut downgraded_cfg = server("brave");
        downgraded_cfg.tools = Some(downgraded);
        r.populate_from_server_config(&downgraded_cfg);

        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::Passthrough,
            "explicit Passthrough must clear the previous alias entry"
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::Passthrough,
            "explicit Passthrough must clear the previous direct entry"
        );
    }
}
