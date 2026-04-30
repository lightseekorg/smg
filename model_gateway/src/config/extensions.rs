//! Generic extension configuration schema.
//!
//! Per spec §Config: the `extensions:` YAML block carries a list of
//! `kind`/`config` pairs. SMG core parses only the generic shape; each
//! extension crate parses its own `config: serde_yml::Value` payload.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExtensionsConfig {
    #[serde(default)]
    pub items: Vec<ExtensionSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExtensionSpec {
    pub kind: String,
    #[serde(flatten)]
    pub config: serde_yml::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum ExtensionConfigError {
    #[error("unknown extension kind '{0}'")]
    Unknown(String),

    #[error("extension '{kind}' requires building with --features {feature}")]
    FeatureGated { kind: String, feature: &'static str },

    #[error("extension '{kind}' build failed: {source}")]
    BuildFailed {
        kind: String,
        #[source]
        source: anyhow::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_extensions_parses() {
        let yaml = "items: []";
        let cfg: ExtensionsConfig = serde_yml::from_str(yaml).unwrap();
        assert!(cfg.items.is_empty());
    }

    #[test]
    fn missing_extensions_field_uses_default() {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(default)]
            extensions: ExtensionsConfig,
        }
        let yaml = "{}";
        let w: Wrapper = serde_yml::from_str(yaml).unwrap();
        assert!(w.extensions.items.is_empty());
    }

    #[test]
    fn extension_spec_captures_arbitrary_config() {
        let yaml = r#"
items:
  - kind: my-extension
    runtime_enabled: true
    nested:
      key: value
"#;
        let cfg: ExtensionsConfig = serde_yml::from_str(yaml).unwrap();
        assert_eq!(cfg.items.len(), 1);
        assert_eq!(cfg.items[0].kind, "my-extension");
        let val: &serde_yml::Value = &cfg.items[0].config;
        assert!(val.get("runtime_enabled").is_some());
        assert!(val.get("nested").is_some());
    }
}
