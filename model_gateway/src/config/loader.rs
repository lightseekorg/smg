//! Config file loading for YAML, JSON, and .env formats.

use std::path::Path;

use super::{ConfigError, ConfigResult, RouterConfig};

/// Supported config file formats, detected from file extension.
enum ConfigFormat {
    Yaml,
    Json,
    Env,
}

impl ConfigFormat {
    fn detect(path: &Path) -> ConfigResult<Self> {
        // Handle `.env` files (no extension, filename is `.env`)
        if path.file_name().is_some_and(|name| name == ".env") {
            return Ok(Self::Env);
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some("yaml" | "yml") => Ok(Self::Yaml),
            Some("json") => Ok(Self::Json),
            Some("env") => Ok(Self::Env),
            Some(ext) => Err(ConfigError::ValidationFailed {
                reason: format!(
                    "Unsupported config file extension '.{ext}'. Use .yaml, .yml, .json, or .env"
                ),
            }),
            None => Err(ConfigError::ValidationFailed {
                reason: "Config file must have an extension (.yaml, .yml, .json, or .env)"
                    .to_string(),
            }),
        }
    }
}

/// Load a YAML or JSON config file into [`RouterConfig`].
///
/// For `.env` files, use [`load_env_file`] instead — those populate
/// environment variables rather than producing a `RouterConfig`.
pub fn load_config_file(path: &str) -> ConfigResult<RouterConfig> {
    let p = Path::new(path);

    if !p.exists() {
        return Err(ConfigError::ValidationFailed {
            reason: format!("Config file not found: {path}"),
        });
    }

    match ConfigFormat::detect(p)? {
        ConfigFormat::Yaml => load_yaml(p),
        ConfigFormat::Json => load_json(p),
        ConfigFormat::Env => Err(ConfigError::ValidationFailed {
            reason: "Use load_env_file() for .env files".to_string(),
        }),
    }
}

/// Load a `.env` file into the process environment.
///
/// Values become available to clap's `env = "..."` attribute parsing.
pub fn load_env_file(path: &str) -> ConfigResult<()> {
    dotenvy::from_path(Path::new(path)).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to load .env file '{path}': {e}"),
    })
}

/// Returns `true` if `path` is a `.env` file (by extension or filename).
pub fn is_env_file(path: &str) -> bool {
    let p = Path::new(path);
    // Match files with `.env` extension (e.g., `config.env`)
    // or files named `.env` (which have no extension in Rust's Path)
    p.extension().is_some_and(|ext| ext == "env")
        || p.file_name().is_some_and(|name| name == ".env")
}

/// Serialize a [`RouterConfig`] to YAML for `--dump-config`.
pub fn dump_config_yaml(config: &RouterConfig) -> ConfigResult<String> {
    serde_yaml::to_string(config).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to serialize config to YAML: {e}"),
    })
}

fn load_yaml(path: &Path) -> ConfigResult<RouterConfig> {
    let contents = read_file(path)?;
    serde_yaml::from_str(&contents).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to parse YAML config '{}': {e}", path.display()),
    })
}

fn load_json(path: &Path) -> ConfigResult<RouterConfig> {
    let contents = read_file(path)?;
    serde_json::from_str(&contents).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to parse JSON config '{}': {e}", path.display()),
    })
}

fn read_file(path: &Path) -> ConfigResult<String> {
    std::fs::read_to_string(path).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to read config file '{}': {e}", path.display()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_roundtrip() {
        let config = RouterConfig::default();
        let yaml = dump_config_yaml(&config).expect("serialize");
        let parsed: RouterConfig = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(config.host, parsed.host);
        assert_eq!(config.port, parsed.port);
    }

    #[test]
    fn partial_yaml_uses_defaults() {
        let yaml = r#"
host: "10.0.0.1"
port: 9000
"#;
        let config: RouterConfig = serde_yaml::from_str(yaml).expect("deserialize partial");
        assert_eq!(config.host, "10.0.0.1");
        assert_eq!(config.port, 9000);
        // Non-specified fields get defaults
        assert!(!config.dp_aware);
        assert!(!config.enable_igw);
    }

    #[test]
    fn json_loading() {
        let json = r#"{"host": "127.0.0.1", "port": 8080}"#;
        let config: RouterConfig = serde_json::from_str(json).expect("deserialize json");
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn format_detection() {
        assert!(matches!(
            ConfigFormat::detect(Path::new("config.yaml")),
            Ok(ConfigFormat::Yaml)
        ));
        assert!(matches!(
            ConfigFormat::detect(Path::new("config.yml")),
            Ok(ConfigFormat::Yaml)
        ));
        assert!(matches!(
            ConfigFormat::detect(Path::new("config.json")),
            Ok(ConfigFormat::Json)
        ));
        assert!(matches!(
            ConfigFormat::detect(Path::new("config.env")),
            Ok(ConfigFormat::Env)
        ));
        assert!(matches!(
            ConfigFormat::detect(Path::new(".env")),
            Ok(ConfigFormat::Env)
        ));
        assert!(ConfigFormat::detect(Path::new("config.toml")).is_err());
        assert!(ConfigFormat::detect(Path::new("config")).is_err());
    }

    #[test]
    fn missing_file_errors() {
        let result = load_config_file("/nonexistent/path.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn is_env_file_detection() {
        assert!(is_env_file("config.env"));
        assert!(is_env_file("/path/to/.env")); // extension is "env"
        assert!(!is_env_file("config.yaml"));
        assert!(!is_env_file("config.json"));
    }
}
