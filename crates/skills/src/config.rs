use std::{fmt, str::FromStr};

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use smg_blob_storage::{
    BlobCacheConfig as SkillsCacheConfig, BlobStoreBackend as SkillsBlobStoreBackend,
    BlobStoreConfig as SkillsBlobStoreConfig,
};

const MEBIBYTE: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillsMissingMcpPolicy {
    #[default]
    Warn,
    Reject,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillsResolutionMode {
    ToolLoop,
    Eager,
    #[default]
    Auto,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillsExecutionAsyncMode {
    #[default]
    PauseTurn,
    PollAndWait,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillsAdminOperation {
    CreateAnyTenant,
    ReadAnyTenant,
    UpdateAnyTenant,
    DeleteAnyTenant,
    CreateAlias,
    ExecuteMigration,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillsRetentionMode {
    #[default]
    Standard,
    NoContentLogs,
    Zdr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillsBudgetLimit {
    Tokens(u32),
    Unlimited,
}

impl Default for SkillsBudgetLimit {
    fn default() -> Self {
        Self::Tokens(16_000)
    }
}

impl FromStr for SkillsBudgetLimit {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("unlimited") {
            return Ok(Self::Unlimited);
        }

        value
            .parse::<u32>()
            .map(Self::Tokens)
            .map_err(|_| "expected an integer token count or 'unlimited'".to_string())
    }
}

impl Serialize for SkillsBudgetLimit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Tokens(tokens) => serializer.serialize_u32(*tokens),
            Self::Unlimited => serializer.serialize_str("unlimited"),
        }
    }
}

impl<'de> Deserialize<'de> for SkillsBudgetLimit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SkillsBudgetLimitVisitor;

        impl<'de> Visitor<'de> for SkillsBudgetLimitVisitor {
            type Value = SkillsBudgetLimit;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an integer token count or the string 'unlimited'")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let tokens = u32::try_from(value)
                    .map_err(|_| E::custom("token count must fit within u32"))?;
                Ok(SkillsBudgetLimit::Tokens(tokens))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value < 0 {
                    return Err(E::custom("token count must be non-negative"));
                }

                self.visit_u64(value as u64)
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                SkillsBudgetLimit::from_str(value).map_err(E::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(SkillsBudgetLimitVisitor)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsConfig {
    pub tenancy: SkillsTenancyConfig,
    pub admin: SkillsAdminConfig,
    pub blob_store: SkillsBlobStoreConfig,
    pub cache: SkillsCacheConfig,
    pub max_upload_size_mb: usize,
    pub max_files_per_version: usize,
    pub max_file_size_mb: usize,
    pub tenant_storage_quota_mb: usize,
    pub tenant_skill_count_quota: usize,
    pub rate_limits: SkillsRateLimitsConfig,
    pub dependencies: SkillsDependenciesConfig,
    pub instruction_budget: SkillsInstructionBudgetConfig,
    pub resolution_mode: SkillsResolutionMode,
    pub tool_loop: SkillsToolLoopConfig,
    pub max_skills_per_request: usize,
    pub execution: SkillsExecutionConfig,
    pub retention: SkillsRetentionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkillUploadLimits {
    pub max_upload_size_bytes: usize,
    pub max_files_per_version: usize,
    pub max_file_size_bytes: usize,
}

impl SkillUploadLimits {
    pub fn from_config(config: &SkillsConfig) -> Result<Self, String> {
        if config.max_upload_size_mb == 0 {
            return Err("skills.max_upload_size_mb must be greater than 0".to_string());
        }
        if config.max_file_size_mb == 0 {
            return Err("skills.max_file_size_mb must be greater than 0".to_string());
        }
        if config.max_file_size_mb > config.max_upload_size_mb {
            return Err(
                "skills.max_file_size_mb must be less than or equal to skills.max_upload_size_mb"
                    .to_string(),
            );
        }
        if config.max_files_per_version == 0 {
            return Err("skills.max_files_per_version must be greater than 0".to_string());
        }

        Ok(Self {
            max_upload_size_bytes: mb_to_bytes(
                config.max_upload_size_mb,
                "skills.max_upload_size_mb",
            )?,
            max_files_per_version: config.max_files_per_version,
            max_file_size_bytes: mb_to_bytes(config.max_file_size_mb, "skills.max_file_size_mb")?,
        })
    }
}

impl Default for SkillUploadLimits {
    fn default() -> Self {
        Self {
            max_upload_size_bytes: 30 * MEBIBYTE,
            max_files_per_version: 500,
            max_file_size_bytes: 25 * MEBIBYTE,
        }
    }
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            tenancy: SkillsTenancyConfig::default(),
            admin: SkillsAdminConfig::default(),
            blob_store: SkillsBlobStoreConfig::default(),
            cache: SkillsCacheConfig::default(),
            max_upload_size_mb: 30,
            max_files_per_version: 500,
            max_file_size_mb: 25,
            tenant_storage_quota_mb: 1024,
            tenant_skill_count_quota: 1000,
            rate_limits: SkillsRateLimitsConfig::default(),
            dependencies: SkillsDependenciesConfig::default(),
            instruction_budget: SkillsInstructionBudgetConfig::default(),
            resolution_mode: SkillsResolutionMode::Auto,
            tool_loop: SkillsToolLoopConfig::default(),
            max_skills_per_request: 8,
            execution: SkillsExecutionConfig::default(),
            retention: SkillsRetentionConfig::default(),
        }
    }
}

fn mb_to_bytes(value_mb: usize, field: &'static str) -> Result<usize, String> {
    value_mb
        .checked_mul(MEBIBYTE)
        .ok_or_else(|| format!("{field} is too large to convert to bytes"))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsTenancyConfig {
    pub trust_external_tenant_header: bool,
    pub external_tenant_header_name: String,
}

impl Default for SkillsTenancyConfig {
    fn default() -> Self {
        Self {
            trust_external_tenant_header: false,
            external_tenant_header_name: "X-SMG-Tenant-Id".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsAdminConfig {
    pub enabled: bool,
    pub allowed_operations: Vec<SkillsAdminOperation>,
}

impl Default for SkillsAdminConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_operations: vec![
                SkillsAdminOperation::CreateAnyTenant,
                SkillsAdminOperation::ReadAnyTenant,
                SkillsAdminOperation::UpdateAnyTenant,
                SkillsAdminOperation::DeleteAnyTenant,
                SkillsAdminOperation::CreateAlias,
                SkillsAdminOperation::ExecuteMigration,
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsRateLimitsConfig {
    pub create_per_min: u32,
    pub create_per_day: u32,
    pub delete_per_min: u32,
    pub list_per_min: u32,
}

impl Default for SkillsRateLimitsConfig {
    fn default() -> Self {
        Self {
            create_per_min: 10,
            create_per_day: 300,
            delete_per_min: 30,
            list_per_min: 120,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsDependenciesConfig {
    pub missing_mcp_policy: SkillsMissingMcpPolicy,
}

impl Default for SkillsDependenciesConfig {
    fn default() -> Self {
        Self {
            missing_mcp_policy: SkillsMissingMcpPolicy::Warn,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsInstructionBudgetConfig {
    pub per_skill_tokens: u32,
    pub per_request_tokens: SkillsBudgetLimit,
}

impl Default for SkillsInstructionBudgetConfig {
    fn default() -> Self {
        Self {
            per_skill_tokens: 4000,
            per_request_tokens: SkillsBudgetLimit::Tokens(16_000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsToolLoopConfig {
    pub require_tool_support: bool,
    pub max_steps: u32,
    pub max_seconds: u64,
    pub max_cumulative_tokens: u64,
    pub heartbeat_secs: u64,
    pub max_result_bytes: usize,
}

impl Default for SkillsToolLoopConfig {
    fn default() -> Self {
        Self {
            require_tool_support: true,
            max_steps: 8,
            max_seconds: 300,
            max_cumulative_tokens: 200_000,
            heartbeat_secs: 2,
            max_result_bytes: 262_144,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsExecutionModeOverrides {
    pub messages: SkillsExecutionAsyncMode,
    pub responses: SkillsExecutionAsyncMode,
}

impl Default for SkillsExecutionModeOverrides {
    fn default() -> Self {
        Self {
            messages: SkillsExecutionAsyncMode::PauseTurn,
            responses: SkillsExecutionAsyncMode::PauseTurn,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsExecutionConfig {
    pub executor_url: Option<String>,
    pub executor_api_key: Option<String>,
    pub timeout_secs: u64,
    pub async_mode: SkillsExecutionAsyncMode,
    pub max_wait_secs: u64,
    pub max_poll_attempts: u32,
    pub poll_interval_ms: u64,
    pub connect_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub retry_max: u32,
    pub retry_base_ms: u64,
    pub retry_max_ms: u64,
    pub bundle_token_ttl_secs: u64,
    pub enable_streaming: bool,
    pub async_mode_overrides: SkillsExecutionModeOverrides,
    pub cancel_endpoint_path: String,
    pub cancel_fire_and_forget_timeout_ms: u64,
    pub max_output_file_bytes: usize,
    pub max_output_bytes: usize,
}

impl Default for SkillsExecutionConfig {
    fn default() -> Self {
        Self {
            executor_url: None,
            executor_api_key: None,
            timeout_secs: 30,
            async_mode: SkillsExecutionAsyncMode::PauseTurn,
            max_wait_secs: 60,
            max_poll_attempts: 60,
            poll_interval_ms: 1000,
            connect_timeout_ms: 2000,
            read_timeout_ms: 30_000,
            retry_max: 3,
            retry_base_ms: 500,
            retry_max_ms: 8000,
            bundle_token_ttl_secs: 600,
            enable_streaming: true,
            async_mode_overrides: SkillsExecutionModeOverrides::default(),
            cancel_endpoint_path: "/cancel".to_string(),
            cancel_fire_and_forget_timeout_ms: 500,
            max_output_file_bytes: 65_536,
            max_output_bytes: 131_072,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsRetentionConfig {
    pub default_mode: SkillsRetentionMode,
    pub allow_per_tenant_override: bool,
    pub zdr: SkillsZdrConfig,
}

impl Default for SkillsRetentionConfig {
    fn default() -> Self {
        Self {
            default_mode: SkillsRetentionMode::Standard,
            allow_per_tenant_override: true,
            zdr: SkillsZdrConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SkillsZdrConfig {
    pub response_state_ttl_hours: u64,
    pub require_byok: bool,
}

impl Default for SkillsZdrConfig {
    fn default() -> Self {
        Self {
            response_state_ttl_hours: 1,
            require_byok: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SkillsBudgetLimit, SkillsConfig, SkillsResolutionMode};

    #[test]
    fn skills_config_defaults_are_safe() {
        let skills = SkillsConfig::default();
        assert_eq!(skills.resolution_mode, SkillsResolutionMode::Auto);
        assert_eq!(skills.max_skills_per_request, 8);
        assert_eq!(skills.cache.max_size_mb, 0);
    }

    #[test]
    fn skills_budget_limit_round_trips_unlimited() {
        let json = "\"unlimited\"";
        let parsed: SkillsBudgetLimit = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, SkillsBudgetLimit::Unlimited);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    }

    #[test]
    fn partial_skills_config_uses_nested_defaults() {
        let parsed: SkillsConfig = serde_json::from_str(
            r#"{
                "execution": {
                    "executor_url": "http://executor.internal"
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            parsed.execution.executor_url.as_deref(),
            Some("http://executor.internal")
        );
        assert_eq!(parsed.tool_loop.max_steps, 8);
        assert_eq!(parsed.cache.max_size_mb, 0);
    }
}
