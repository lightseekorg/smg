use std::{sync::Arc, time::Duration};

use chrono::DateTime;
use hmac::{Hmac, KeyInit, Mac};
use reqwest::Url;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

type HmacSha256 = Hmac<Sha256>;
const AWS_METADATA_HTTP_TIMEOUT_SECS: u64 = 5;
/// Refresh temporary credentials this many seconds before their expiry.
const CREDENTIAL_REFRESH_WINDOW_SECS: i64 = 300;

#[derive(Clone)]
pub(crate) struct AwsSigner {
    region: String,
    service: String,
    credentials: Arc<RwLock<Option<CachedCredentials>>>,
    http_client: reqwest::Client,
}

pub(crate) struct SignedHeaders {
    pub authorization: String,
    pub amz_date: String,
    pub payload_hash: String,
    pub security_token: Option<String>,
}

#[derive(Clone, Debug)]
struct AwsCredentials {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

#[derive(Clone, Debug)]
struct CachedCredentials {
    creds: AwsCredentials,
    /// None for static credentials (env/profile) that never expire.
    expires_at: Option<DateTime<chrono::Utc>>,
}

impl CachedCredentials {
    fn needs_refresh(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => {
                let refresh_at =
                    expires_at - chrono::Duration::seconds(CREDENTIAL_REFRESH_WINDOW_SECS);
                chrono::Utc::now() >= refresh_at
            }
            None => false,
        }
    }
}

impl AwsSigner {
    pub fn new(region: String, service: String) -> Self {
        Self {
            region,
            service,
            credentials: Arc::new(RwLock::new(None)),
            http_client: build_metadata_http_client(),
        }
    }

    pub async fn sign(
        &self,
        method: &str,
        url: &Url,
        body: &[u8],
    ) -> Result<SignedHeaders, String> {
        let creds = self.get_or_refresh_credentials().await?;

        let now = chrono::Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        let payload_hash = hex_sha256(body);
        let host = match (url.host_str(), url.port()) {
            (Some(h), Some(p)) => {
                let is_default_port = matches!((url.scheme(), p), ("http", 80) | ("https", 443));
                if is_default_port {
                    h.to_string()
                } else {
                    format!("{h}:{p}")
                }
            }
            (Some(h), None) => h.to_string(),
            _ => return Err("Bedrock URL missing host".to_string()),
        };

        let canonical_uri = canonical_uri(url.path());
        let canonical_query = url.query().unwrap_or("");

        let mut canonical_headers =
            format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
        let mut signed_headers = String::from("host;x-amz-content-sha256;x-amz-date");

        if let Some(token) = &creds.session_token {
            canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
            signed_headers.push_str(";x-amz-security-token");
        }

        let canonical_request = format!(
            "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let hashed_request = hex_sha256(canonical_request.as_bytes());
        let scope = format!("{date_stamp}/{}/{}/aws4_request", self.region, self.service);
        let string_to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashed_request}");

        let k_date = hmac_sha256(
            format!("AWS4{}", creds.secret_access_key).as_bytes(),
            date_stamp.as_bytes(),
        )?;
        let k_region = hmac_sha256(&k_date, self.region.as_bytes())?;
        let k_service = hmac_sha256(&k_region, self.service.as_bytes())?;
        let k_signing = hmac_sha256(&k_service, b"aws4_request")?;
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes())?);

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            creds.access_key_id, scope, signed_headers, signature
        );

        Ok(SignedHeaders {
            authorization,
            amz_date,
            payload_hash,
            security_token: creds.session_token.clone(),
        })
    }

    async fn get_or_refresh_credentials(&self) -> Result<AwsCredentials, String> {
        {
            let guard = self.credentials.read().await;
            if let Some(cached) = guard.as_ref() {
                if !cached.needs_refresh() {
                    return Ok(cached.creds.clone());
                }
            }
        }
        // Acquire write lock and double-check before refreshing.
        let mut guard = self.credentials.write().await;
        if let Some(cached) = guard.as_ref() {
            if !cached.needs_refresh() {
                return Ok(cached.creds.clone());
            }
        }
        let cached = self.load_credentials().await?;
        let creds = cached.creds.clone();
        *guard = Some(cached);
        Ok(creds)
    }

    async fn load_credentials(&self) -> Result<CachedCredentials, String> {
        if let Some(c) = load_env_credentials() {
            return Ok(CachedCredentials {
                creds: c,
                expires_at: None,
            });
        }
        if let Some(c) = load_profile_credentials() {
            return Ok(CachedCredentials {
                creds: c,
                expires_at: None,
            });
        }
        if let Some(c) = self.load_web_identity_credentials().await? {
            return Ok(c);
        }
        if let Some(c) = self.load_ecs_credentials().await? {
            return Ok(c);
        }
        if let Some(c) = self.load_imds_credentials().await? {
            return Ok(c);
        }
        Err(
            "Unable to resolve AWS credentials from env, profile, web identity, ECS, or IMDS"
                .to_string(),
        )
    }

    async fn load_web_identity_credentials(&self) -> Result<Option<CachedCredentials>, String> {
        let token_file = match std::env::var("AWS_WEB_IDENTITY_TOKEN_FILE").ok() {
            Some(f) if !f.is_empty() => f,
            _ => return Ok(None),
        };
        let role_arn = match std::env::var("AWS_ROLE_ARN").ok() {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };
        let session_name = std::env::var("AWS_ROLE_SESSION_NAME")
            .unwrap_or_else(|_| "smg-bedrock-session".to_string());

        let token = std::fs::read_to_string(&token_file)
            .map_err(|e| format!("Failed to read web identity token file {token_file}: {e}"))?;
        let token = token.trim().to_string();

        let region = self.region.clone();
        let sts_url = format!("https://sts.{region}.amazonaws.com/");
        let body = format!(
            "Action=AssumeRoleWithWebIdentity&Version=2011-06-15&RoleArn={}&RoleSessionName={}&WebIdentityToken={}",
            urlencoding_simple(&role_arn),
            urlencoding_simple(&session_name),
            urlencoding_simple(&token),
        );

        let resp = self
            .http_client
            .post(&sts_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| format!("STS AssumeRoleWithWebIdentity request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!(
                "STS AssumeRoleWithWebIdentity returned {status} \
                 (AWS_ROLE_ARN={role_arn}): {body}"
            ));
        }

        let xml = resp
            .text()
            .await
            .map_err(|e| format!("STS response read failed: {e}"))?;

        Ok(extract_sts_credentials(&xml))
    }

    async fn load_ecs_credentials(&self) -> Result<Option<CachedCredentials>, String> {
        let relative = std::env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").ok();
        let full = std::env::var("AWS_CONTAINER_CREDENTIALS_FULL_URI").ok();
        let (uri, auth_token) = match (full, relative) {
            (Some(full), _) => (full, read_container_authorization_token()?),
            (None, Some(rel)) => (format!("http://169.254.170.2{rel}"), None),
            (None, None) => return Ok(None),
        };

        let mut req = self.http_client.get(&uri);
        if let Some(token) = auth_token {
            req = req.header("Authorization", token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("ECS credential request failed: {e}"))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Invalid ECS credential response: {e}"))?;
        Ok(extract_credential_fields(&value))
    }

    async fn load_imds_credentials(&self) -> Result<Option<CachedCredentials>, String> {
        let disabled = std::env::var("AWS_EC2_METADATA_DISABLED")
            .ok()
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        if disabled {
            return Ok(None);
        }

        let token_resp = self
            .http_client
            .put("http://169.254.169.254/latest/api/token")
            .header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
            .send()
            .await
            .map_err(|e| format!("IMDS token request failed: {e}"))?;
        if !token_resp.status().is_success() {
            return Ok(None);
        }
        let token = token_resp
            .text()
            .await
            .map_err(|e| format!("IMDS token read failed: {e}"))?;

        let role_resp = self
            .http_client
            .get("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
            .header("X-aws-ec2-metadata-token", &token)
            .send()
            .await
            .map_err(|e| format!("IMDS role request failed: {e}"))?;
        if !role_resp.status().is_success() {
            return Ok(None);
        }
        let role_name = role_resp
            .text()
            .await
            .map_err(|e| format!("IMDS role read failed: {e}"))?
            .lines()
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
        let Some(role_name) = role_name else {
            return Ok(None);
        };

        let creds_resp = self
            .http_client
            .get(format!(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}"
            ))
            .header("X-aws-ec2-metadata-token", &token)
            .send()
            .await
            .map_err(|e| format!("IMDS credentials request failed: {e}"))?;
        if !creds_resp.status().is_success() {
            return Ok(None);
        }
        let value: serde_json::Value = creds_resp
            .json()
            .await
            .map_err(|e| format!("Invalid IMDS credential response: {e}"))?;
        Ok(extract_credential_fields(&value))
    }
}

fn build_metadata_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(AWS_METADATA_HTTP_TIMEOUT_SECS))
        .connect_timeout(Duration::from_secs(AWS_METADATA_HTTP_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    let mut mac =
        HmacSha256::new_from_slice(key).map_err(|e| format!("Invalid HMAC key length: {e}"))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn hex_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn canonical_uri(path: &str) -> String {
    let path = if path.is_empty() { "/" } else { path };
    let mut out = String::with_capacity(path.len());
    for &byte in path.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

/// Resolve the ECS container authorization token for use with
/// `AWS_CONTAINER_CREDENTIALS_FULL_URI`. Prefers the inline
/// `AWS_CONTAINER_AUTHORIZATION_TOKEN`, falling back to reading the file
/// referenced by `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE`. Returns `Ok(None)`
/// when neither is set.
fn read_container_authorization_token() -> Result<Option<String>, String> {
    if let Some(token) = std::env::var("AWS_CONTAINER_AUTHORIZATION_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
    {
        return Ok(Some(token));
    }
    if let Some(path) = std::env::var("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let token = std::fs::read_to_string(&path).map_err(|e| {
            format!("Failed to read AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE {path}: {e}")
        })?;
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Ok(Some(token));
        }
    }
    Ok(None)
}

fn load_env_credentials() -> Option<AwsCredentials> {
    // Treat empty env vars as absent so we fall back to web identity / ECS / IMDS
    // rather than signing with invalid blank credentials.
    let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
        .ok()
        .filter(|s| !s.is_empty())?;
    let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
        .ok()
        .filter(|s| !s.is_empty())?;
    let session_token = std::env::var("AWS_SESSION_TOKEN")
        .ok()
        .filter(|s| !s.is_empty());
    Some(AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token,
    })
}

fn load_profile_credentials() -> Option<AwsCredentials> {
    let profile = std::env::var("AWS_PROFILE").unwrap_or_else(|_| "default".to_string());
    let path = std::env::var("AWS_SHARED_CREDENTIALS_FILE")
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| format!("{home}/.aws/credentials"))
        })?;
    let content = std::fs::read_to_string(path).ok()?;

    let mut current_section = String::new();
    let mut access_key_id = None;
    let mut secret_access_key = None;
    let mut session_token = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].trim().to_string();
            continue;
        }
        if current_section != profile {
            continue;
        }
        let Some((k, v)) = trimmed.split_once('=') else {
            continue;
        };
        match k.trim() {
            "aws_access_key_id" => access_key_id = Some(v.trim().to_string()),
            "aws_secret_access_key" => secret_access_key = Some(v.trim().to_string()),
            "aws_session_token" => session_token = Some(v.trim().to_string()),
            _ => {}
        }
    }

    Some(AwsCredentials {
        access_key_id: access_key_id?,
        secret_access_key: secret_access_key?,
        session_token,
    })
}

/// Minimal percent-encoding for form values (encodes everything except unreserved chars).
pub(super) fn urlencoding_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

/// Extract credentials from STS AssumeRoleWithWebIdentity XML response.
fn extract_sts_credentials(xml: &str) -> Option<CachedCredentials> {
    let access_key_id = xml_text_between(xml, "<AccessKeyId>", "</AccessKeyId>")?;
    let secret_access_key = xml_text_between(xml, "<SecretAccessKey>", "</SecretAccessKey>")?;
    let session_token = xml_text_between(xml, "<SessionToken>", "</SessionToken>");
    let expires_at = xml_text_between(xml, "<Expiration>", "</Expiration>")
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    Some(CachedCredentials {
        creds: AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        },
        expires_at,
    })
}

fn xml_text_between(xml: &str, open: &str, close: &str) -> Option<String> {
    let start = xml.find(open)? + open.len();
    let end = xml[start..].find(close)? + start;
    Some(xml[start..end].trim().to_string())
}

fn extract_credential_fields(value: &serde_json::Value) -> Option<CachedCredentials> {
    let access_key_id = value
        .get("AccessKeyId")
        .or_else(|| value.get("accessKeyId"))
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let secret_access_key = value
        .get("SecretAccessKey")
        .or_else(|| value.get("secretAccessKey"))
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let session_token = value
        .get("Token")
        .or_else(|| value.get("sessionToken"))
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    let expires_at = value
        .get("Expiration")
        .or_else(|| value.get("expiration"))
        .and_then(serde_json::Value::as_str)
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    Some(CachedCredentials {
        creds: AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        },
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::canonical_uri;

    #[test]
    fn canonical_uri_encodes_model_id_colon_once() {
        assert_eq!(
            canonical_uri("/model/us.anthropic.claude-opus-4-5-20251101-v1:0/converse"),
            "/model/us.anthropic.claude-opus-4-5-20251101-v1%3A0/converse"
        );
    }

    #[test]
    fn canonical_uri_encodes_converse_stream_path() {
        assert_eq!(
            canonical_uri("/model/us.anthropic.claude-opus-4-5-20251101-v1:0/converse-stream"),
            "/model/us.anthropic.claude-opus-4-5-20251101-v1%3A0/converse-stream"
        );
    }

    #[test]
    fn canonical_uri_double_encodes_pre_encoded_path() {
        // SigV4 for non-S3 services requires double-encoding: the wire path
        // has %3A, and the canonical request encodes % again → %253A.
        assert_eq!(
            canonical_uri("/model/us.anthropic.claude-opus-4-5-20251101-v1%3A0/converse"),
            "/model/us.anthropic.claude-opus-4-5-20251101-v1%253A0/converse"
        );
    }
}
