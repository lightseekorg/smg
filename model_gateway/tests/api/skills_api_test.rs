#![expect(
    clippy::disallowed_methods,
    clippy::expect_used,
    reason = "integration test intentionally uses panic-on-failure helpers and a background test server task"
)]

use std::{
    any::Any,
    io::{Cursor, Write},
    sync::Arc,
};

use async_trait::async_trait;
use axum::{body::Body, extract::Request, response::Response, Router};
use reqwest::multipart::{Form, Part};
use serde_json::Value;
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::RouterTrait,
};
use smg_skills::{SkillsAdminOperation, SkillsConfig};
use tempfile::TempDir;
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::common::test_app::create_test_app_with_context;

#[derive(Debug)]
struct NoopRouter;

#[async_trait]
impl RouterTrait for NoopRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "noop"
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        Response::new(Body::empty())
    }
}

fn skills_test_config(
    blob_dir: &TempDir,
    cache_dir: &TempDir,
    admin_enabled: bool,
) -> RouterConfig {
    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://worker1:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    config.api_key = Some("test-admin-key".to_string());
    config.skills_enabled = true;

    let mut skills = SkillsConfig::default();
    skills.admin.enabled = admin_enabled;
    skills.blob_store.path = blob_dir.path().display().to_string();
    skills.cache.path = cache_dir.path().display().to_string();
    config.skills = Some(skills);

    config
}

async fn create_skills_test_app(config: RouterConfig) -> Router {
    let context = Arc::new(
        AppContext::from_config(config, 60, None, None)
            .await
            .expect("build test app context"),
    );
    create_test_app_with_context(Arc::new(NoopRouter), context)
}

async fn spawn_app(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("listener addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve test app");
    });
    (format!("http://{addr}"), handle)
}

fn build_skill_zip(skill_md: &[u8], extra_files: &[(&str, &[u8])]) -> Vec<u8> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    writer
        .start_file("bundle/SKILL.md", options)
        .expect("start SKILL.md entry");
    writer.write_all(skill_md).expect("write SKILL.md");

    for (path, contents) in extra_files {
        writer
            .start_file(format!("bundle/{path}"), options)
            .expect("start extra zip entry");
        writer.write_all(contents).expect("write extra zip entry");
    }

    writer.finish().expect("finish zip writer").into_inner()
}

async fn create_skill_via_api(
    client: &reqwest::Client,
    base_url: &str,
    tenant_id: &str,
    skill_md: &[u8],
) -> Value {
    let response = client
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", tenant_id.to_string()).part(
                "files[]",
                Part::bytes(skill_md.to_vec())
                    .file_name("SKILL.md")
                    .mime_str("text/markdown")
                    .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill request");
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    response.json().await.expect("json response")
}

async fn create_skill_version_via_api(
    client: &reqwest::Client,
    base_url: &str,
    tenant_id: &str,
    skill_id: &str,
    skill_md: &[u8],
) -> Value {
    let response = client
        .post(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", tenant_id.to_string()).part(
                "files[]",
                Part::bytes(skill_md.to_vec())
                    .file_name("SKILL.md")
                    .mime_str("text/markdown")
                    .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill version request");
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    response.json().await.expect("json response")
}

fn extract_etag(response: &reqwest::Response) -> String {
    response
        .headers()
        .get(reqwest::header::ETAG)
        .expect("etag header")
        .to_str()
        .expect("etag text")
        .to_string()
}

async fn assert_if_none_match_status(
    request: reqwest::RequestBuilder,
    etag: &str,
    expected_status: reqwest::StatusCode,
) {
    let if_none_match = request
        .try_clone()
        .expect("clone request builder for etag")
        .header(reqwest::header::IF_NONE_MATCH, etag)
        .send()
        .await
        .expect("send If-None-Match request");
    assert_eq!(if_none_match.status(), expected_status);
}

#[tokio::test]
async fn create_skill_returns_created_skill_and_version() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new()
                .text("tenant_id", "tenant-a")
                .part(
                    "files[]",
                    Part::bytes(
                        b"---\nname: acme:map\ndescription: Map the repo\nmetadata:\n  short-description: Map it\n---\nUse rg."
                            .to_vec(),
                    )
                    .file_name("SKILL.md")
                    .mime_str("text/markdown")
                    .expect("valid markdown mime"),
                )
                .part(
                    "files[]",
                    Part::bytes(b"print('hi')".to_vec())
                        .file_name("scripts/run.py")
                        .mime_str("text/x-python")
                        .expect("valid python mime"),
                ),
        )
        .send()
        .await
        .expect("send create skill request");

    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    let body: Value = response.json().await.expect("json response");
    assert_eq!(body["skill"]["name"], "acme:map");
    assert_eq!(body["version"]["version_number"], 1);
    assert_eq!(body["version"]["files"].as_array().map(Vec::len), Some(2));
    assert!(body["skill"]["id"]
        .as_str()
        .is_some_and(|id| id.starts_with("skill_") && id.len() == 32));

    server.abort();
}

#[tokio::test]
async fn create_skill_accepts_zip_bundle_upload() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;

    let bundle = build_skill_zip(
        b"---\nname: acme:zip\ndescription: Zip based skill\nmetadata:\n  short-description: Zip it\n---\nUse zip uploads.",
        &[("scripts/run.py", b"print('zip')")],
    );

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", "tenant-a").part(
                "bundle",
                Part::bytes(bundle)
                    .file_name("acme-skill.zip")
                    .mime_str("application/zip")
                    .expect("valid zip mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill zip request");

    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    let body: Value = response.json().await.expect("json response");
    assert_eq!(body["skill"]["name"], "acme:zip");
    assert_eq!(body["version"]["version_number"], 1);
    assert_eq!(body["version"]["files"].as_array().map(Vec::len), Some(2));

    server.abort();
}

#[tokio::test]
async fn create_skill_rejects_missing_target_tenant() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().part(
                "files[]",
                Part::bytes(
                    b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.".to_vec(),
                )
                .file_name("SKILL.md")
                .mime_str("text/markdown")
                .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill request");

    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json response");
    assert_eq!(body["error"]["code"], "missing_target_tenant");

    server.abort();
}

#[tokio::test]
async fn create_skill_version_returns_incremented_version() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let create_response = client
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", "tenant-a").part(
                "files[]",
                Part::bytes(
                    b"---\nname: acme:map\ndescription: Map the repo\nmetadata:\n  short-description: Map it\n---\nUse rg."
                        .to_vec(),
                )
                .file_name("SKILL.md")
                .mime_str("text/markdown")
                .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill request");
    assert_eq!(create_response.status(), reqwest::StatusCode::CREATED);
    let create_body: Value = create_response.json().await.expect("json response");
    let skill_id = create_body["skill"]["id"]
        .as_str()
        .expect("skill id in create response")
        .to_string();

    let version_response = client
        .post(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new()
                .text("tenant_id", "tenant-a")
                .part(
                    "files[]",
                    Part::bytes(
                        b"---\nname: acme:map\ndescription: Updated mapping skill\nmetadata:\n  short-description: Map it better\n---\nUse rg --files."
                            .to_vec(),
                    )
                    .file_name("SKILL.md")
                    .mime_str("text/markdown")
                    .expect("valid markdown mime"),
                )
                .part(
                    "files[]",
                    Part::bytes(b"print('updated')".to_vec())
                        .file_name("scripts/run.py")
                        .mime_str("text/x-python")
                        .expect("valid python mime"),
                ),
        )
        .send()
        .await
        .expect("send create skill version request");

    assert_eq!(version_response.status(), reqwest::StatusCode::CREATED);
    let version_body: Value = version_response.json().await.expect("json response");
    assert_eq!(version_body["skill"]["id"], skill_id);
    assert_eq!(
        version_body["skill"]["latest_version"],
        version_body["version"]["version"]
    );
    assert_eq!(version_body["version"]["version_number"], 2);
    assert_eq!(
        version_body["version"]["files"].as_array().map(Vec::len),
        Some(2)
    );

    server.abort();
}

#[tokio::test]
async fn list_skills_returns_paginated_results_and_supports_if_none_match() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let first_skill = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.",
    )
    .await;
    let second_skill = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd.",
    )
    .await;

    let list_response = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a"), ("limit", "1")])
        .send()
        .await
        .expect("send list skills request");
    assert_eq!(list_response.status(), reqwest::StatusCode::OK);
    let etag = list_response
        .headers()
        .get(reqwest::header::ETAG)
        .expect("etag header")
        .to_str()
        .expect("etag text")
        .to_string();
    assert!(list_response
        .headers()
        .contains_key(reqwest::header::LAST_MODIFIED));
    let list_body: Value = list_response.json().await.expect("json response");
    assert_eq!(list_body["object"], "list");
    assert_eq!(list_body["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(list_body["has_more"], true);

    let listed_skill_id = list_body["data"][0]["id"]
        .as_str()
        .expect("listed skill id")
        .to_string();
    let after = list_body["last_id"]
        .as_str()
        .expect("last id on first page")
        .to_string();

    let not_modified = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a"), ("limit", "1")])
        .header(reqwest::header::IF_NONE_MATCH, etag)
        .send()
        .await
        .expect("send conditional list skills request");
    assert_eq!(not_modified.status(), reqwest::StatusCode::NOT_MODIFIED);

    create_skill_version_via_api(
        &client,
        &base_url,
        "tenant-a",
        &listed_skill_id,
        b"---\nname: acme:map\ndescription: Updated list ordering skill\n---\nUse rg --files.",
    )
    .await;

    let next_page = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[
            ("tenant_id", "tenant-a"),
            ("limit", "1"),
            ("after", after.as_str()),
        ])
        .send()
        .await
        .expect("send second list skills request");
    assert_eq!(next_page.status(), reqwest::StatusCode::OK);
    let next_body: Value = next_page.json().await.expect("json response");
    assert_eq!(next_body["data"].as_array().map(Vec::len), Some(1));
    let returned_id = next_body["data"][0]["id"].as_str().expect("skill id");
    assert_ne!(returned_id, listed_skill_id);
    assert!(
        returned_id == first_skill["skill"]["id"].as_str().expect("first skill id")
            || returned_id
                == second_skill["skill"]["id"]
                    .as_str()
                    .expect("second skill id")
    );

    server.abort();
}

#[tokio::test]
async fn get_skill_and_version_read_endpoints_require_target_tenant_and_return_cache_headers() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let created = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:map\ndescription: Map the repo\nmetadata:\n  short-description: Map it\n---\nUse rg.",
    )
    .await;
    let skill_id = created["skill"]["id"]
        .as_str()
        .expect("skill id")
        .to_string();

    create_skill_version_via_api(
        &client,
        &base_url,
        "tenant-a",
        &skill_id,
        b"---\nname: acme:map\ndescription: Updated mapping skill\nmetadata:\n  short-description: Map it better\n---\nUse rg --files.",
    )
    .await;

    let missing_tenant = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .send()
        .await
        .expect("send missing tenant get request");
    assert_eq!(missing_tenant.status(), reqwest::StatusCode::BAD_REQUEST);
    let missing_body: Value = missing_tenant.json().await.expect("json response");
    assert_eq!(missing_body["error"]["code"], "missing_target_tenant");

    let get_skill = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send get skill request");
    assert_eq!(get_skill.status(), reqwest::StatusCode::OK);
    let skill_etag = get_skill
        .headers()
        .get(reqwest::header::ETAG)
        .expect("etag header")
        .to_str()
        .expect("etag text")
        .to_string();
    let skill_last_modified = get_skill
        .headers()
        .get(reqwest::header::LAST_MODIFIED)
        .expect("last-modified header")
        .to_str()
        .expect("last-modified text")
        .to_string();
    assert!(get_skill
        .headers()
        .contains_key(reqwest::header::LAST_MODIFIED));
    let skill_body: Value = get_skill.json().await.expect("json response");
    assert_eq!(skill_body["id"], skill_id);

    let list_versions = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send list versions request");
    assert_eq!(list_versions.status(), reqwest::StatusCode::OK);
    let versions_body: Value = list_versions.json().await.expect("json response");
    assert_eq!(versions_body["object"], "list");
    assert_eq!(versions_body["data"].as_array().map(Vec::len), Some(2));

    let get_version = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions/2"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send get version request");
    assert_eq!(get_version.status(), reqwest::StatusCode::OK);
    assert!(get_version.headers().contains_key(reqwest::header::ETAG));
    let version_body: Value = get_version.json().await.expect("json response");
    assert_eq!(version_body["version_number"], 2);

    let not_modified = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .header(reqwest::header::IF_NONE_MATCH, skill_etag)
        .send()
        .await
        .expect("send conditional get skill request");
    assert_eq!(not_modified.status(), reqwest::StatusCode::NOT_MODIFIED);

    let not_modified_since = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .header(
            reqwest::header::IF_MODIFIED_SINCE,
            (chrono::DateTime::parse_from_rfc2822(&skill_last_modified)
                .expect("parse last-modified header")
                + chrono::Duration::seconds(1))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string(),
        )
        .send()
        .await
        .expect("send if-modified-since get skill request");
    assert_eq!(
        not_modified_since.status(),
        reqwest::StatusCode::NOT_MODIFIED
    );

    server.abort();
}

#[tokio::test]
async fn invalid_after_cursor_returns_bad_request_before_conditional_cache_checks() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let created = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.",
    )
    .await;
    let skill_id = created["skill"]["id"]
        .as_str()
        .expect("skill id")
        .to_string();
    create_skill_version_via_api(
        &client,
        &base_url,
        "tenant-a",
        &skill_id,
        b"---\nname: acme:map\ndescription: Updated mapping skill\n---\nUse rg --files.",
    )
    .await;

    let list_response = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send list skills request");
    let list_etag = list_response
        .headers()
        .get(reqwest::header::ETAG)
        .expect("skills list etag")
        .to_str()
        .expect("etag text")
        .to_string();

    let invalid_skills_after = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a"), ("after", "not-a-valid-cursor")])
        .header(reqwest::header::IF_NONE_MATCH, list_etag)
        .send()
        .await
        .expect("send invalid after list request");
    assert_eq!(
        invalid_skills_after.status(),
        reqwest::StatusCode::BAD_REQUEST
    );
    let invalid_skills_body: Value = invalid_skills_after.json().await.expect("json response");
    assert_eq!(invalid_skills_body["error"]["code"], "invalid_after_cursor");

    let versions_response = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send list skill versions request");
    let versions_etag = versions_response
        .headers()
        .get(reqwest::header::ETAG)
        .expect("skill versions etag")
        .to_str()
        .expect("etag text")
        .to_string();

    let invalid_versions_after = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a"), ("after", "bogus-version")])
        .header(reqwest::header::IF_NONE_MATCH, versions_etag)
        .send()
        .await
        .expect("send invalid versions after request");
    assert_eq!(
        invalid_versions_after.status(),
        reqwest::StatusCode::BAD_REQUEST
    );
    let invalid_versions_body: Value = invalid_versions_after.json().await.expect("json response");
    assert_eq!(
        invalid_versions_body["error"]["code"],
        "invalid_after_cursor"
    );

    server.abort();
}

#[tokio::test]
async fn patch_skill_and_version_endpoints_update_default_and_deprecated_state() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let created = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.",
    )
    .await;
    let skill_id = created["skill"]["id"]
        .as_str()
        .expect("skill id")
        .to_string();
    let second = create_skill_version_via_api(
        &client,
        &base_url,
        "tenant-a",
        &skill_id,
        b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd.",
    )
    .await;
    let skill_get = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send initial get skill request");
    assert_eq!(skill_get.status(), reqwest::StatusCode::OK);
    let skill_etag = extract_etag(&skill_get);
    let _: Value = skill_get.json().await.expect("json response");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &skill_etag,
        reqwest::StatusCode::NOT_MODIFIED,
    )
    .await;

    let patch_skill = client
        .patch(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .json(&serde_json::json!({
            "default_version": 2
        }))
        .send()
        .await
        .expect("send patch skill request");
    assert_eq!(patch_skill.status(), reqwest::StatusCode::OK);
    let patched_skill: Value = patch_skill.json().await.expect("json response");
    assert_eq!(
        patched_skill["default_version"],
        second["version"]["version"]
    );
    assert_eq!(patched_skill["name"], "acme:search");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &skill_etag,
        reqwest::StatusCode::OK,
    )
    .await;

    let versions_before_patch = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send pre-patch list versions request");
    assert_eq!(versions_before_patch.status(), reqwest::StatusCode::OK);
    let versions_etag = extract_etag(&versions_before_patch);
    let _: Value = versions_before_patch.json().await.expect("json response");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &versions_etag,
        reqwest::StatusCode::NOT_MODIFIED,
    )
    .await;

    let patch_version = client
        .patch(format!(
            "{base_url}/v1/skills/{skill_id}/versions/{}",
            second["version"]["version"].as_str().expect("version id")
        ))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .json(&serde_json::json!({
            "deprecated": true
        }))
        .send()
        .await
        .expect("send patch skill version request");
    assert_eq!(patch_version.status(), reqwest::StatusCode::OK);
    let patched_version: Value = patch_version.json().await.expect("json response");
    assert_eq!(patched_version["deprecated"], true);
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &versions_etag,
        reqwest::StatusCode::OK,
    )
    .await;

    let list_versions = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send list versions request");
    assert_eq!(list_versions.status(), reqwest::StatusCode::OK);
    let versions_body: Value = list_versions.json().await.expect("json response");
    assert_eq!(versions_body["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(versions_body["data"][0]["version_number"], 1);

    let include_deprecated = client
        .get(format!("{base_url}/v1/skills/{skill_id}/versions"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a"), ("include_deprecated", "true")])
        .send()
        .await
        .expect("send list versions include deprecated request");
    assert_eq!(include_deprecated.status(), reqwest::StatusCode::OK);
    let include_body: Value = include_deprecated.json().await.expect("json response");
    assert_eq!(include_body["data"].as_array().map(Vec::len), Some(2));

    server.abort();
}

#[tokio::test]
async fn delete_skill_version_and_skill_endpoints_enforce_default_and_cascade_rules() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, true)).await;
    let (base_url, server) = spawn_app(app).await;
    let client = reqwest::Client::new();

    let created = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.",
    )
    .await;
    let skill_id = created["skill"]["id"]
        .as_str()
        .expect("skill id")
        .to_string();
    let second = create_skill_version_via_api(
        &client,
        &base_url,
        "tenant-a",
        &skill_id,
        b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd.",
    )
    .await;

    let delete_default = client
        .delete(format!(
            "{base_url}/v1/skills/{skill_id}/versions/{}",
            created["version"]["version"]
                .as_str()
                .expect("default version id")
        ))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send delete default version request");
    assert_eq!(delete_default.status(), reqwest::StatusCode::CONFLICT);
    let delete_default_body: Value = delete_default.json().await.expect("json response");
    assert_eq!(
        delete_default_body["error"]["code"],
        "default_version_conflict"
    );

    let patch_skill = client
        .patch(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .json(&serde_json::json!({
            "default_version": second["version"]["version"]
        }))
        .send()
        .await
        .expect("send patch skill request");
    assert_eq!(patch_skill.status(), reqwest::StatusCode::OK);

    let skill_before_delete = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send get skill before delete request");
    assert_eq!(skill_before_delete.status(), reqwest::StatusCode::OK);
    let skill_etag = extract_etag(&skill_before_delete);
    let _: Value = skill_before_delete.json().await.expect("json response");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &skill_etag,
        reqwest::StatusCode::NOT_MODIFIED,
    )
    .await;

    let delete_first = client
        .delete(format!(
            "{base_url}/v1/skills/{skill_id}/versions/{}",
            created["version"]["version"]
                .as_str()
                .expect("first version id")
        ))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send delete first version request");
    assert_eq!(delete_first.status(), reqwest::StatusCode::NO_CONTENT);
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &skill_etag,
        reqwest::StatusCode::OK,
    )
    .await;

    let get_skill = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send get skill request");
    assert_eq!(get_skill.status(), reqwest::StatusCode::OK);
    let post_delete_skill_etag = extract_etag(&get_skill);
    let _: Value = get_skill.json().await.expect("json response");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &post_delete_skill_etag,
        reqwest::StatusCode::NOT_MODIFIED,
    )
    .await;

    let delete_last = client
        .delete(format!(
            "{base_url}/v1/skills/{skill_id}/versions/{}",
            second["version"]["version"]
                .as_str()
                .expect("second version id")
        ))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send delete last version request");
    assert_eq!(delete_last.status(), reqwest::StatusCode::NO_CONTENT);
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills/{skill_id}"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &post_delete_skill_etag,
        reqwest::StatusCode::NOT_FOUND,
    )
    .await;

    let missing_skill = client
        .get(format!("{base_url}/v1/skills/{skill_id}"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send get missing skill request");
    assert_eq!(missing_skill.status(), reqwest::StatusCode::NOT_FOUND);

    let created_again = create_skill_via_api(
        &client,
        &base_url,
        "tenant-a",
        b"---\nname: acme:delete\ndescription: Delete the repo map\n---\nUse rm carefully.",
    )
    .await;
    let list_before_delete = client
        .get(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send list skills before delete request");
    assert_eq!(list_before_delete.status(), reqwest::StatusCode::OK);
    let list_etag = extract_etag(&list_before_delete);
    let _: Value = list_before_delete.json().await.expect("json response");
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &list_etag,
        reqwest::StatusCode::NOT_MODIFIED,
    )
    .await;

    let delete_skill = client
        .delete(format!(
            "{base_url}/v1/skills/{}",
            created_again["skill"]["id"].as_str().expect("skill id")
        ))
        .bearer_auth("test-admin-key")
        .query(&[("tenant_id", "tenant-a")])
        .send()
        .await
        .expect("send delete skill request");
    assert_eq!(delete_skill.status(), reqwest::StatusCode::NO_CONTENT);
    assert_if_none_match_status(
        client
            .get(format!("{base_url}/v1/skills"))
            .bearer_auth("test-admin-key")
            .query(&[("tenant_id", "tenant-a")]),
        &list_etag,
        reqwest::StatusCode::OK,
    )
    .await;

    server.abort();
}

#[tokio::test]
async fn create_skill_route_is_not_mounted_when_skills_admin_is_disabled() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let app = create_skills_test_app(skills_test_config(&blob_dir, &cache_dir, false)).await;
    let (base_url, server) = spawn_app(app).await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", "tenant-a").part(
                "files[]",
                Part::bytes(
                    b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.".to_vec(),
                )
                .file_name("SKILL.md")
                .mime_str("text/markdown")
                .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill request");

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

    server.abort();
}

#[tokio::test]
async fn create_skill_enforces_configured_upload_limits_before_service() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let mut config = skills_test_config(&blob_dir, &cache_dir, true);
    let skills = config.skills.as_mut().expect("skills config");
    skills.max_upload_size_mb = 2;
    skills.max_file_size_mb = 1;
    let app = create_skills_test_app(config).await;
    let (base_url, server) = spawn_app(app).await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new()
                .text("tenant_id", "tenant-a")
                .part(
                    "files[]",
                    Part::bytes(
                        b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.".to_vec(),
                    )
                    .file_name("SKILL.md")
                    .mime_str("text/markdown")
                    .expect("valid markdown mime"),
                )
                .part(
                    "files[]",
                    Part::bytes(vec![b'x'; 1024 * 1024 + 1])
                        .file_name("too-large.txt")
                        .mime_str("text/plain")
                        .expect("valid text mime"),
                ),
        )
        .send()
        .await
        .expect("send create skill request");

    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json response");
    assert_eq!(body["error"]["code"], "skill_upload_too_large");

    server.abort();
}

#[tokio::test]
async fn skills_admin_allowed_operations_gate_create_requests() {
    let blob_dir = tempfile::tempdir().expect("blob tempdir");
    let cache_dir = tempfile::tempdir().expect("cache tempdir");
    let mut config = skills_test_config(&blob_dir, &cache_dir, true);
    config
        .skills
        .as_mut()
        .expect("skills config")
        .admin
        .allowed_operations = vec![SkillsAdminOperation::ReadAnyTenant];
    let app = create_skills_test_app(config).await;
    let (base_url, server) = spawn_app(app).await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/skills"))
        .bearer_auth("test-admin-key")
        .multipart(
            Form::new().text("tenant_id", "tenant-a").part(
                "files[]",
                Part::bytes(
                    b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg.".to_vec(),
                )
                .file_name("SKILL.md")
                .mime_str("text/markdown")
                .expect("valid markdown mime"),
            ),
        )
        .send()
        .await
        .expect("send create skill request");

    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);
    let body: Value = response.json().await.expect("json response");
    assert_eq!(body["error"]["code"], "skills_operation_not_allowed");

    server.abort();
}
