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
use smg_skills::SkillsConfig;
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
