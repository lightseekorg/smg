use std::io::Cursor;

use anyhow::{anyhow, Result};
use chrono::Utc;
use smg_blob_storage::{
    create_blob_store, BlobCacheConfig, BlobKey, BlobStoreBackend, BlobStoreConfig,
    GetBlobResponse, PutBlobRequest,
};
use smg_skills::{
    BundleTokenClaim, ContinuationCookieClaim, NormalizedSkillBundle, NormalizedSkillFile,
    SkillRecord, SkillService, SkillVersionRecord, TenantAliasRecord,
};
use tempfile::TempDir;
use tokio::io::AsyncReadExt;

fn put_request(bytes: &[u8]) -> PutBlobRequest {
    PutBlobRequest {
        reader: Box::pin(Cursor::new(bytes.to_vec())),
        content_length: bytes.len() as u64,
        content_type: None,
    }
}

async fn read_all(response: GetBlobResponse) -> Result<Vec<u8>> {
    let mut reader = response.reader;
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).await?;
    Ok(buffer)
}

#[tokio::test]
async fn in_memory_service_supports_metadata_tokens_and_filesystem_blob_reads() -> Result<()> {
    let blob_root = TempDir::new()?;
    let cache_root = TempDir::new()?;
    let blob_store = create_blob_store(
        &BlobStoreConfig {
            backend: BlobStoreBackend::Filesystem,
            path: blob_root.path().display().to_string(),
            ..BlobStoreConfig::default()
        },
        Some(&BlobCacheConfig {
            path: cache_root.path().display().to_string(),
            max_size_mb: 16,
        }),
    )?;
    let service = SkillService::in_memory(blob_store);
    let now = Utc::now();

    let metadata_store = service
        .metadata_store()
        .ok_or_else(|| anyhow!("metadata store missing"))?;
    metadata_store
        .put_skill(SkillRecord {
            tenant_id: "tenant-a".to_string(),
            skill_id: "skill-1".to_string(),
            name: "map".to_string(),
            short_description: Some("Map the codebase".to_string()),
            description: Some("Reads and maps the codebase".to_string()),
            source: "custom".to_string(),
            has_code_files: true,
            latest_version: Some("20260420".to_string()),
            default_version: Some("20260420".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;
    metadata_store
        .put_skill_version(SkillVersionRecord {
            skill_id: "skill-1".to_string(),
            version: "20260420".to_string(),
            version_number: 20260420,
            name: "map".to_string(),
            short_description: Some("Map the codebase".to_string()),
            description: "Reads and maps the codebase".to_string(),
            interface: None,
            dependencies: None,
            policy: None,
            deprecated: false,
            file_manifest: Vec::new(),
            instruction_token_counts: Default::default(),
            created_at: now,
        })
        .await?;

    let listed = metadata_store.list_skills("tenant-a").await?;
    assert_eq!(listed.len(), 1);
    let version = metadata_store
        .get_skill_version("skill-1", "20260420")
        .await?
        .ok_or_else(|| anyhow!("skill version missing"))?;
    assert_eq!(version.version_number, 20260420);

    let alias_store = service
        .tenant_alias_store()
        .ok_or_else(|| anyhow!("alias store missing"))?;
    alias_store
        .put_tenant_alias(TenantAliasRecord {
            alias_tenant_id: "tenant-alias".to_string(),
            canonical_tenant_id: "tenant-a".to_string(),
            created_at: now,
            expires_at: None,
        })
        .await?;
    assert_eq!(
        alias_store
            .get_tenant_alias("tenant-alias")
            .await?
            .ok_or_else(|| anyhow!("tenant alias missing"))?
            .canonical_tenant_id,
        "tenant-a"
    );

    let bundle_token_store = service
        .bundle_token_store()
        .ok_or_else(|| anyhow!("bundle token store missing"))?;
    bundle_token_store
        .put_bundle_token(BundleTokenClaim {
            token_hash: "tokhash".to_string(),
            tenant_id: "tenant-a".to_string(),
            exec_id: "exec-1".to_string(),
            skill_id: "skill-1".to_string(),
            skill_version: "20260420".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;
    assert!(bundle_token_store
        .get_bundle_token("tokhash")
        .await?
        .is_some());
    assert_eq!(
        bundle_token_store
            .revoke_bundle_tokens_for_exec("exec-1")
            .await?,
        1
    );

    let continuation_cookie_store = service
        .continuation_cookie_store()
        .ok_or_else(|| anyhow!("continuation cookie store missing"))?;
    continuation_cookie_store
        .put_continuation_cookie(ContinuationCookieClaim {
            cookie_hash: "cookiehash".to_string(),
            tenant_id: "tenant-a".to_string(),
            exec_id: "exec-2".to_string(),
            request_id: "req-1".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;
    assert_eq!(
        continuation_cookie_store
            .revoke_continuation_cookies_for_exec("exec-2")
            .await?,
        1
    );

    let bundle = NormalizedSkillBundle {
        files: vec![
            NormalizedSkillFile {
                relative_path: "SKILL.md".to_string(),
                contents: b"---\nname: map\ndescription: test\n---\nbody".to_vec(),
            },
            NormalizedSkillFile {
                relative_path: "scripts/run.py".to_string(),
                contents: b"print('mapped')".to_vec(),
            },
        ],
        skill_md_path: "SKILL.md".to_string(),
        openai_sidecar_path: None,
        has_code_files: true,
    };
    let mut manifest = bundle.file_manifest();
    for entry in &mut manifest {
        entry.blob_key = Some(BlobKey(format!(
            "skills/tenant-a/skill-1/20260420/{}",
            entry.relative_path
        )));
    }

    let blob_store = service
        .blob_store()
        .ok_or_else(|| anyhow!("blob store missing"))?;
    for (file, entry) in bundle.files.iter().zip(&manifest) {
        let blob_key = entry
            .blob_key
            .as_ref()
            .ok_or_else(|| anyhow!("blob key missing from manifest"))?;
        blob_store
            .put_stream(blob_key, put_request(&file.contents))
            .await?;
    }

    let script_key = manifest[1]
        .blob_key
        .as_ref()
        .ok_or_else(|| anyhow!("script blob key missing"))?;
    let script_bytes = read_all(blob_store.get(script_key).await?).await?;
    assert_eq!(script_bytes, b"print('mapped')");

    Ok(())
}
