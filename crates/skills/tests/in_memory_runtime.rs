use anyhow::{anyhow, Result};
use chrono::Utc;
use smg_blob_storage::{
    create_blob_store, BlobCacheConfig, BlobKey, BlobStoreBackend, BlobStoreConfig,
};
use smg_skills::{
    BundleTokenClaim, ContinuationCookieClaim, NormalizedSkillBundle, NormalizedSkillFile,
    SkillRecord, SkillService, SkillVersionRecord, TenantAliasRecord,
};
use tempfile::TempDir;

#[path = "../../../test_support/blob_test_utils.rs"]
mod blob_test_utils;

use blob_test_utils::{put_request, read_all};

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

#[tokio::test]
async fn in_memory_service_scopes_listings_to_exact_tenant_and_skill() -> Result<()> {
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
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;
    metadata_store
        .put_skill(SkillRecord {
            tenant_id: "tenant-b".to_string(),
            skill_id: "skill-2".to_string(),
            name: "review".to_string(),
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;
    metadata_store
        .put_skill(SkillRecord {
            tenant_id: "tenant-b".to_string(),
            skill_id: "skill-10".to_string(),
            name: "map-10".to_string(),
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;
    metadata_store
        .put_skill_version(SkillVersionRecord {
            skill_id: "skill-1".to_string(),
            version: "1".to_string(),
            version_number: 1,
            name: "map".to_string(),
            short_description: None,
            description: "map".to_string(),
            interface: None,
            dependencies: None,
            policy: None,
            deprecated: false,
            file_manifest: Vec::new(),
            instruction_token_counts: Default::default(),
            created_at: now,
        })
        .await?;
    metadata_store
        .put_skill_version(SkillVersionRecord {
            skill_id: "skill-10".to_string(),
            version: "1".to_string(),
            version_number: 1,
            name: "map-10".to_string(),
            short_description: None,
            description: "map-10".to_string(),
            interface: None,
            dependencies: None,
            policy: None,
            deprecated: false,
            file_manifest: Vec::new(),
            instruction_token_counts: Default::default(),
            created_at: now,
        })
        .await?;

    let tenant_a_skills = metadata_store.list_skills("tenant-a").await?;
    assert_eq!(tenant_a_skills.len(), 1);
    assert_eq!(tenant_a_skills[0].skill_id, "skill-1");

    let skill_1_versions = metadata_store.list_skill_versions("skill-1").await?;
    assert_eq!(skill_1_versions.len(), 1);
    assert_eq!(skill_1_versions[0].skill_id, "skill-1");

    Ok(())
}

#[tokio::test]
async fn in_memory_service_rejects_cross_tenant_skill_id_reuse() -> Result<()> {
    let blob_root = TempDir::new()?;
    let blob_store = create_blob_store(
        &BlobStoreConfig {
            backend: BlobStoreBackend::Filesystem,
            path: blob_root.path().display().to_string(),
            ..BlobStoreConfig::default()
        },
        None,
    )?;
    let service = SkillService::in_memory(blob_store);
    let now = Utc::now();
    let metadata_store = service
        .metadata_store()
        .ok_or_else(|| anyhow!("metadata store missing"))?;

    metadata_store
        .put_skill(SkillRecord {
            tenant_id: "tenant-a".to_string(),
            skill_id: "shared-skill".to_string(),
            name: "map".to_string(),
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;

    let error = metadata_store
        .put_skill(SkillRecord {
            tenant_id: "tenant-b".to_string(),
            skill_id: "shared-skill".to_string(),
            name: "map".to_string(),
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await
        .expect_err("cross-tenant skill_id reuse should fail");
    assert!(matches!(
        error,
        smg_skills::SkillsStoreError::InvalidData(_)
    ));

    Ok(())
}

#[tokio::test]
async fn in_memory_service_requires_parent_skill_before_writing_versions() -> Result<()> {
    let blob_root = TempDir::new()?;
    let blob_store = create_blob_store(
        &BlobStoreConfig {
            backend: BlobStoreBackend::Filesystem,
            path: blob_root.path().display().to_string(),
            ..BlobStoreConfig::default()
        },
        None,
    )?;
    let service = SkillService::in_memory(blob_store);
    let metadata_store = service
        .metadata_store()
        .ok_or_else(|| anyhow!("metadata store missing"))?;

    let error = metadata_store
        .put_skill_version(SkillVersionRecord {
            skill_id: "missing-skill".to_string(),
            version: "1".to_string(),
            version_number: 1,
            name: "map".to_string(),
            short_description: None,
            description: "map".to_string(),
            interface: None,
            dependencies: None,
            policy: None,
            deprecated: false,
            file_manifest: Vec::new(),
            instruction_token_counts: Default::default(),
            created_at: Utc::now(),
        })
        .await
        .expect_err("writing a version without a parent skill should fail");

    assert!(matches!(
        error,
        smg_skills::SkillsStoreError::InvalidData(_)
    ));

    Ok(())
}

#[tokio::test]
async fn in_memory_service_deletes_versions_with_their_parent_skill() -> Result<()> {
    let blob_root = TempDir::new()?;
    let blob_store = create_blob_store(
        &BlobStoreConfig {
            backend: BlobStoreBackend::Filesystem,
            path: blob_root.path().display().to_string(),
            ..BlobStoreConfig::default()
        },
        None,
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
            short_description: None,
            description: None,
            source: "custom".to_string(),
            has_code_files: false,
            latest_version: Some("1".to_string()),
            default_version: Some("1".to_string()),
            created_at: now,
            updated_at: now,
        })
        .await?;
    metadata_store
        .put_skill_version(SkillVersionRecord {
            skill_id: "skill-1".to_string(),
            version: "1".to_string(),
            version_number: 1,
            name: "map".to_string(),
            short_description: None,
            description: "map".to_string(),
            interface: None,
            dependencies: None,
            policy: None,
            deprecated: false,
            file_manifest: Vec::new(),
            instruction_token_counts: Default::default(),
            created_at: now,
        })
        .await?;

    assert!(metadata_store.delete_skill("tenant-a", "skill-1").await?);
    assert!(metadata_store
        .get_skill("tenant-a", "skill-1")
        .await?
        .is_none());
    assert!(metadata_store
        .get_skill_version("skill-1", "1")
        .await?
        .is_none());

    Ok(())
}

#[tokio::test]
async fn in_memory_service_reindexes_reused_bundle_and_cookie_hashes() -> Result<()> {
    let blob_root = TempDir::new()?;
    let blob_store = create_blob_store(
        &BlobStoreConfig {
            backend: BlobStoreBackend::Filesystem,
            path: blob_root.path().display().to_string(),
            ..BlobStoreConfig::default()
        },
        None,
    )?;
    let service = SkillService::in_memory(blob_store);
    let now = Utc::now();

    let bundle_token_store = service
        .bundle_token_store()
        .ok_or_else(|| anyhow!("bundle token store missing"))?;
    bundle_token_store
        .put_bundle_token(BundleTokenClaim {
            token_hash: "tokhash".to_string(),
            tenant_id: "tenant-a".to_string(),
            exec_id: "exec-1".to_string(),
            skill_id: "skill-1".to_string(),
            skill_version: "1".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;
    bundle_token_store
        .put_bundle_token(BundleTokenClaim {
            token_hash: "tokhash".to_string(),
            tenant_id: "tenant-a".to_string(),
            exec_id: "exec-2".to_string(),
            skill_id: "skill-1".to_string(),
            skill_version: "1".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;

    assert_eq!(
        bundle_token_store
            .revoke_bundle_tokens_for_exec("exec-1")
            .await?,
        0
    );
    assert_eq!(
        bundle_token_store
            .get_bundle_token("tokhash")
            .await?
            .ok_or_else(|| anyhow!("bundle token missing"))?
            .exec_id,
        "exec-2"
    );
    assert_eq!(
        bundle_token_store
            .revoke_bundle_tokens_for_exec("exec-2")
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
            exec_id: "exec-3".to_string(),
            request_id: "req-1".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;
    continuation_cookie_store
        .put_continuation_cookie(ContinuationCookieClaim {
            cookie_hash: "cookiehash".to_string(),
            tenant_id: "tenant-a".to_string(),
            exec_id: "exec-4".to_string(),
            request_id: "req-1".to_string(),
            created_at: now,
            expires_at: now,
        })
        .await?;

    assert_eq!(
        continuation_cookie_store
            .revoke_continuation_cookies_for_exec("exec-3")
            .await?,
        0
    );
    assert_eq!(
        continuation_cookie_store
            .get_continuation_cookie("cookiehash")
            .await?
            .ok_or_else(|| anyhow!("continuation cookie missing"))?
            .exec_id,
        "exec-4"
    );
    assert_eq!(
        continuation_cookie_store
            .revoke_continuation_cookies_for_exec("exec-4")
            .await?,
        1
    );

    Ok(())
}
