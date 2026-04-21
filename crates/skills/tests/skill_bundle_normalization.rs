use std::io::{Cursor, Write};

use anyhow::{bail, Result};
use smg_skills::{
    is_code_file_path, normalize_skill_bundle_zip, NormalizedSkillBundle, SkillBundleArchiveError,
};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

enum TestZipEntry<'a> {
    File {
        path: &'a str,
        contents: &'a [u8],
        unix_mode: Option<u32>,
    },
    Directory {
        path: &'a str,
        unix_mode: Option<u32>,
    },
    Symlink {
        path: &'a str,
        target: &'a str,
    },
}

fn build_zip(entries: &[TestZipEntry<'_>]) -> Result<Vec<u8>> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(cursor);

    for entry in entries {
        match entry {
            TestZipEntry::File {
                path,
                contents,
                unix_mode,
            } => {
                let mut options =
                    SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
                if let Some(mode) = unix_mode {
                    options = options.unix_permissions(*mode);
                }
                writer.start_file(path, options)?;
                writer.write_all(contents)?;
            }
            TestZipEntry::Directory { path, unix_mode } => {
                let mut options =
                    SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
                if let Some(mode) = unix_mode {
                    options = options.unix_permissions(*mode);
                }
                writer.add_directory(*path, options)?;
            }
            TestZipEntry::Symlink { path, target } => {
                let options =
                    SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
                writer.add_symlink(path, target, options)?;
            }
        }
    }

    Ok(writer.finish()?.into_inner())
}

fn normalize(
    entries: &[TestZipEntry<'_>],
) -> Result<NormalizedSkillBundle, SkillBundleArchiveError> {
    let zip_bytes = build_zip(entries).map_err(|error| SkillBundleArchiveError::InvalidZip {
        message: error.to_string(),
    })?;
    normalize_skill_bundle_zip(&zip_bytes)
}

#[test]
fn normalizes_zip_paths_to_skill_root_manifest() -> Result<()> {
    let normalized = normalize(&[
        TestZipEntry::Directory {
            path: "gh-fix-ci/",
            unix_mode: Some(0o040755),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/agents/openai.yaml",
            contents: b"interface: {}",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/scripts/analyze.py",
            contents: b"print('hello')",
            unix_mode: Some(0o100755),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/assets/readme.txt",
            contents: b"not code",
            unix_mode: Some(0o100644),
        },
    ])?;

    let paths = normalized
        .files
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec![
            "SKILL.md",
            "agents/openai.yaml",
            "assets/readme.txt",
            "scripts/analyze.py",
        ]
    );
    assert_eq!(normalized.skill_md_path, "SKILL.md");
    assert_eq!(
        normalized.openai_sidecar_path.as_deref(),
        Some("agents/openai.yaml")
    );
    assert!(normalized.has_code_files);

    let manifest_paths = normalized
        .file_manifest()
        .into_iter()
        .map(|file| file.relative_path)
        .collect::<Vec<_>>();
    assert_eq!(manifest_paths, paths);
    Ok(())
}

#[test]
fn recognizes_openai_sidecar_case_insensitively() -> Result<()> {
    let normalized = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/agents/OpenAI.YAML",
            contents: b"interface: {}",
            unix_mode: Some(0o100644),
        },
    ])?;

    assert_eq!(
        normalized.openai_sidecar_path.as_deref(),
        Some("agents/OpenAI.YAML")
    );
    Ok(())
}

#[test]
fn rejects_archives_without_a_single_top_level_directory() {
    let error = normalize(&[
        TestZipEntry::File {
            path: "first/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "second/scripts/analyze.py",
            contents: b"print('hello')",
            unix_mode: Some(0o100755),
        },
    ])
    .expect_err("multiple top-level directories must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::MultipleTopLevelDirectories {
            first: "first".to_owned(),
            second: "second".to_owned(),
        }
    );
}

#[test]
fn rejects_root_level_files_without_top_level_directory() {
    let error = normalize(&[TestZipEntry::File {
        path: "SKILL.md",
        contents: b"skill body",
        unix_mode: Some(0o100644),
    }])
    .expect_err("root-level file must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::InvalidPath {
            path: "SKILL.md".to_owned(),
            message: "archive entries must live under a single top-level directory".to_owned(),
        }
    );
}

#[test]
fn rejects_path_traversal_and_absolute_paths() {
    let traversal = normalize(&[TestZipEntry::File {
        path: "gh-fix-ci/../SKILL.md",
        contents: b"skill body",
        unix_mode: Some(0o100644),
    }])
    .expect_err("path traversal must fail");
    assert_eq!(
        traversal,
        SkillBundleArchiveError::InvalidPath {
            path: "gh-fix-ci/../SKILL.md".to_owned(),
            message: "must not contain path traversal".to_owned(),
        }
    );

    let absolute = normalize(&[TestZipEntry::File {
        path: "/gh-fix-ci/SKILL.md",
        contents: b"skill body",
        unix_mode: Some(0o100644),
    }])
    .expect_err("absolute path must fail");
    assert_eq!(
        absolute,
        SkillBundleArchiveError::InvalidPath {
            path: "/gh-fix-ci/SKILL.md".to_owned(),
            message: "must not be absolute".to_owned(),
        }
    );
}

#[test]
fn rejects_duplicate_normalized_paths() {
    let error = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci\\scripts\\analyze.py",
            contents: b"print('one')",
            unix_mode: Some(0o100755),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/scripts/analyze.py",
            contents: b"print('two')",
            unix_mode: Some(0o100755),
        },
    ])
    .expect_err("duplicate normalized path must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::DuplicateNormalizedPath {
            path: "scripts/analyze.py".to_owned(),
        }
    );
}

#[test]
fn rejects_case_only_duplicate_normalized_paths() {
    let error = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/scripts/Run.py",
            contents: b"print('one')",
            unix_mode: Some(0o100755),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/scripts/run.py",
            contents: b"print('two')",
            unix_mode: Some(0o100755),
        },
    ])
    .expect_err("case-only duplicate normalized path must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::DuplicateNormalizedPath {
            path: "scripts/run.py".to_owned(),
        }
    );
}

#[test]
fn rejects_symlinks() {
    let symlink = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::Symlink {
            path: "gh-fix-ci/scripts/link.py",
            target: "../scripts/analyze.py",
        },
    ])
    .expect_err("symlink must fail");
    assert_eq!(
        symlink,
        SkillBundleArchiveError::UnsupportedEntryType {
            path: "gh-fix-ci/scripts/link.py".to_owned(),
            entry_type: "symbolic link".to_owned(),
        }
    );
}

#[test]
fn rejects_multiple_root_level_skill_md_files_case_insensitively() {
    let error = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/skill.md",
            contents: b"duplicate",
            unix_mode: Some(0o100644),
        },
    ])
    .expect_err("duplicate root-level SKILL.md must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::MultipleSkillMd {
            first: "SKILL.md".to_owned(),
            second: "skill.md".to_owned(),
        }
    );
}

#[test]
fn rejects_nested_skill_md() {
    let error = normalize(&[TestZipEntry::File {
        path: "gh-fix-ci/docs/SKILL.md",
        contents: b"skill body",
        unix_mode: Some(0o100644),
    }])
    .expect_err("nested SKILL.md must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::InvalidPath {
            path: "docs/SKILL.md".to_owned(),
            message: "SKILL.md must live at the skill root".to_owned(),
        }
    );
}

#[test]
fn rejects_archives_without_root_level_skill_md() {
    let error = normalize(&[TestZipEntry::File {
        path: "gh-fix-ci/notes.txt",
        contents: b"docs only",
        unix_mode: Some(0o100644),
    }])
    .expect_err("missing SKILL.md must fail");

    assert_eq!(error, SkillBundleArchiveError::MissingSkillMd);
}

#[test]
fn rejects_more_than_max_regular_files() {
    let dynamic_paths = (0..500)
        .map(|index| format!("gh-fix-ci/assets/file-{index}.txt"))
        .collect::<Vec<_>>();
    let mut entries = Vec::with_capacity(501);
    entries.push(TestZipEntry::File {
        path: "gh-fix-ci/SKILL.md",
        contents: b"skill body",
        unix_mode: Some(0o100644),
    });
    for path in &dynamic_paths {
        entries.push(TestZipEntry::File {
            path,
            contents: b"payload",
            unix_mode: Some(0o100644),
        });
    }

    let error = normalize(&entries).expect_err("file count limit must fail");
    assert_eq!(
        error,
        SkillBundleArchiveError::TooManyFiles { max_files: 500 }
    );
}

#[test]
fn rejects_too_many_archive_entries() -> Result<()> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for index in 0..1024 {
        writer.add_directory(format!("gh-fix-ci/dir-{index}/"), options)?;
    }
    writer.start_file("gh-fix-ci/SKILL.md", options)?;
    writer.write_all(b"skill body")?;

    let zip_bytes = writer.finish()?.into_inner();
    let error = normalize_skill_bundle_zip(&zip_bytes).expect_err("entry count limit must fail");
    assert_eq!(
        error,
        SkillBundleArchiveError::TooManyEntries { max_entries: 1024 }
    );
    Ok(())
}

#[test]
fn rejects_entry_larger_than_max_uncompressed_size() {
    let oversized = vec![b'x'; 25 * 1024 * 1024 + 1];
    let error = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/assets/huge.bin",
            contents: &oversized,
            unix_mode: Some(0o100644),
        },
    ])
    .expect_err("oversized entry must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::EntryTooLarge {
            path: "assets/huge.bin".to_owned(),
            max_bytes: 25 * 1024 * 1024,
        }
    );
}

#[test]
fn rejects_total_uncompressed_bundle_size_larger_than_limit() {
    let large_payload = vec![b'x'; 15 * 1024 * 1024];
    let error = normalize(&[
        TestZipEntry::File {
            path: "gh-fix-ci/SKILL.md",
            contents: b"skill body",
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/assets/part-1.bin",
            contents: &large_payload,
            unix_mode: Some(0o100644),
        },
        TestZipEntry::File {
            path: "gh-fix-ci/assets/part-2.bin",
            contents: &large_payload,
            unix_mode: Some(0o100644),
        },
    ])
    .expect_err("oversized bundle must fail");

    assert_eq!(
        error,
        SkillBundleArchiveError::BundleTooLarge {
            max_bytes: 30 * 1024 * 1024,
        }
    );
}

#[test]
fn classifies_code_files_deterministically() -> Result<()> {
    let positives = [
        "scripts/analyze.py",
        "scripts/ANALYZE.PY",
        "src/tool.rs",
        "assets/module.wasm",
        "Dockerfile",
        "dockerfile",
        "Makefile",
        "config/run.json",
        "images/main.svg",
        "notes/exec.txt",
    ];
    for path in positives {
        if !is_code_file_path(path) {
            bail!("expected `{path}` to classify as code");
        }
    }

    let negatives = [
        "SKILL.md",
        "agents/openai.yaml",
        "docs/readme.txt",
        "assets/data.json",
        "assets/logo.png",
        "styles/site.css",
        "notes/report",
        "assets/😀.txt",
    ];
    for path in negatives {
        if is_code_file_path(path) {
            bail!("expected `{path}` to classify as non-code");
        }
    }
    Ok(())
}

#[test]
fn surfaces_invalid_zip_payloads() -> Result<()> {
    let error = normalize_skill_bundle_zip(b"not-a-zip").expect_err("invalid zip must fail");
    match error {
        SkillBundleArchiveError::InvalidZip { message } => {
            if message.is_empty() {
                bail!("expected zip error message");
            }
        }
        other => bail!("expected invalid zip error, got {other:?}"),
    }
    Ok(())
}
