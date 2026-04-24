use std::{
    collections::HashSet,
    fmt::Write as _,
    io::{Cursor, Read},
};

use serde::Deserialize;
use serde_yml::{Mapping, Value};
use thiserror::Error;
use zip::ZipArchive;

use crate::{
    config::SkillUploadLimits,
    types::{
        NormalizedSkillBundle, NormalizedSkillFile, ParsedSkillBundle, SkillDependencyTool,
        SkillInterfaceMetadata, SkillParseWarning, SkillParseWarningKind, SkillPolicyMetadata,
        SkillSidecarDependencies,
    },
};

const MAX_NAME_LEN: usize = 64;
const MAX_DISPLAY_NAME_LEN: usize = 64;
const MAX_DESCRIPTION_LEN: usize = 1024;
const MAX_SIDECAR_STRING_LEN: usize = 1024;
const MAX_BUNDLE_ARCHIVE_ENTRY_COUNT: usize = 1024;
const RESERVED_SKILL_NAMES: [&str; 3] = ["anthropic", "claude", "openai"];
const SKILL_MD_PATH: &str = "SKILL.md";
const OPENAI_SIDECAR_PATH: &str = "agents/openai.yaml";
const UNIX_FILE_TYPE_MASK: u32 = 0o170000;
const UNIX_FILE_TYPE_DIRECTORY: u32 = 0o040000;
const UNIX_FILE_TYPE_REGULAR: u32 = 0o100000;
const CODE_FILE_EXTENSIONS: &[&str] = &[
    "py", "sh", "bash", "zsh", "js", "mjs", "cjs", "ts", "mts", "cts", "rb", "lua", "pl", "php",
    "wasm", "r", "jl", "scala", "go", "rs",
];
const NON_CODE_FILE_EXTENSIONS: &[&str] = &[
    "md", "txt", "csv", "json", "yaml", "yml", "toml", "xml", "html", "css", "png", "jpg", "jpeg",
    "gif", "webp", "svg", "ico", "bmp", "tiff", "ttf", "otf", "woff", "woff2",
];
const CODE_FILE_BASENAMES: &[&str] = &["Dockerfile", "Makefile"];
const CODE_FILE_PREFIXES: &[&str] = &["run.", "main.", "exec."];

/// Errors returned while parsing `SKILL.md` and its optional OpenAI sidecar.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkillParseError {
    #[error("SKILL.md must begin with a YAML frontmatter block delimited by --- lines")]
    MissingFrontmatter,
    #[error("SKILL.md frontmatter is missing a closing --- delimiter")]
    MissingFrontmatterTerminator,
    #[error("SKILL.md frontmatter YAML is invalid: {message}")]
    InvalidFrontmatterYaml { message: String },
    #[error("SKILL.md field `{field}` is required")]
    MissingRequiredField { field: &'static str },
    #[error("SKILL.md field `{field}` is invalid: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
}

/// Errors returned while validating and normalizing uploaded skill-bundle zip
/// archives.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkillBundleArchiveError {
    #[error("skill bundle zip archive is invalid: {message}")]
    InvalidZip { message: String },
    #[error("skill bundle path `{path}` is invalid: {message}")]
    InvalidPath { path: String, message: String },
    #[error("skill bundle has multiple top-level directories: `{first}` and `{second}`")]
    MultipleTopLevelDirectories { first: String, second: String },
    #[error("skill bundle contains duplicate normalized path `{path}`")]
    DuplicateNormalizedPath { path: String },
    #[error("skill bundle contains unsupported entry type `{entry_type}` at `{path}`")]
    UnsupportedEntryType { path: String, entry_type: String },
    #[error("skill bundle contains more than {max_entries} archive entries")]
    TooManyEntries { max_entries: usize },
    #[error("skill bundle contains more than {max_files} regular files")]
    TooManyFiles { max_files: usize },
    #[error(
        "skill bundle entry `{path}` exceeds the maximum uncompressed size of {max_bytes} bytes"
    )]
    EntryTooLarge { path: String, max_bytes: u64 },
    #[error("skill bundle exceeds the maximum total uncompressed size of {max_bytes} bytes")]
    BundleTooLarge { max_bytes: u64 },
    #[error("skill bundle must contain exactly one root-level SKILL.md file")]
    MissingSkillMd,
    #[error("skill bundle contains multiple root-level SKILL.md files: `{first}` and `{second}`")]
    MultipleSkillMd { first: String, second: String },
    #[error("failed to read skill bundle entry `{path}`: {message}")]
    ReadEntry { path: String, message: String },
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    metadata: SkillFrontmatterMetadata,
}

#[derive(Debug, Default, Deserialize)]
struct SkillFrontmatterMetadata {
    #[serde(rename = "short-description")]
    short_description: Option<String>,
}

/// Parse a root-level `SKILL.md` plus optional `agents/openai.yaml` sidecar.
///
/// The sidecar is fail-open: invalid sidecar content is ignored and surfaced as
/// warnings instead of making the entire bundle invalid.
pub fn parse_skill_bundle(
    skill_md: &str,
    openai_yaml: Option<&str>,
) -> Result<ParsedSkillBundle, SkillParseError> {
    let (frontmatter_yaml, instructions_body) = split_frontmatter(skill_md)?;
    let frontmatter: SkillFrontmatter = serde_yml::from_str(frontmatter_yaml).map_err(|error| {
        SkillParseError::InvalidFrontmatterYaml {
            message: error.to_string(),
        }
    })?;

    let name = validate_required_name(frontmatter.name)?;
    let description = validate_required_description(frontmatter.description)?;
    let metadata_short_description = validate_optional_short_description(
        frontmatter.metadata.short_description,
        "metadata.short-description",
    )?;

    let mut warnings = Vec::new();
    let sidecar = parse_openai_sidecar(openai_yaml, &mut warnings);
    let short_description = sidecar
        .interface
        .as_ref()
        .and_then(|interface| interface.short_description.clone())
        .or(metadata_short_description);

    Ok(ParsedSkillBundle {
        name,
        description,
        short_description,
        instructions_body: instructions_body.to_owned(),
        interface: sidecar.interface,
        dependencies: sidecar.dependencies,
        policy: sidecar.policy,
        warnings,
    })
}

/// Normalize an uploaded skill-bundle zip archive into skill-root-relative files.
///
/// This strips the single required top-level directory, rejects unsafe entries,
/// and produces the canonical manifest shape used by later storage and upload
/// steps.
pub fn normalize_skill_bundle_zip(
    zip_bytes: &[u8],
) -> Result<NormalizedSkillBundle, SkillBundleArchiveError> {
    normalize_skill_bundle_zip_with_limits(zip_bytes, SkillUploadLimits::default())
}

/// Normalize an uploaded skill-bundle zip archive using explicit upload limits.
pub fn normalize_skill_bundle_zip_with_limits(
    zip_bytes: &[u8],
    limits: SkillUploadLimits,
) -> Result<NormalizedSkillBundle, SkillBundleArchiveError> {
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes)).map_err(|error| {
        SkillBundleArchiveError::InvalidZip {
            message: error.to_string(),
        }
    })?;
    if archive.len() > MAX_BUNDLE_ARCHIVE_ENTRY_COUNT {
        return Err(SkillBundleArchiveError::TooManyEntries {
            max_entries: MAX_BUNDLE_ARCHIVE_ENTRY_COUNT,
        });
    }

    let mut top_level_dir: Option<String> = None;
    let mut seen_relative_paths = HashSet::new();
    let mut seen_folded_relative_paths = HashSet::new();
    let mut files = Vec::new();
    let mut skill_md_path: Option<String> = None;
    let mut openai_sidecar_path = None;
    let mut total_uncompressed_size_bytes: u64 = 0;

    for index in 0..archive.len() {
        let entry =
            archive
                .by_index(index)
                .map_err(|error| SkillBundleArchiveError::InvalidZip {
                    message: format!("failed to read zip entry {index}: {error}"),
                })?;

        let original_path = entry.name().to_owned();
        let normalized_segments = normalize_zip_entry_path(entry.name_raw(), &original_path)?;
        if normalized_segments.is_empty() {
            continue;
        }

        let current_top_level = normalized_segments[0].clone();
        match &top_level_dir {
            Some(existing) if existing != &current_top_level => {
                return Err(SkillBundleArchiveError::MultipleTopLevelDirectories {
                    first: existing.clone(),
                    second: current_top_level,
                });
            }
            None => top_level_dir = Some(current_top_level),
            Some(_) => {}
        }

        let entry_type = classify_zip_entry_type(&entry, &original_path)?;
        let relative_segments = &normalized_segments[1..];
        if relative_segments.is_empty() {
            if matches!(entry_type, ZipEntryType::Directory) {
                continue;
            }
            return Err(SkillBundleArchiveError::InvalidPath {
                path: original_path,
                message: "archive entries must live under a single top-level directory".to_owned(),
            });
        }

        let relative_path = relative_segments.join("/");
        validate_special_bundle_paths(&relative_path)?;
        let is_skill_md = relative_path.eq_ignore_ascii_case(SKILL_MD_PATH);

        if !seen_relative_paths.insert(relative_path.clone()) {
            return Err(SkillBundleArchiveError::DuplicateNormalizedPath {
                path: relative_path,
            });
        }
        if !seen_folded_relative_paths.insert(relative_path.to_ascii_lowercase()) {
            if is_skill_md {
                if let Some(existing) = &skill_md_path {
                    return Err(SkillBundleArchiveError::MultipleSkillMd {
                        first: existing.clone(),
                        second: relative_path,
                    });
                }
            }
            return Err(SkillBundleArchiveError::DuplicateNormalizedPath {
                path: relative_path,
            });
        }

        match entry_type {
            ZipEntryType::Directory => continue,
            ZipEntryType::RegularFile => {
                if files.len() >= limits.max_files_per_version {
                    return Err(SkillBundleArchiveError::TooManyFiles {
                        max_files: limits.max_files_per_version,
                    });
                }

                let advertised_size_bytes = entry.size();
                if advertised_size_bytes > limits.max_file_size_bytes as u64 {
                    return Err(SkillBundleArchiveError::EntryTooLarge {
                        path: relative_path,
                        max_bytes: limits.max_file_size_bytes as u64,
                    });
                }

                let remaining_bundle_bytes = (limits.max_upload_size_bytes as u64)
                    .checked_sub(total_uncompressed_size_bytes)
                    .ok_or(SkillBundleArchiveError::BundleTooLarge {
                        max_bytes: limits.max_upload_size_bytes as u64,
                    })?;
                if advertised_size_bytes > remaining_bundle_bytes {
                    return Err(SkillBundleArchiveError::BundleTooLarge {
                        max_bytes: limits.max_upload_size_bytes as u64,
                    });
                }

                let max_read_bytes = remaining_bundle_bytes.min(limits.max_file_size_bytes as u64);
                let mut contents = Vec::new();
                let mut limited_entry = entry.take(max_read_bytes + 1);
                limited_entry.read_to_end(&mut contents).map_err(|error| {
                    SkillBundleArchiveError::ReadEntry {
                        path: relative_path.clone(),
                        message: error.to_string(),
                    }
                })?;
                let actual_size_bytes = u64::try_from(contents.len()).map_err(|_| {
                    SkillBundleArchiveError::ReadEntry {
                        path: relative_path.clone(),
                        message: "decompressed entry size overflowed u64".to_owned(),
                    }
                })?;
                if actual_size_bytes > limits.max_file_size_bytes as u64 {
                    return Err(SkillBundleArchiveError::EntryTooLarge {
                        path: relative_path,
                        max_bytes: limits.max_file_size_bytes as u64,
                    });
                }
                total_uncompressed_size_bytes = total_uncompressed_size_bytes
                    .checked_add(actual_size_bytes)
                    .ok_or(SkillBundleArchiveError::BundleTooLarge {
                        max_bytes: limits.max_upload_size_bytes as u64,
                    })?;
                if total_uncompressed_size_bytes > limits.max_upload_size_bytes as u64 {
                    return Err(SkillBundleArchiveError::BundleTooLarge {
                        max_bytes: limits.max_upload_size_bytes as u64,
                    });
                }

                if is_skill_md {
                    if let Some(existing) = &skill_md_path {
                        return Err(SkillBundleArchiveError::MultipleSkillMd {
                            first: existing.clone(),
                            second: relative_path.clone(),
                        });
                    }
                    skill_md_path = Some(relative_path.clone());
                }

                if relative_path.eq_ignore_ascii_case(OPENAI_SIDECAR_PATH) {
                    openai_sidecar_path = Some(relative_path.clone());
                }

                files.push(NormalizedSkillFile {
                    relative_path,
                    contents,
                });
            }
        }
    }

    let skill_md_path = skill_md_path.ok_or(SkillBundleArchiveError::MissingSkillMd)?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let has_code_files = files
        .iter()
        .any(|file| is_code_file_path(&file.relative_path));

    Ok(NormalizedSkillBundle {
        files,
        skill_md_path,
        openai_sidecar_path,
        has_code_files,
    })
}

/// Deterministically classify whether a bundle file should count as code for
/// `has_code_files`.
///
/// This intentionally errs on the side of false positives: basename matches for
/// `run.*`, `main.*`, and `exec.*` win even if the extension would otherwise be
/// considered non-code.
pub fn is_code_file_path(path: &str) -> bool {
    let basename = path.rsplit('/').next().unwrap_or(path);
    if CODE_FILE_BASENAMES
        .iter()
        .any(|candidate| basename.eq_ignore_ascii_case(candidate))
        || CODE_FILE_PREFIXES.iter().any(|prefix| {
            basename.len() > prefix.len()
                && basename
                    .get(..prefix.len())
                    .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
        })
    {
        return true;
    }

    let Some((_, extension)) = basename.rsplit_once('.') else {
        return false;
    };

    if NON_CODE_FILE_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        return false;
    }

    CODE_FILE_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZipEntryType {
    Directory,
    RegularFile,
}

fn normalize_zip_entry_path(
    raw_name: &[u8],
    display_name: &str,
) -> Result<Vec<String>, SkillBundleArchiveError> {
    if raw_name.contains(&0) {
        return Err(SkillBundleArchiveError::InvalidPath {
            path: display_name.to_owned(),
            message: "must not contain NUL bytes".to_owned(),
        });
    }

    let normalized_name = display_name.replace('\\', "/");
    if normalized_name.starts_with('/') {
        return Err(SkillBundleArchiveError::InvalidPath {
            path: display_name.to_owned(),
            message: "must not be absolute".to_owned(),
        });
    }

    let segments = normalized_name.split('/').collect::<Vec<_>>();
    let trailing_empty_segments = segments
        .iter()
        .rev()
        .take_while(|segment| segment.is_empty())
        .count();
    if trailing_empty_segments > 1 {
        return Err(SkillBundleArchiveError::InvalidPath {
            path: display_name.to_owned(),
            message: "must not contain empty path segments".to_owned(),
        });
    }

    let meaningful_len = segments.len().saturating_sub(trailing_empty_segments);
    if meaningful_len == 0 {
        return Ok(Vec::new());
    }

    let mut normalized_segments = Vec::with_capacity(meaningful_len);
    for segment in &segments[..meaningful_len] {
        match *segment {
            "" => {
                return Err(SkillBundleArchiveError::InvalidPath {
                    path: display_name.to_owned(),
                    message: "must not contain empty path segments".to_owned(),
                });
            }
            "." => continue,
            ".." => {
                return Err(SkillBundleArchiveError::InvalidPath {
                    path: display_name.to_owned(),
                    message: "must not contain path traversal".to_owned(),
                });
            }
            _ if is_windows_drive_prefix(segment) => {
                return Err(SkillBundleArchiveError::InvalidPath {
                    path: display_name.to_owned(),
                    message: "must not be absolute".to_owned(),
                });
            }
            _ => normalized_segments.push((*segment).to_owned()),
        }
    }

    if normalized_segments.is_empty() {
        return Err(SkillBundleArchiveError::InvalidPath {
            path: display_name.to_owned(),
            message: "must not resolve to an empty path".to_owned(),
        });
    }

    Ok(normalized_segments)
}

fn classify_zip_entry_type<R: Read + ?Sized>(
    entry: &zip::read::ZipFile<'_, R>,
    original_path: &str,
) -> Result<ZipEntryType, SkillBundleArchiveError> {
    if entry.is_dir() {
        return Ok(ZipEntryType::Directory);
    }
    if entry.is_symlink() {
        return Err(SkillBundleArchiveError::UnsupportedEntryType {
            path: original_path.to_owned(),
            entry_type: "symbolic link".to_owned(),
        });
    }
    if let Some(mode) = entry.unix_mode() {
        match mode & UNIX_FILE_TYPE_MASK {
            0 => {}
            UNIX_FILE_TYPE_DIRECTORY => return Ok(ZipEntryType::Directory),
            UNIX_FILE_TYPE_REGULAR => {}
            file_type => {
                return Err(SkillBundleArchiveError::UnsupportedEntryType {
                    path: original_path.to_owned(),
                    entry_type: describe_unix_file_type(file_type).to_owned(),
                });
            }
        }
    }
    if entry.is_file() {
        return Ok(ZipEntryType::RegularFile);
    }

    Err(SkillBundleArchiveError::UnsupportedEntryType {
        path: original_path.to_owned(),
        entry_type: "non-regular file".to_owned(),
    })
}

fn validate_special_bundle_paths(relative_path: &str) -> Result<(), SkillBundleArchiveError> {
    let basename = relative_path.rsplit('/').next().unwrap_or(relative_path);
    if basename.eq_ignore_ascii_case(SKILL_MD_PATH)
        && !relative_path.eq_ignore_ascii_case(SKILL_MD_PATH)
    {
        return Err(SkillBundleArchiveError::InvalidPath {
            path: relative_path.to_owned(),
            message: "SKILL.md must live at the skill root".to_owned(),
        });
    }
    Ok(())
}

fn is_windows_drive_prefix(segment: &str) -> bool {
    segment.len() == 2
        && segment.as_bytes()[0].is_ascii_alphabetic()
        && segment.as_bytes()[1] == b':'
}

fn describe_unix_file_type(file_type: u32) -> &'static str {
    match file_type {
        0o020000 => "character device",
        0o060000 => "block device",
        0o010000 => "named pipe",
        0o140000 => "socket",
        _ => "non-regular file",
    }
}

fn split_frontmatter(skill_md: &str) -> Result<(&str, &str), SkillParseError> {
    let Some(first_line_end) = find_line_end(skill_md, 0) else {
        return Err(SkillParseError::MissingFrontmatter);
    };

    if trim_line_ending(&skill_md[..first_line_end]) != "---" {
        return Err(SkillParseError::MissingFrontmatter);
    }

    let yaml_start = first_line_end;
    let mut cursor = yaml_start;
    while cursor < skill_md.len() {
        let next_line_end = find_line_end(skill_md, cursor).unwrap_or(skill_md.len());
        let line = trim_line_ending(&skill_md[cursor..next_line_end]);
        if line == "---" {
            let body = &skill_md[next_line_end..];
            let frontmatter = &skill_md[yaml_start..cursor];
            return Ok((frontmatter, body));
        }
        cursor = next_line_end;
    }

    Err(SkillParseError::MissingFrontmatterTerminator)
}

fn find_line_end(content: &str, start: usize) -> Option<usize> {
    if start >= content.len() {
        return None;
    }

    content[start..]
        .find('\n')
        .map(|offset| start + offset + 1)
        .or(Some(content.len()))
}

fn trim_line_ending(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r'])
}

fn validate_required_name(name: Option<String>) -> Result<String, SkillParseError> {
    let name = name.ok_or(SkillParseError::MissingRequiredField { field: "name" })?;
    validate_name(&name)?;
    Ok(name)
}

fn validate_required_description(description: Option<String>) -> Result<String, SkillParseError> {
    let description = description.ok_or(SkillParseError::MissingRequiredField {
        field: "description",
    })?;
    validate_description(&description)?;
    Ok(description)
}

fn validate_optional_short_description(
    value: Option<String>,
    field: &'static str,
) -> Result<Option<String>, SkillParseError> {
    match value {
        Some(short_description) => {
            validate_length(field, &short_description, MAX_DESCRIPTION_LEN)?;
            Ok(Some(short_description))
        }
        None => Ok(None),
    }
}

fn validate_name(name: &str) -> Result<(), SkillParseError> {
    if name.is_empty() {
        return Err(SkillParseError::InvalidField {
            field: "name",
            message: "must not be empty".to_owned(),
        });
    }
    if name.len() > MAX_NAME_LEN {
        return Err(SkillParseError::InvalidField {
            field: "name",
            message: format!("must be at most {MAX_NAME_LEN} characters"),
        });
    }
    for segment in name.split(':') {
        if RESERVED_SKILL_NAMES
            .iter()
            .any(|reserved| reserved.eq_ignore_ascii_case(segment))
        {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "is reserved".to_owned(),
            });
        }
        if segment.is_empty() {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "must not contain empty namespace segments".to_owned(),
            });
        }

        let mut chars = segment.chars();
        let first = chars.next().unwrap_or('\0');

        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "must start each namespace segment with a lowercase letter or digit"
                    .to_owned(),
            });
        }

        if chars.any(|ch| !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '-') {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message:
                    "may only contain lowercase letters, digits, hyphens, and namespace colons"
                        .to_owned(),
            });
        }
    }

    Ok(())
}

fn validate_description(description: &str) -> Result<(), SkillParseError> {
    if description.trim().is_empty() {
        return Err(SkillParseError::InvalidField {
            field: "description",
            message: "must not be empty".to_owned(),
        });
    }
    validate_length("description", description, MAX_DESCRIPTION_LEN)?;
    if contains_xml_like_tag(description) {
        return Err(SkillParseError::InvalidField {
            field: "description",
            message: "must not contain XML-like tags".to_owned(),
        });
    }
    Ok(())
}

fn validate_length(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), SkillParseError> {
    if value.chars().count() > max_len {
        return Err(SkillParseError::InvalidField {
            field,
            message: format!("must be at most {max_len} characters"),
        });
    }
    Ok(())
}

fn contains_xml_like_tag(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'<' {
            index += 1;
            continue;
        }

        let mut tag_index = index + 1;
        if tag_index < bytes.len() && bytes[tag_index] == b'/' {
            tag_index += 1;
        }

        if tag_index >= bytes.len() || !bytes[tag_index].is_ascii_alphabetic() {
            index += 1;
            continue;
        }

        tag_index += 1;
        while tag_index < bytes.len() {
            match bytes[tag_index] {
                b'>' => return true,
                b'<' => break,
                _ => {
                    tag_index += 1;
                }
            }
        }

        if tag_index >= bytes.len() {
            return false;
        }

        index = tag_index;
    }
    false
}

#[derive(Default)]
struct ParsedOpenAISidecar {
    interface: Option<SkillInterfaceMetadata>,
    dependencies: Option<SkillSidecarDependencies>,
    policy: Option<SkillPolicyMetadata>,
}

fn parse_openai_sidecar(
    openai_yaml: Option<&str>,
    warnings: &mut Vec<SkillParseWarning>,
) -> ParsedOpenAISidecar {
    let Some(openai_yaml) = openai_yaml else {
        return ParsedOpenAISidecar::default();
    };

    let value = match serde_yml::from_str::<Value>(openai_yaml) {
        Ok(value) => value,
        Err(error) => {
            warnings.push(SkillParseWarning {
                kind: SkillParseWarningKind::SidecarFileIgnored,
                path: "agents/openai.yaml".to_owned(),
                message: format!("ignored invalid YAML: {error}"),
            });
            return ParsedOpenAISidecar::default();
        }
    };

    let Some(mapping) = value.as_mapping() else {
        warnings.push(SkillParseWarning {
            kind: SkillParseWarningKind::SidecarFieldIgnored,
            path: "agents/openai.yaml".to_owned(),
            message: "expected a mapping at the YAML document root".to_owned(),
        });
        return ParsedOpenAISidecar::default();
    };

    let interface = parse_interface(mapping_get(mapping, "interface"), warnings);
    let dependencies = parse_dependencies(mapping_get(mapping, "dependencies"), warnings);
    let policy = parse_policy(mapping_get(mapping, "policy"), warnings);

    ParsedOpenAISidecar {
        interface,
        dependencies,
        policy,
    }
}

fn parse_interface(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillInterfaceMetadata> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.interface",
            "expected a mapping",
        );
        return None;
    };

    let interface = SkillInterfaceMetadata {
        display_name: parse_string_field(
            mapping,
            "display_name",
            "agents/openai.yaml.interface.display_name",
            warnings,
            |value| {
                validate_sidecar_string_len("interface.display_name", value, MAX_DISPLAY_NAME_LEN)
            },
        ),
        short_description: parse_string_field(
            mapping,
            "short_description",
            "agents/openai.yaml.interface.short_description",
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "interface.short_description",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        icon_small: parse_string_field(
            mapping,
            "icon_small",
            "agents/openai.yaml.interface.icon_small",
            warnings,
            validate_sidecar_path,
        ),
        icon_large: parse_string_field(
            mapping,
            "icon_large",
            "agents/openai.yaml.interface.icon_large",
            warnings,
            validate_sidecar_path,
        ),
        brand_color: parse_string_field(
            mapping,
            "brand_color",
            "agents/openai.yaml.interface.brand_color",
            warnings,
            validate_brand_color,
        ),
        default_prompt: parse_string_field(
            mapping,
            "default_prompt",
            "agents/openai.yaml.interface.default_prompt",
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "interface.default_prompt",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
    };

    (!interface.is_empty()).then_some(interface)
}

fn parse_dependencies(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillSidecarDependencies> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.dependencies",
            "expected a mapping",
        );
        return None;
    };

    let tools_value = mapping_get(mapping, "tools")?;
    let Some(sequence) = tools_value.as_sequence() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.dependencies.tools",
            "expected a sequence",
        );
        return None;
    };

    let tools = sequence
        .iter()
        .enumerate()
        .filter_map(|(index, value)| parse_dependency_tool(value, index, warnings))
        .collect::<Vec<_>>();

    let dependencies = SkillSidecarDependencies { tools };
    (!dependencies.is_empty()).then_some(dependencies)
}

fn parse_dependency_tool(
    value: &Value,
    index: usize,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillDependencyTool> {
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            &format!("agents/openai.yaml.dependencies.tools[{index}]"),
            "expected a mapping",
        );
        return None;
    };

    let path = format!("agents/openai.yaml.dependencies.tools[{index}]");
    let tool_type = parse_required_string_field(mapping, "type", &path, warnings, |value| {
        validate_sidecar_string_len("dependencies.tools[].type", value, MAX_SIDECAR_STRING_LEN)
    })?;
    let value = parse_required_string_field(mapping, "value", &path, warnings, |value| {
        validate_sidecar_string_len("dependencies.tools[].value", value, MAX_SIDECAR_STRING_LEN)
    })?;

    Some(SkillDependencyTool {
        tool_type,
        value,
        description: parse_string_field(
            mapping,
            "description",
            &format!("{path}.description"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].description",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        transport: parse_string_field(
            mapping,
            "transport",
            &format!("{path}.transport"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].transport",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        command: parse_string_field(
            mapping,
            "command",
            &format!("{path}.command"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].command",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        url: parse_string_field(mapping, "url", &format!("{path}.url"), warnings, |value| {
            validate_sidecar_string_len("dependencies.tools[].url", value, MAX_SIDECAR_STRING_LEN)
        }),
    })
}

fn parse_policy(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillPolicyMetadata> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(warnings, "agents/openai.yaml.policy", "expected a mapping");
        return None;
    };

    let allow_implicit_invocation = parse_bool_field(
        mapping,
        "allow_implicit_invocation",
        "agents/openai.yaml.policy.allow_implicit_invocation",
        warnings,
    );
    let products = parse_string_sequence_field(
        mapping,
        "products",
        "agents/openai.yaml.policy.products",
        warnings,
    );

    let policy = SkillPolicyMetadata {
        allow_implicit_invocation,
        products,
    };
    (!policy.is_empty()).then_some(policy)
}

fn parse_string_field<F>(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
    validator: F,
) -> Option<String>
where
    F: Fn(&str) -> Result<(), String>,
{
    let value = mapping_get(mapping, key)?;
    match value {
        Value::String(string_value) => match validator(string_value) {
            Ok(()) => Some(string_value.clone()),
            Err(message) => {
                push_field_warning(warnings, path, &message);
                None
            }
        },
        _ => {
            push_field_warning(warnings, path, "expected a string");
            None
        }
    }
}

fn parse_required_string_field<F>(
    mapping: &Mapping,
    key: &str,
    parent_path: &str,
    warnings: &mut Vec<SkillParseWarning>,
    validator: F,
) -> Option<String>
where
    F: Fn(&str) -> Result<(), String>,
{
    let mut field_path = String::from(parent_path);
    let _ = write!(&mut field_path, ".{key}");
    let Some(value) = mapping_get(mapping, key) else {
        push_field_warning(warnings, &field_path, "missing required field");
        return None;
    };
    match value {
        Value::String(string_value) => match validator(string_value) {
            Ok(()) => Some(string_value.clone()),
            Err(message) => {
                push_field_warning(warnings, &field_path, &message);
                None
            }
        },
        _ => {
            push_field_warning(warnings, &field_path, "expected a string");
            None
        }
    }
}

fn parse_bool_field(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<bool> {
    let value = mapping_get(mapping, key)?;
    match value {
        Value::Bool(boolean_value) => Some(*boolean_value),
        _ => {
            push_field_warning(warnings, path, "expected a boolean");
            None
        }
    }
}

fn parse_string_sequence_field(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
) -> Vec<String> {
    let Some(value) = mapping_get(mapping, key) else {
        return Vec::new();
    };
    let Some(sequence) = value.as_sequence() else {
        push_field_warning(warnings, path, "expected a sequence");
        return Vec::new();
    };

    sequence
        .iter()
        .enumerate()
        .filter_map(|(index, value)| match value {
            Value::String(string_value) => Some(string_value.clone()),
            _ => {
                push_field_warning(warnings, &format!("{path}[{index}]"), "expected a string");
                None
            }
        })
        .collect()
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.iter().find_map(|(candidate, value)| {
        candidate
            .as_str()
            .filter(|candidate_key| *candidate_key == key)
            .map(|_| value)
    })
}

fn push_field_warning(warnings: &mut Vec<SkillParseWarning>, path: &str, message: &str) {
    warnings.push(SkillParseWarning {
        kind: SkillParseWarningKind::SidecarFieldIgnored,
        path: path.to_owned(),
        message: message.to_owned(),
    });
}

fn validate_sidecar_string_len(field: &str, value: &str, max_len: usize) -> Result<(), String> {
    if value.chars().count() > max_len {
        return Err(format!("{field} must be at most {max_len} characters"));
    }
    Ok(())
}

fn validate_brand_color(value: &str) -> Result<(), String> {
    if value.len() != 7 {
        return Err("brand color must be a CSS hex color like #1D4ED8".to_owned());
    }
    let bytes = value.as_bytes();
    if bytes.first() != Some(&b'#') || bytes[1..].iter().any(|byte| !byte.is_ascii_hexdigit()) {
        return Err("brand color must be a CSS hex color like #1D4ED8".to_owned());
    }
    Ok(())
}

fn validate_sidecar_path(value: &str) -> Result<(), String> {
    validate_sidecar_string_len("interface.icon", value, MAX_SIDECAR_STRING_LEN)?;
    if value.is_empty() {
        return Err("path must not be empty".to_owned());
    }
    if value.starts_with('/') || value.starts_with('\\') {
        return Err("path must be relative to the skill root".to_owned());
    }
    if value.contains('\0') {
        return Err("path must not contain NUL bytes".to_owned());
    }
    if value.contains('\\') {
        return Err("path must use forward slashes".to_owned());
    }

    let mut segments = value.split('/').peekable();
    let mut index = 0;
    while let Some(segment) = segments.next() {
        if segment.is_empty() {
            return Err("path must not contain empty segments".to_owned());
        }
        if segment == "." {
            if index == 0 && segments.peek().is_some() {
                index += 1;
                continue;
            }
            return Err("path must not contain standalone current-directory segments".to_owned());
        }
        if segment == ".." {
            return Err("path must not contain traversal segments".to_owned());
        }
        index += 1;
    }

    Ok(())
}
