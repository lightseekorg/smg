use std::{
    io::{Cursor, Read, Seek},
    path::{Component, Path},
};

use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tracing::warn;
use zip::ZipArchive;

use crate::stream_bundle::{StreamBundle, MAX_UNCOMPRESSED_SIZE, MAX_ZIP_ENTRIES};

fn checked_add_uncompressed_size(total: u64, entry_size: u64) -> Result<u64, String> {
    total
        .checked_add(entry_size)
        .ok_or_else(|| "Zip archive total uncompressed size overflowed u64".to_string())
}

/// Temporary extracted bundle directory with explicit cleanup support.
pub struct ExtractedArchiveDir {
    temp_dir: TempDir,
}

impl ExtractedArchiveDir {
    pub fn path(&self) -> &Path {
        self.temp_dir.path()
    }

    pub fn cleanup(self) -> Result<(), String> {
        let path = self.temp_dir.path().to_string_lossy().into_owned();
        self.temp_dir
            .close()
            .map_err(|e| format!("failed to cleanup temp dir '{}': {}", path, e))
    }
}

pub fn validate_bundle_sha256(bundle: &StreamBundle) -> Result<(), String> {
    if bundle.sha256.is_empty() {
        return Ok(());
    }

    let computed = format!("{:x}", Sha256::digest(&bundle.compressed_data));
    if !computed.eq_ignore_ascii_case(&bundle.sha256) {
        return Err(format!(
            "Bundle fingerprint mismatch: expected {}, got {}",
            bundle.sha256, computed
        ));
    }
    Ok(())
}

pub fn validate_zip_archive<R: Read + Seek>(reader: R) -> Result<ZipArchive<R>, String> {
    let mut archive =
        ZipArchive::new(reader).map_err(|e| format!("Failed to open zip archive: {}", e))?;

    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(format!(
            "Zip archive has too many entries ({} > {})",
            archive.len(),
            MAX_ZIP_ENTRIES
        ));
    }

    let mut total_uncompressed: u64 = 0;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read zip entry {}: {}", i, e))?;
        let path = entry.name();
        let has_traversal = Path::new(path).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
        if has_traversal {
            return Err(format!("Zip archive contains unsafe path: {}", path));
        }
        total_uncompressed = checked_add_uncompressed_size(total_uncompressed, entry.size())?;
    }

    if total_uncompressed > MAX_UNCOMPRESSED_SIZE {
        return Err(format!(
            "Zip archive uncompressed size too large ({} bytes > {} bytes)",
            total_uncompressed, MAX_UNCOMPRESSED_SIZE
        ));
    }

    Ok(archive)
}

pub fn extract_bundle_to_tempdir(bundle: &StreamBundle) -> Result<ExtractedArchiveDir, String> {
    let mut archive = validate_zip_archive(Cursor::new(bundle.compressed_data.as_slice()))?;
    let dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {}", e))?;
    archive
        .extract(dir.path())
        .map_err(|e| format!("archive extraction failed: {}", e))?;

    Ok(ExtractedArchiveDir { temp_dir: dir })
}

pub fn with_extracted_bundle<R>(
    bundle: &StreamBundle,
    operation: impl FnOnce(&Path) -> Result<R, String>,
) -> Result<R, String> {
    let extracted = extract_bundle_to_tempdir(bundle)?;
    let result = operation(extracted.path());

    if let Err(e) = extracted.cleanup() {
        warn!("Bundle extraction tempdir cleanup failed: {}", e);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use sha2::{Digest, Sha256};
    use zip::{write::SimpleFileOptions, ZipWriter};

    use super::*;

    fn build_test_zip(entry_count: usize, payload: &[u8]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);

        for i in 0..entry_count {
            writer
                .start_file(format!("file-{}.txt", i), SimpleFileOptions::default())
                .unwrap();
            writer.write_all(payload).unwrap();
        }

        writer.finish().unwrap().into_inner()
    }

    fn make_bundle(compressed_data: Vec<u8>, sha256: String) -> StreamBundle {
        StreamBundle {
            sha256,
            compressed_data,
        }
    }

    #[test]
    fn test_validate_bundle_sha256_accepts_matching_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let sha256 = format!("{:x}", Sha256::digest(&compressed_data));
        let bundle = make_bundle(compressed_data, sha256);

        validate_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn test_validate_bundle_sha256_accepts_uppercase_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let sha256 = format!("{:x}", Sha256::digest(&compressed_data)).to_uppercase();
        let bundle = make_bundle(compressed_data, sha256);

        validate_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn test_validate_bundle_sha256_rejects_mismatch() {
        let bundle = make_bundle(b"test-bundle".to_vec(), "deadbeef".to_string());

        let err = validate_bundle_sha256(&bundle).unwrap_err();
        assert!(err.contains("fingerprint mismatch"));
    }

    #[test]
    fn test_validate_bundle_sha256_allows_missing_fingerprint() {
        let bundle = make_bundle(b"test-bundle".to_vec(), String::new());
        validate_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn test_validate_zip_archive_accepts_valid_zip() {
        let zip_bytes = build_test_zip(1, b"hello");
        let archive = validate_zip_archive(Cursor::new(zip_bytes)).unwrap();
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn test_validate_zip_archive_rejects_invalid_zip_data() {
        let err = validate_zip_archive(Cursor::new(vec![1, 2, 3, 4])).unwrap_err();
        assert!(err.contains("Failed to open zip archive"));
    }

    #[test]
    fn test_validate_zip_archive_rejects_too_many_entries() {
        let zip_bytes = build_test_zip(MAX_ZIP_ENTRIES + 1, b"x");
        let err = validate_zip_archive(Cursor::new(zip_bytes)).unwrap_err();
        assert!(err.contains("too many entries"));
    }

    #[test]
    fn test_validate_zip_archive_rejects_unsafe_paths() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        writer
            .start_file("../evil.txt", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"x").unwrap();
        let zip_bytes = writer.finish().unwrap().into_inner();

        let err = validate_zip_archive(Cursor::new(zip_bytes)).unwrap_err();
        assert!(err.contains("unsafe path"));
    }

    #[test]
    fn test_checked_add_uncompressed_size_rejects_u64_overflow() {
        let err = checked_add_uncompressed_size(u64::MAX, 1).unwrap_err();
        assert!(err.contains("overflowed u64"));
    }

    #[test]
    fn test_extract_bundle_to_tempdir_extracts_files() {
        let zip_bytes = build_test_zip(1, b"hello");
        let sha256 = format!("{:x}", Sha256::digest(&zip_bytes));
        let bundle = make_bundle(zip_bytes, sha256);

        let extracted = extract_bundle_to_tempdir(&bundle).unwrap();
        let file_path = extracted.path().join("file-0.txt");
        let content = fs::read(file_path).unwrap();
        assert_eq!(content, b"hello");
        extracted.cleanup().unwrap();
    }
}
