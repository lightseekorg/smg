use std::io::{Read, Seek};

use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::bundle::{TokenizerBundle, MAX_UNCOMPRESSED_SIZE, MAX_ZIP_ENTRIES};

/// Validate the SHA-256 fingerprint of a tokenizer bundle
pub fn validate_tokenizer_bundle_sha256(bundle: &TokenizerBundle) -> Result<(), String> {
    if bundle.metadata.fingerprint.is_empty() {
        return Ok(());
    }

    let computed = format!("{:x}", Sha256::digest(&bundle.compressed_data));
    if !computed.eq_ignore_ascii_case(&bundle.metadata.fingerprint) {
        return Err(format!(
            "Tokenizer bundle fingerprint mismatch: expected {}, got {}",
            bundle.metadata.fingerprint, computed
        ));
    }
    Ok(())
}

/// Validate and open a zip archive from bytes
///
/// checks:
/// - Maximum number of entries
/// - Maximum total uncompressed size
pub fn validate_zip_archive<R: Read + Seek>(reader: R) -> Result<ZipArchive<R>, String> {
    let mut archive =
        ZipArchive::new(reader).map_err(|e| format!("Failed to open zip archive: {}", e))?;

    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(format!(
            "Tokenizer zip archive has too many entries ({} > {})",
            archive.len(),
            MAX_ZIP_ENTRIES
        ));
    }

    let total_uncompressed: u64 = (0..archive.len())
        .map(|i| archive.by_index_raw(i).map(|f| f.size()).unwrap_or(0))
        .sum();

    if total_uncompressed > MAX_UNCOMPRESSED_SIZE {
        return Err(format!(
            "Tokenizer zip archive uncompressed size too large ({} bytes > {} bytes)",
            total_uncompressed, MAX_UNCOMPRESSED_SIZE
        ));
    }

    Ok(archive)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use sha2::{Digest, Sha256};
    use zip::{write::SimpleFileOptions, ZipWriter};

    use super::*;
    use crate::bundle::TokenizerMetadata;

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

    fn make_bundle(compressed_data: Vec<u8>, fingerprint: String) -> TokenizerBundle {
        TokenizerBundle {
            metadata: TokenizerMetadata {
                model_identifier: "test-model".to_string(),
                fingerprint,
                files: vec![],
                bundle_format: "zip".to_string(),
            },
            compressed_data,
        }
    }

    #[test]
    fn test_validate_tokenizer_bundle_sha256_accepts_matching_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let fingerprint = format!("{:x}", Sha256::digest(&compressed_data));
        let bundle = make_bundle(compressed_data, fingerprint);

        validate_tokenizer_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn test_validate_tokenizer_bundle_sha256_accepts_uppercase_fingerprint() {
        let compressed_data = b"test-bundle".to_vec();
        let fingerprint = format!("{:x}", Sha256::digest(&compressed_data)).to_uppercase();
        let bundle = make_bundle(compressed_data, fingerprint);

        validate_tokenizer_bundle_sha256(&bundle).unwrap();
    }

    #[test]
    fn test_validate_tokenizer_bundle_sha256_rejects_mismatch() {
        let bundle = make_bundle(b"test-bundle".to_vec(), "deadbeef".to_string());

        let err = validate_tokenizer_bundle_sha256(&bundle).unwrap_err();
        assert!(err.contains("fingerprint mismatch"));
    }

    #[test]
    fn test_validate_tokenizer_bundle_sha256_allows_missing_fingerprint() {
        let bundle = make_bundle(b"test-bundle".to_vec(), String::new());
        validate_tokenizer_bundle_sha256(&bundle).unwrap();
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
}
