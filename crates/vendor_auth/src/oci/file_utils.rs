// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/file_utils.rs.
// Note: this file is NOT in the design-doc §5 13-file list (design doc tags it
// as "RP v2 dep only"). It is required as a transitive compile dependency of
// private_key_supplier.rs (which IS in the v1 list) — without file_utils, the
// `expand_user_home` import in private_key_supplier.rs fails to resolve.

use dirs;
use std::path::PathBuf;

/// Generates the absolute file path of the path provided based on the underlying platform
///
/// # Arguments
///
/// * `file_path`: The file path that needs to be corrected
///
/// # Returns
///
/// The absolute file path to be used based on the platform being used
///
pub fn expand_user_home(file_path: &str) -> String {
    if file_path.starts_with("~/") || file_path.starts_with("~\\") {
        if let Some(home_dir) = dirs::home_dir() {
            let full_path = home_dir.join(PathBuf::from(correct_path(&file_path[2..])));
            String::from(format!("{}", full_path.display()))
        } else {
            panic!("Error reading home directory");
        }
    } else {
        String::from(file_path)
    }
}

/// Generates the correct file path based on the platform
///
/// # Arguments
///
/// * `file_path`: The file path that needs to be corrected
///
/// # Returns
///
/// The correct the file path to be used based on the platform being used
///
fn correct_path(file_path: &str) -> String {
    if cfg!(target_os = "windows") {
        let file_path_corrected = file_path.replace("/", "\\");
        String::from(file_path_corrected)
    } else {
        String::from(file_path)
    }
}

/// Checks if provided string is absoulte path
/// # Arguments
///
/// * `file_path`: The file path that needs to be checked
///
/// # Returns
///
/// True if the the path is absolute otherwise false
///
pub fn is_absolute_path(file_path: &str) -> bool {
    let expanded_path = expand_user_home(file_path);
    std::path::Path::new(&expanded_path).is_absolute()
}
