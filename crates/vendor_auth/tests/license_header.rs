//! Wire `scripts/check_oracle_headers.sh` into `cargo test`.
//!
//! Per design doc R9: UPL-1.0 requires retention of the Oracle copyright
//! notice in every copied file. This test fails the suite if any file in
//! `src/oci/` is missing the header. It corresponds to STEP 3h of the
//! CB-1 implementer plan.

use std::process::Command;

#[test]
fn oracle_copyright_header_present_in_all_oci_files() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let script = format!("{manifest_dir}/scripts/check_oracle_headers.sh");

    let output = Command::new("bash")
        .arg(&script)
        .output()
        .expect("run check_oracle_headers.sh");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "license-header check failed.\nscript: {script}\nstdout: {stdout}\nstderr: {stderr}"
    );
}
