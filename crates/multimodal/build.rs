fn main() {
    // Link libjpeg-turbo's TurboJPEG API so JPEG decode matches PIL/libjpeg-turbo
    // (what vLLM uses) bit-for-bit. Provided by the `libturbojpeg0-dev` package.
    println!("cargo:rustc-link-lib=turbojpeg");

    // On Debian/Ubuntu multiarch layouts libturbojpeg lives under
    // /usr/lib/<triplet> rather than a default linker search dir. Derive the
    // triplet from the build target (not hardcoded to x86_64) and only add the
    // path when it actually exists. Other platforms (macOS/Homebrew, Windows,
    // non-multiarch distros) resolve turbojpeg via the default linker search
    // path, so we add nothing there and avoid breaking the build.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "linux" {
        let triplet = match arch.as_str() {
            "x86_64" => Some("x86_64-linux-gnu"),
            "aarch64" => Some("aarch64-linux-gnu"),
            _ => None,
        };
        if let Some(triplet) = triplet {
            let dir = format!("/usr/lib/{triplet}");
            if std::path::Path::new(&dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
    }
}
