fn main() {
    // Link libjpeg-turbo's TurboJPEG API so JPEG decode matches PIL/libjpeg-turbo
    // (what vLLM uses) bit-for-bit. Provided by the `libturbojpeg0-dev` package.
    println!("cargo:rustc-link-lib=turbojpeg");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
}
