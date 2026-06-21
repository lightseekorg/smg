//! Minimal FFI to libjpeg-turbo's TurboJPEG API for JPEG decode.
//!
//! PIL/Pillow (and therefore vLLM) decode JPEGs with libjpeg-turbo using its
//! default options: accurate (islow) integer IDCT and "fancy" (bilinear) chroma
//! upsampling. The pure-Rust `image`/`zune-jpeg` decoder differs by a few levels
//! per pixel, which the vision encoder amplifies into a large embedding shift,
//! making TokenSpeed's multimodal accuracy diverge from vLLM. Decoding through
//! libjpeg-turbo with the same defaults makes SMG's pixel values match vLLM's.
//!
//! We bind only the three functions needed for RGB decode. Default flags (0)
//! select accurate DCT + fancy upsampling, matching Pillow.
//!
//! This module is the crate's only FFI surface, so it locally overrides the
//! workspace-wide `unsafe_code = "deny"` for the C bindings.
#![allow(unsafe_code)]

use std::os::raw::{c_int, c_uchar, c_ulong, c_void};

use image::{DynamicImage, RgbImage};

type TjHandle = *mut c_void;
const TJPF_RGB: c_int = 0;

#[link(name = "turbojpeg")]
extern "C" {
    fn tjInitDecompress() -> TjHandle;
    fn tjDecompressHeader3(
        handle: TjHandle,
        jpeg_buf: *const c_uchar,
        jpeg_size: c_ulong,
        width: *mut c_int,
        height: *mut c_int,
        jpeg_subsamp: *mut c_int,
        jpeg_colorspace: *mut c_int,
    ) -> c_int;
    fn tjDecompress2(
        handle: TjHandle,
        jpeg_buf: *const c_uchar,
        jpeg_size: c_ulong,
        dst_buf: *mut c_uchar,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        flags: c_int,
    ) -> c_int;
    fn tjDestroy(handle: TjHandle) -> c_int;
}

/// True if `bytes` start with the JPEG SOI marker.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// Decode a JPEG to an RGB8 `DynamicImage` via libjpeg-turbo (PIL-compatible
/// defaults). Returns `None` on any failure so the caller can fall back to the
/// pure-Rust decoder.
pub fn decode_jpeg_rgb(bytes: &[u8]) -> Option<DynamicImage> {
    if !is_jpeg(bytes) {
        return None;
    }
    // SAFETY: handle is checked for null; buffers are sized from the decoded
    // header; the handle is always destroyed before returning.
    unsafe {
        let handle = tjInitDecompress();
        if handle.is_null() {
            return None;
        }
        let (mut w, mut h, mut subsamp, mut colorspace) = (0_i32, 0_i32, 0_i32, 0_i32);
        let hdr = tjDecompressHeader3(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            &mut w,
            &mut h,
            &mut subsamp,
            &mut colorspace,
        );
        if hdr != 0 || w <= 0 || h <= 0 {
            tjDestroy(handle);
            return None;
        }
        let (wu, hu) = (w as usize, h as usize);
        // Guard against absurd dimensions before allocating.
        let pixels = wu.checked_mul(hu).and_then(|p| p.checked_mul(3));
        let nbytes = match pixels {
            Some(n) => n,
            None => {
                tjDestroy(handle);
                return None;
            }
        };
        let mut buf = vec![0_u8; nbytes];
        let rc = tjDecompress2(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            buf.as_mut_ptr(),
            w,
            0, // pitch = 0 -> width * pixelsize
            h,
            TJPF_RGB,
            0, // default flags: accurate IDCT + fancy upsampling (matches Pillow)
        );
        tjDestroy(handle);
        if rc != 0 {
            return None;
        }
        RgbImage::from_raw(w as u32, h as u32, buf).map(DynamicImage::ImageRgb8)
    }
}
