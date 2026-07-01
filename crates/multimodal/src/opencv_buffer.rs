//! Safe wrapper for OpenCV's buffered video capture constructor.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_void, CStr};

use opencv::{traits::OpenCVFromExtern, videoio};

unsafe extern "C" {
    fn smg_opencv_capture_from_buffer(
        data: *const u8,
        size: usize,
        decoder_threads: i32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
}

pub(crate) fn open_capture(
    bytes: &[u8],
    decoder_threads: i32,
) -> Result<videoio::VideoCapture, String> {
    let mut error = [0 as c_char; 512];
    // SAFETY: the bridge only reads the buffer while the returned capture is
    // alive. The caller retains `bytes` until it drops the capture after decode.
    let capture = unsafe {
        smg_opencv_capture_from_buffer(
            bytes.as_ptr(),
            bytes.len(),
            decoder_threads,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if capture.is_null() {
        // SAFETY: the bridge always writes a NUL-terminated message on failure.
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }

    // SAFETY: the bridge returns a heap-allocated cv::VideoCapture compatible
    // with the opencv crate's generated ownership wrapper.
    Ok(unsafe { videoio::VideoCapture::opencv_from_extern(capture) })
}
