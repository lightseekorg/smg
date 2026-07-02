//! Safe wrapper for OpenCV's buffered video capture constructor.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_void, CStr};

use bytes::Bytes;
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

pub(crate) struct BufferedCapture {
    capture: videoio::VideoCapture,
    _bytes: Bytes,
}

impl BufferedCapture {
    pub(crate) fn capture_mut(&mut self) -> &mut videoio::VideoCapture {
        &mut self.capture
    }
}

pub(crate) fn open_capture(bytes: Bytes, decoder_threads: i32) -> Result<BufferedCapture, String> {
    let mut error = [0 as c_char; 512];
    // SAFETY: `BufferedCapture` owns `bytes` for at least as long as the capture.
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

    Ok(BufferedCapture {
        // SAFETY: the bridge returns a heap-allocated cv::VideoCapture compatible
        // with the opencv crate's generated ownership wrapper.
        capture: unsafe { videoio::VideoCapture::opencv_from_extern(capture) },
        _bytes: bytes,
    })
}
