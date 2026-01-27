//! Error handling for FFI functions

use std::{ffi::CString, os::raw::c_char, ptr};

/// Error codes returned by FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglErrorCode {
    Success = 0,
    InvalidArgument = 1,
    TokenizationError = 2,
    ParsingError = 3,
    MemoryError = 4,
    UnknownError = 99,
}

/// Helper to set error message in FFI output parameter
///
/// # Safety
/// - `error_out` may be null; if non-null, must point to valid writable memory
/// - Caller must free any previous string at `*error_out` before calling
pub unsafe fn set_error_message(error_out: *mut *mut c_char, message: &str) {
    if !error_out.is_null() {
        if let Ok(cstr) = CString::new(message) {
            *error_out = cstr.into_raw();
        } else {
            *error_out = ptr::null_mut();
        }
    }
}

/// Helper to set error message from format string
///
/// # Safety
/// - `error_out` may be null; if non-null, must point to valid writable memory
pub unsafe fn set_error_message_fmt(error_out: *mut *mut c_char, fmt: std::fmt::Arguments) {
    if !error_out.is_null() {
        let msg = format!("{}", fmt);
        set_error_message(error_out, &msg);
    }
}

/// Helper to clear error message
///
/// # Safety
/// - `error_out` may be null; if non-null, must point to valid writable memory
pub unsafe fn clear_error_message(error_out: *mut *mut c_char) {
    if !error_out.is_null() {
        *error_out = ptr::null_mut();
    }
}

// Helper functions for error handling
// Note: Some helper functions are kept for potential future use
