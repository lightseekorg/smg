//! Load balancing policy FFI bindings for Go SDK
//!
//! This module provides FFI functions to create and use load balancing policies
//! from the model_gateway crate. It enables the Go SDK to distribute requests
//! across multiple gRPC workers using the same policy implementations as the
//! Rust gateway.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

use smg::{
    grpc_client::sglang_scheduler::SglangSchedulerClient,
    policies::{
        CacheAwarePolicy, LoadBalancingPolicy, PolicyFactory, RandomPolicy, RoundRobinPolicy,
    },
    protocols::chat::ChatCompletionRequest,
    routers::grpc::utils::{generate_tool_constraints, process_chat_messages},
    tokenizer::{create_tokenizer_from_file, traits::Tokenizer},
};
use uuid::Uuid;

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::sgl_grpc_response_converter_create,
    runtime::RUNTIME,
    stream::SglangStreamHandle,
    tokenizer::TokenizerHandle,
};

/// Simplified worker for Go SDK - wraps a gRPC client with health tracking
pub struct GrpcWorker {
    pub(crate) client: Arc<SglangSchedulerClient>,
    pub(crate) endpoint: String,
    pub(crate) healthy: AtomicBool,
}

impl GrpcWorker {
    pub fn new(client: Arc<SglangSchedulerClient>, endpoint: String) -> Self {
        Self {
            client,
            endpoint,
            healthy: AtomicBool::new(true),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }
}

/// Adapter to make GrpcWorker compatible with LoadBalancingPolicy
/// The gateway's policies expect Arc<dyn Worker>, but we have a simpler GrpcWorker.
/// We implement a minimal Worker-like interface here.
impl std::fmt::Debug for GrpcWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcWorker")
            .field("endpoint", &self.endpoint)
            .field("healthy", &self.is_healthy())
            .finish()
    }
}

/// Handle for a multi-worker client with load balancing
pub struct MultiWorkerClientHandle {
    pub(crate) workers: Vec<Arc<GrpcWorker>>,
    pub(crate) policy: Arc<dyn LoadBalancingPolicy>,
    pub(crate) tokenizer_path: String,
    pub(crate) counter: AtomicUsize, // For round-robin fallback
}

impl MultiWorkerClientHandle {
    /// Select a worker using the configured policy
    /// Falls back to round-robin if policy selection fails
    pub fn select_worker(&self) -> Option<Arc<GrpcWorker>> {
        let healthy_workers: Vec<(usize, &Arc<GrpcWorker>)> = self
            .workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .collect();

        if healthy_workers.is_empty() {
            return None;
        }

        // Use round-robin for now since the policy's select_worker expects Arc<dyn Worker>
        // TODO: Implement proper policy integration when Worker trait is exposed via FFI
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let idx = count % healthy_workers.len();
        Some(Arc::clone(healthy_workers[idx].1))
    }
}

/// Create a multi-worker client with load balancing
///
/// # Arguments
/// * `endpoints` - Comma-separated list of gRPC endpoints (e.g., "grpc://host1:20000,grpc://host2:20001")
/// * `tokenizer_path` - Path to tokenizer directory
/// * `policy_name` - Load balancing policy name ("round_robin", "random", "cache_aware")
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to MultiWorkerClientHandle on success, null on failure
///
/// # Safety
/// - All string arguments must be valid null-terminated C strings
/// - Caller owns the returned handle and must free it with `sgl_multi_client_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_create(
    endpoints: *const c_char,
    tokenizer_path: *const c_char,
    policy_name: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut MultiWorkerClientHandle {
    if endpoints.is_null() || tokenizer_path.is_null() || policy_name.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let endpoints_str = match CStr::from_ptr(endpoints).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in endpoints");
            return ptr::null_mut();
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return ptr::null_mut();
        }
    };

    let policy_name_str = match CStr::from_ptr(policy_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in policy_name");
            return ptr::null_mut();
        }
    };

    // Parse endpoints
    let endpoint_list: Vec<&str> = endpoints_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if endpoint_list.is_empty() {
        set_error_message(error_out, "No valid endpoints provided");
        return ptr::null_mut();
    }

    // Create policy
    let policy: Arc<dyn LoadBalancingPolicy> = match policy_name_str {
        "round_robin" => Arc::new(RoundRobinPolicy::new()),
        "random" => Arc::new(RandomPolicy::new()),
        "cache_aware" => Arc::new(CacheAwarePolicy::new()),
        _ => {
            // Try factory
            match PolicyFactory::create_by_name(policy_name_str) {
                Some(p) => p,
                None => {
                    set_error_message(
                        error_out,
                        &format!(
                            "Unknown policy: {}. Available: round_robin, random, cache_aware",
                            policy_name_str
                        ),
                    );
                    return ptr::null_mut();
                }
            }
        }
    };

    // Create gRPC clients for all endpoints
    let mut workers = Vec::with_capacity(endpoint_list.len());
    for endpoint in endpoint_list {
        let client =
            match RUNTIME.block_on(async { SglangSchedulerClient::connect(endpoint).await }) {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    set_error_message(
                        error_out,
                        &format!("Failed to connect to {}: {}", endpoint, e),
                    );
                    return ptr::null_mut();
                }
            };
        workers.push(Arc::new(GrpcWorker::new(client, endpoint.to_string())));
    }

    Box::into_raw(Box::new(MultiWorkerClientHandle {
        workers,
        policy,
        tokenizer_path: tokenizer_path_str,
        counter: AtomicUsize::new(0),
    }))
}

/// Free a multi-worker client handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`, or null
/// - `handle` must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_free(handle: *mut MultiWorkerClientHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Get the number of workers in the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_worker_count(
    handle: *mut MultiWorkerClientHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).workers.len()
}

/// Get the number of healthy workers in the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_healthy_count(
    handle: *mut MultiWorkerClientHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).workers.iter().filter(|w| w.is_healthy()).count()
}

/// Mark a worker as unhealthy by index
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_set_worker_health(
    handle: *mut MultiWorkerClientHandle,
    worker_index: usize,
    healthy: bool,
) -> SglErrorCode {
    if handle.is_null() {
        return SglErrorCode::InvalidArgument;
    }
    let client = &*handle;
    if worker_index >= client.workers.len() {
        return SglErrorCode::InvalidArgument;
    }
    client.workers[worker_index].set_healthy(healthy);
    SglErrorCode::Success
}

/// Get the policy name
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - Returned string must be freed with `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_policy_name(
    handle: *mut MultiWorkerClientHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let policy_name = (*handle).policy.name();
    match CString::new(policy_name) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Get the tokenizer path from the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - Returned string must be freed with `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_tokenizer_path(
    handle: *mut MultiWorkerClientHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    match CString::new((*handle).tokenizer_path.as_str()) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Send a chat completion request using load-balanced worker selection
///
/// # Arguments
/// * `client_handle` - Multi-worker client handle
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `stream_handle_out` - Pointer to receive stream handle
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `client_handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - `request_json` must be a valid null-terminated C string containing valid JSON
/// - `stream_handle_out` must be a valid pointer to writable memory
/// - Caller owns the stream handle and must free it with `sgl_stream_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_chat_completion_stream(
    client_handle: *mut MultiWorkerClientHandle,
    request_json: *const c_char,
    stream_handle_out: *mut *mut SglangStreamHandle,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if client_handle.is_null() || request_json.is_null() || stream_handle_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let multi_client = &*client_handle;

    // Select a worker using the policy
    let worker = match multi_client.select_worker() {
        Some(w) => w,
        None => {
            set_error_message(error_out, "No healthy workers available");
            return SglErrorCode::UnknownError;
        }
    };

    let client = Arc::clone(&worker.client);

    // Create tokenizer
    let tokenizer: Arc<dyn Tokenizer> =
        match create_tokenizer_from_file(&multi_client.tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                set_error_message(error_out, &format!("Failed to create tokenizer: {}", e));
                return SglErrorCode::TokenizationError;
            }
        };

    // Parse OpenAI ChatCompletionRequest
    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Process messages and apply chat template
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref()) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to process messages: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };

    // Tokenize
    let token_ids = match tokenizer.encode(&processed_messages.text, false) {
        Ok(encoding) => encoding.token_ids().to_vec(),
        Err(e) => {
            set_error_message(error_out, &format!("Failed to tokenize: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };
    let prompt_tokens = token_ids.len() as i32;

    // Generate tool constraints if needed
    let tool_constraint = if let Some(tools) = chat_request.tools.as_ref() {
        match generate_tool_constraints(tools, &chat_request.tool_choice, &chat_request.model) {
            Ok(Some((constraint_type, constraint_value))) => {
                Some((constraint_type, constraint_value))
            }
            Ok(None) => None,
            Err(e) => {
                set_error_message(
                    error_out,
                    &format!("Failed to generate tool constraints: {}", e),
                );
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Build GenerateRequest
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let proto_request = match client.build_generate_request_from_chat(
        request_id.clone(),
        &chat_request,
        processed_messages.text,
        token_ids,
        processed_messages.multimodal_inputs,
        tool_constraint,
    ) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to build generate request: {}", e),
            );
            return SglErrorCode::ParsingError;
        }
    };

    // Send request and get stream
    let stream = match RUNTIME.block_on(async { client.generate(proto_request).await }) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to send request: {}", e));
            return SglErrorCode::UnknownError;
        }
    };

    // Create response converter
    let tools_json = chat_request
        .tools
        .as_ref()
        .and_then(|t| serde_json::to_string(t).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let tool_choice_json = chat_request
        .tool_choice
        .as_ref()
        .and_then(|tc| serde_json::to_string(tc).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_json = chat_request
        .stop
        .as_ref()
        .and_then(|s| serde_json::to_string(s).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_token_ids_json = chat_request
        .stop_token_ids
        .as_ref()
        .and_then(|ids| serde_json::to_string(ids).ok())
        .map(|s| CString::new(s).unwrap().into_raw());

    // Create tokenizer handle for converter
    let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
        tokenizer: Arc::clone(&tokenizer),
    }));

    let converter = sgl_grpc_response_converter_create(
        tokenizer_handle,
        CString::new(chat_request.model.clone()).unwrap().as_ptr(),
        CString::new(request_id.clone()).unwrap().as_ptr(),
        tools_json.unwrap_or(ptr::null_mut()),
        tool_choice_json.unwrap_or(ptr::null_mut()),
        stop_json.unwrap_or(ptr::null_mut()),
        stop_token_ids_json.unwrap_or(ptr::null_mut()),
        if chat_request.skip_special_tokens {
            1
        } else {
            0
        },
        error_out,
    );

    // Free temporary tokenizer handle (converter now owns the tokenizer)
    let _ = Box::from_raw(tokenizer_handle);

    if converter.is_null() {
        return SglErrorCode::MemoryError;
    }

    // Clean up temporary CStrings
    if let Some(ptr) = tools_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = tool_choice_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_token_ids_json {
        let _ = CString::from_raw(ptr);
    }

    // Create converter handle and set initial_prompt_tokens
    let mut converter_handle = *Box::from_raw(converter);
    converter_handle.initial_prompt_tokens = Some(prompt_tokens);

    // Create stream handle
    *stream_handle_out = Box::into_raw(Box::new(SglangStreamHandle {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        converter: Arc::new(tokio::sync::Mutex::new(converter_handle)),
        client: Arc::clone(&client),
        prompt_tokens,
    }));

    SglErrorCode::Success
}
