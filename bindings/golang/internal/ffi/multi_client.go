// Package ffi provides Go bindings for SMG's Rust FFI (Foreign Function Interface).
//
// This file provides multi-worker client FFI bindings with load balancing support.
package ffi

/*
#cgo LDFLAGS: -lsmg_go -ldl
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// Error codes
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Opaque handles
typedef void* MultiWorkerClientHandle;
typedef void* SglangStreamHandle;

// Multi-worker client functions
MultiWorkerClientHandle* sgl_multi_client_create(const char* endpoints, const char* tokenizer_path, const char* policy_name, char** error_out);
void sgl_multi_client_free(MultiWorkerClientHandle* handle);
size_t sgl_multi_client_worker_count(MultiWorkerClientHandle* handle);
size_t sgl_multi_client_healthy_count(MultiWorkerClientHandle* handle);
SglErrorCode sgl_multi_client_set_worker_health(MultiWorkerClientHandle* handle, size_t worker_index, bool healthy);
char* sgl_multi_client_policy_name(MultiWorkerClientHandle* handle);
char* sgl_multi_client_tokenizer_path(MultiWorkerClientHandle* handle);
SglErrorCode sgl_multi_client_chat_completion_stream(MultiWorkerClientHandle* client_handle, const char* request_json, SglangStreamHandle** stream_handle_out, char** error_out);

// Stream and memory functions (already declared in client.go, but needed for this file)
SglErrorCode sgl_stream_read_next(SglangStreamHandle* stream_handle, char** response_json_out, int* is_done_out, char** error_out);
void sgl_stream_free(SglangStreamHandle* handle);
void sgl_free_string(char* s);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MultiWorkerClientHandle wraps the Rust multi-worker client FFI handle.
//
// This struct maintains connections to multiple SMG gRPC servers and uses
// a load balancing policy to distribute requests across workers.
type MultiWorkerClientHandle struct {
	handle *C.MultiWorkerClientHandle
}

// NewMultiWorkerClient creates a new multi-worker client with load balancing.
//
// Parameters:
// - endpoints: Comma-separated list of gRPC endpoints (e.g., "grpc://host1:20000,grpc://host2:20001")
// - tokenizerPath: Path to tokenizer directory
// - policyName: Load balancing policy name ("round_robin", "random", "cache_aware")
//
// Returns:
// - *MultiWorkerClientHandle: A new multi-worker client handle
// - error: An error if client creation failed
func NewMultiWorkerClient(endpoints, tokenizerPath, policyName string) (*MultiWorkerClientHandle, error) {
	cEndpoints := C.CString(endpoints)
	defer C.free(unsafe.Pointer(cEndpoints))

	cTokenizerPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cTokenizerPath))

	cPolicyName := C.CString(policyName)
	defer C.free(unsafe.Pointer(cPolicyName))

	var errorPtr *C.char
	handle := C.sgl_multi_client_create(cEndpoints, cTokenizerPath, cPolicyName, &errorPtr)

	if handle == nil {
		errorMsg := ""
		if errorPtr != nil {
			errorMsg = C.GoString(errorPtr)
			C.sgl_free_string(errorPtr)
		}
		if errorMsg == "" {
			errorMsg = "failed to create multi-worker client"
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &MultiWorkerClientHandle{handle: handle}, nil
}

// Free releases the multi-worker client handle
func (h *MultiWorkerClientHandle) Free() {
	if h.handle != nil {
		C.sgl_multi_client_free(h.handle)
		h.handle = nil
	}
}

// WorkerCount returns the total number of workers
func (h *MultiWorkerClientHandle) WorkerCount() int {
	if h.handle == nil {
		return 0
	}
	return int(C.sgl_multi_client_worker_count(h.handle))
}

// HealthyCount returns the number of healthy workers
func (h *MultiWorkerClientHandle) HealthyCount() int {
	if h.handle == nil {
		return 0
	}
	return int(C.sgl_multi_client_healthy_count(h.handle))
}

// SetWorkerHealth marks a worker as healthy or unhealthy by index
func (h *MultiWorkerClientHandle) SetWorkerHealth(workerIndex int, healthy bool) error {
	if h.handle == nil {
		return fmt.Errorf("multi-worker client handle is nil")
	}
	result := C.sgl_multi_client_set_worker_health(h.handle, C.size_t(workerIndex), C.bool(healthy))
	if ErrorCode(result) != ErrorSuccess {
		return fmt.Errorf("failed to set worker health: error code %d", result)
	}
	return nil
}

// PolicyName returns the name of the load balancing policy
func (h *MultiWorkerClientHandle) PolicyName() string {
	if h.handle == nil {
		return ""
	}
	cName := C.sgl_multi_client_policy_name(h.handle)
	if cName == nil {
		return ""
	}
	defer C.sgl_free_string(cName)
	return C.GoString(cName)
}

// TokenizerPath returns the tokenizer path
func (h *MultiWorkerClientHandle) TokenizerPath() string {
	if h.handle == nil {
		return ""
	}
	cPath := C.sgl_multi_client_tokenizer_path(h.handle)
	if cPath == nil {
		return ""
	}
	defer C.sgl_free_string(cPath)
	return C.GoString(cPath)
}

// ChatCompletionStream creates a streaming chat completion request with load balancing
func (h *MultiWorkerClientHandle) ChatCompletionStream(requestJSON string) (*SglangStreamHandle, error) {
	if h.handle == nil {
		return nil, fmt.Errorf("multi-worker client handle is nil")
	}

	cRequestJSON := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(cRequestJSON))

	var streamHandle *C.SglangStreamHandle
	var errorPtr *C.char

	result := C.sgl_multi_client_chat_completion_stream(
		h.handle,
		cRequestJSON,
		&streamHandle,
		&errorPtr,
	)

	if ErrorCode(result) != ErrorSuccess {
		errorMsg := ""
		if errorPtr != nil {
			errorMsg = C.GoString(errorPtr)
			C.sgl_free_string(errorPtr)
		}
		if errorMsg == "" {
			errorMsg = fmt.Sprintf("error code %d", result)
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	if streamHandle == nil {
		return nil, fmt.Errorf("stream handle is nil")
	}

	return &SglangStreamHandle{handle: streamHandle}, nil
}
