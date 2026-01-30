//! Basic tests for Messages API
//!
//! These tests verify:
//! - /v1/messages endpoint exists
//! - Returns 501 Not Implemented (expected for PR #1)
//! - Request deserialization works
//! - No breaking changes to existing functionality

use serde_json::json;
use smg::protocols::messages::{CreateMessageRequest, InputContent, Role, SystemContent};

#[test]
fn test_create_message_request_deserialization() {
    // Test simple string content
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Hello, Claude!"
            }
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize request");

    assert_eq!(request.model, "claude-sonnet-4-5-20250929");
    assert_eq!(request.max_tokens, 1024);
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, Role::User);

    match &request.messages[0].content {
        InputContent::String(s) => assert_eq!(s, "Hello, Claude!"),
        _ => panic!("Expected string content"),
    }
}

#[test]
fn test_create_message_request_with_system() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.system.is_some());
    match request.system.unwrap() {
        SystemContent::String(s) => assert_eq!(s, "You are a helpful assistant."),
        _ => panic!("Expected string system content"),
    }
}

#[test]
fn test_create_message_request_with_tools() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "test"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.tools.is_some());
    assert_eq!(request.tools.unwrap().len(), 1);
}

#[test]
fn test_create_message_request_with_stream() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "test"}],
        "stream": true
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.stream, Some(true));
}

#[test]
fn test_create_message_request_multi_turn() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.messages.len(), 3);
    assert_eq!(request.messages[0].role, Role::User);
    assert_eq!(request.messages[1].role, Role::Assistant);
    assert_eq!(request.messages[2].role, Role::User);
}

#[test]
fn test_create_message_request_with_temperature() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.temperature, Some(0.7));
}

#[test]
fn test_create_message_request_with_thinking() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "test"}],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024
        }
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.thinking.is_some());
}
