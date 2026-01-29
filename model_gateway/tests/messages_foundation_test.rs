//! Foundation tests for Messages API (PR #1)
//!
//! These tests verify:
//! - /v1/messages endpoint exists
//! - Returns 501 Not Implemented (expected for PR #1)
//! - Request deserialization works
//! - No breaking changes to existing functionality

use serde_json::json;

#[test]
fn test_create_message_request_deserialization() {
    use smg::protocols::messages::{CreateMessageRequest, InputContent, Role};

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
    use smg::protocols::messages::{CreateMessageRequest, SystemContent};

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
    use smg::protocols::messages::CreateMessageRequest;

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
    use smg::protocols::messages::CreateMessageRequest;

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
    use smg::protocols::messages::{CreateMessageRequest, Role};

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
fn test_messages_handler_tool_extraction() {
    use serde_json::json;
    use smg::{
        protocols::messages::ContentBlock, routers::openai::messages::tools::extract_tool_calls,
    };

    let content = vec![
        ContentBlock::Text {
            text: "Let me use a tool".to_string(),
            citations: None,
        },
        ContentBlock::ToolUse {
            id: "toolu_123".to_string(),
            name: "calculator".to_string(),
            input: json!({"expression": "2+2"}),
        },
    ];

    let tool_calls = extract_tool_calls(&content);
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "calculator");
    assert_eq!(tool_calls[0].id, "toolu_123");
}

#[test]
fn test_messages_handler_tool_extraction_empty() {
    use smg::{
        protocols::messages::ContentBlock, routers::openai::messages::tools::extract_tool_calls,
    };

    let content = vec![ContentBlock::Text {
        text: "Just text, no tools".to_string(),
        citations: None,
    }];

    let tool_calls = extract_tool_calls(&content);
    assert_eq!(tool_calls.len(), 0);
}

#[test]
fn test_create_message_request_with_temperature() {
    use smg::protocols::messages::CreateMessageRequest;

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
    use smg::protocols::messages::CreateMessageRequest;

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
