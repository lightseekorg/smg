use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{Function, Tool, ToolChoice, ToolChoiceValue},
};
use reqwest::Url;
use serde_json::json;
use serial_test::serial;

use super::{
    request_map::map_chat_request, response_map::map_non_stream_response, signing::AwsSigner,
};

#[test]
fn maps_chat_request_to_bedrock_shape() {
    let req = ChatCompletionRequest {
        model: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
        messages: vec![
            ChatMessage::System {
                content: MessageContent::Text("You are concise".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            },
        ],
        ..Default::default()
    };

    let mapped = map_chat_request(&req);
    assert_eq!(mapped.system.len(), 1);
    assert_eq!(mapped.messages.len(), 1);
    assert_eq!(mapped.messages[0].role, "user");
    assert!(mapped.tool_config.is_none());
}

#[test]
fn maps_tools_to_bedrock_tool_config() {
    let req = ChatCompletionRequest {
        model: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("What is the weather?".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
        ..Default::default()
    };

    let mapped = map_chat_request(&req);
    let tc = mapped.tool_config.expect("tool_config should be present");
    assert_eq!(tc.tools.len(), 1);
    assert_eq!(tc.tools[0].tool_spec.name, "get_weather");
    assert_eq!(
        tc.tools[0].tool_spec.description.as_deref(),
        Some("Get weather for a location")
    );

    let serialized = serde_json::to_value(&tc.tool_choice).unwrap();
    assert_eq!(serialized, json!({"auto": {}}));
}

#[test]
fn maps_tool_choice_required_to_bedrock_any() {
    let req = ChatCompletionRequest {
        model: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("Call a tool".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "my_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
        ..Default::default()
    };

    let mapped = map_chat_request(&req);
    let tc = mapped.tool_config.unwrap();
    let serialized = serde_json::to_value(&tc.tool_choice).unwrap();
    assert_eq!(serialized, json!({"any": {}}));
}

#[test]
fn maps_converse_response_to_openai() {
    let raw = br#"{
      "output": {"message": {"content": [{"text": "Hi there"}]}},
      "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
      "stopReason": "end_turn"
    }"#;
    let mapped =
        map_non_stream_response(raw, "us.anthropic.claude-opus-4-5-20251101-v1:0").expect("maps");
    assert_eq!(
        mapped["choices"][0]["message"]["content"].as_str(),
        Some("Hi there")
    );
    assert_eq!(mapped["usage"]["total_tokens"].as_u64(), Some(15));
}

#[tokio::test]
#[serial]
async fn signer_builds_authorization_header_with_env_credentials() {
    struct EnvRestore {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }
    impl EnvRestore {
        fn new(keys: &[&'static str]) -> Self {
            let saved = keys
                .iter()
                .map(|k| (*k, std::env::var_os(k)))
                .collect::<Vec<_>>();
            Self { saved }
        }
    }
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    let _guard = EnvRestore::new(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_EC2_METADATA_DISABLED",
    ]);
    std::env::set_var("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret-example");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");

    let signer = AwsSigner::new("us-east-1".to_string(), "bedrock".to_string());
    let url = Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse")
        .expect("url");
    let signed = signer
        .sign("POST", &url, br#"{"a":1}"#)
        .await
        .expect("signed");

    assert!(signed
        .authorization
        .starts_with("AWS4-HMAC-SHA256 Credential="));
    assert!(signed.authorization.contains("SignedHeaders="));
    assert!(!signed.amz_date.is_empty());
    assert_eq!(signed.payload_hash.len(), 64);
}
