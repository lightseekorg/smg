use std::{
    collections::{BTreeSet, HashMap},
    error::Error,
    io,
};

use openai_protocol::responses::{
    McpTool, RequireApproval, RequireApprovalMode, ResponseStatus, ResponseTool, ResponsesResponse,
};
use serde_json::Value;

fn object_keys(value: &Value) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let object = value
        .as_object()
        .ok_or_else(|| io::Error::other("fixture/response is object"))?;
    Ok(object.keys().cloned().collect())
}

fn load_fixture(json: &str) -> serde_json::Result<Value> {
    serde_json::from_str(json)
}

fn response_value(response: ResponsesResponse) -> serde_json::Result<Value> {
    serde_json::to_value(response)
}

#[test]
fn self_hosted_response_keeps_openai_field_set_except_billing() -> Result<(), Box<dyn Error>> {
    let fixture = load_fixture(include_str!("fixtures/openai_responses/A_min.json"))?;
    let response = ResponsesResponse::builder("resp_test", "gpt-5-mini-2025-08-07")
        .status(ResponseStatus::Completed)
        .completed_at(1777482018)
        .build();
    let response = response_value(response)?;

    let mut expected = object_keys(&fixture)?;
    expected.remove("billing");
    assert_eq!(object_keys(&response)?, expected);
    assert_eq!(response["tools"], Value::Array(Vec::new()));
    assert!(response.get("conversation").is_none());
    assert!(response.get("billing").is_none());
    assert!(response["completed_at"].as_i64().is_some_and(|ts| ts > 0));
    Ok(())
}

#[test]
fn conversation_echo_uses_canonical_object_shape() -> Result<(), Box<dyn Error>> {
    let fixture = load_fixture(include_str!("fixtures/openai_responses/G_with_conv.json"))?;
    let response = ResponsesResponse::builder("resp_test", "gpt-5-mini-2025-08-07")
        .status(ResponseStatus::Completed)
        .completed_at(1777482568)
        .conversation("conv_fixture")
        .build();
    let response = response_value(response)?;

    assert_eq!(fixture["conversation"], response["conversation"]);
    Ok(())
}

#[test]
fn mcp_tool_echo_scrubs_secrets_and_preserves_null_fields() -> Result<(), Box<dyn Error>> {
    let response = ResponsesResponse::builder("resp_test", "model")
        .status(ResponseStatus::Completed)
        .completed_at(1)
        .tools(vec![ResponseTool::Mcp(McpTool {
            server_url: Some("https://mcp.example.test/sse".to_string()),
            authorization: Some("secret-token".to_string()),
            headers: Some(HashMap::from([(
                "Authorization".to_string(),
                "Bearer secret".to_string(),
            )])),
            server_label: "deepwiki".to_string(),
            server_description: None,
            require_approval: Some(RequireApproval::Mode(RequireApprovalMode::Never)),
            allowed_tools: None,
            connector_id: None,
            defer_loading: None,
        })])
        .build();
    let response = response_value(response)?;
    let tool = &response["tools"][0];

    assert_eq!(tool["type"], "mcp");
    assert_eq!(tool["server_label"], "deepwiki");
    assert_eq!(tool["server_description"], Value::Null);
    assert_eq!(tool["allowed_tools"], Value::Null);
    assert_eq!(tool["headers"], Value::Null);
    assert!(tool.get("authorization").is_none());
    Ok(())
}
