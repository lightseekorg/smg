//! Response transformer for MCP to OpenAI format conversion.

use openai_protocol::responses::{
    CodeInterpreterCallStatus, CodeInterpreterOutput, FileSearchCallStatus, FileSearchResult,
    ImageGenerationCallStatus, ResponseOutputItem, WebSearchAction, WebSearchCallStatus,
    WebSearchSource,
};
use serde_json::Value;

use super::ResponseFormat;

/// Normalize an MCP response item id source into an external `mcp_call.id`.
///
/// The input may be an upstream output item id (`fc_*`), an internal call id
/// (`call_*`), or an already-normalized MCP id (`mcp_*`).
pub fn mcp_response_item_id(source_id: &str) -> String {
    if source_id.starts_with("mcp_") {
        return source_id.to_string();
    }

    if let Some(stripped) = source_id
        .strip_prefix("call_")
        .or_else(|| source_id.strip_prefix("fc_"))
    {
        return format!("mcp_{stripped}");
    }

    format!("mcp_{source_id}")
}

/// Extract image-generation fallback text from JSON-RPC result content wrapper:
/// `{"result":{"content":[{"type":"text","text":"..."}]}}`.
pub fn extract_image_generation_fallback_text(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|obj| obj.get("result"))
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("content"))
        .and_then(|v| v.as_array())
        .and_then(|content| {
            content.iter().find_map(|item| {
                item.as_object()
                    .filter(|o| o.get("type").and_then(|v| v.as_str()) == Some("text"))
                    .and_then(|o| o.get("text"))
                    .and_then(|v| v.as_str())
                    .filter(|text| !text.trim().is_empty())
                    .map(str::to_string)
            })
        })
}

/// Read image-generation error status from JSON-RPC payload:
/// `result.isError` (defaults to `false` when missing).
pub fn is_image_generation_error(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|obj| obj.get("result"))
        .and_then(|v| v.as_object())
        .and_then(|result_obj| result_obj.get("isError"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Transforms MCP CallToolResult to OpenAI Responses API output items.
pub struct ResponseTransformer;

impl ResponseTransformer {
    fn is_image_payload_candidate(obj: &serde_json::Map<String, Value>) -> bool {
        obj.get("result").and_then(|v| v.as_str()).is_some()
    }

    /// Extract image payload from MCP JSON-RPC `tools/call` response.
    ///
    /// Success example (`isError=false`), where `text` contains JSON with base64 result:
    /// `{"result":{"content":[{"type":"text","text":"{\"result\":\"<base64>\"}"}],"isError":false}}`
    ///
    /// Error example (`isError=true`), where `text` is plain error text:
    /// `{"result":{"content":[{"type":"text","text":"Error executing tool generate_image: ..."}],"isError":true}}`
    ///
    /// Any other shape is treated as unexpected and handled by the caller's
    /// single fallback path.
    fn image_payload_from_wrapped_content(result: &Value) -> Option<Value> {
        // Parse the JSON-RPC wrapper object under top-level `result`.
        let result_obj = result.get("result").and_then(|v| v.as_object())?;

        // `isError` determines whether we extract raw error text or image payload JSON.
        let is_error = is_image_generation_error(result);

        // Error responses should preserve the raw text error payload.
        if is_error {
            return extract_image_generation_fallback_text(result).map(Value::String);
        }

        let content = result_obj.get("content").and_then(|v| v.as_array())?;
        for item in content {
            let Some(obj) = item.as_object() else {
                continue;
            };
            // Only parse text content entries from MCP content blocks.
            if obj.get("type").and_then(|v| v.as_str()) != Some("text") {
                continue;
            }
            let Some(text) = obj.get("text").and_then(|v| v.as_str()) else {
                continue;
            };
            // Success payload is expected as JSON text in the content block.
            if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                // Accept only image payload objects that include a string result.
                if parsed
                    .as_object()
                    .is_some_and(Self::is_image_payload_candidate)
                {
                    return Some(parsed);
                }
            }
        }

        None
    }

    /// Transform an MCP result based on the configured response format.
    ///
    /// Returns a `ResponseOutputItem` from the protocols crate.
    pub fn transform(
        result: &Value,
        format: &ResponseFormat,
        tool_call_id: &str,
        server_label: &str,
        tool_name: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        match format {
            ResponseFormat::Passthrough => {
                Self::to_mcp_call(result, tool_call_id, server_label, tool_name, arguments)
            }
            ResponseFormat::WebSearchCall => Self::to_web_search_call(result, tool_call_id),
            ResponseFormat::CodeInterpreterCall => {
                Self::to_code_interpreter_call(result, tool_call_id)
            }
            ResponseFormat::ImageGenerationCall => {
                Self::to_image_generation_call(result, tool_call_id)
            }
            ResponseFormat::FileSearchCall => Self::to_file_search_call(result, tool_call_id),
        }
    }

    /// Transform to mcp_call output (passthrough).
    fn to_mcp_call(
        result: &Value,
        tool_call_id: &str,
        server_label: &str,
        tool_name: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        ResponseOutputItem::McpCall {
            id: mcp_response_item_id(tool_call_id),
            status: "completed".to_string(),
            approval_request_id: None,
            arguments: arguments.to_string(),
            error: None,
            name: tool_name.to_string(),
            output: Self::flatten_mcp_output(result),
            server_label: server_label.to_string(),
        }
    }

    /// Flatten passthrough MCP results into plain text for OpenAI-compatible output.
    fn flatten_mcp_output(result: &Value) -> String {
        match result {
            Value::String(text) => text.clone(),
            _ => {
                let mut text_parts = Vec::new();
                Self::collect_text_parts(result, &mut text_parts);
                if text_parts.is_empty() {
                    result.to_string()
                } else {
                    text_parts.join("\n")
                }
            }
        }
    }

    fn collect_text_parts(value: &Value, text_parts: &mut Vec<String>) {
        match value {
            Value::Array(items) => {
                for item in items {
                    Self::collect_text_parts(item, text_parts);
                }
            }
            Value::Object(obj) => {
                if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                    }
                    return;
                }

                if obj.get("type").is_none() {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                        return;
                    }
                }

                if let Some(message) = obj.get("message").and_then(|v| v.as_str()) {
                    text_parts.push(message.to_string());
                    return;
                }

                if let Some(error) = obj.get("error") {
                    let before = text_parts.len();
                    Self::collect_text_parts(error, text_parts);
                    if text_parts.len() > before {
                        return;
                    }
                }

                if let Some(content) = obj.get("content") {
                    Self::collect_text_parts(content, text_parts);
                }
            }
            _ => {}
        }
    }

    /// Transform MCP web search results to OpenAI web_search_call format.
    fn to_web_search_call(result: &Value, tool_call_id: &str) -> ResponseOutputItem {
        let sources = Self::extract_web_sources(result);
        let queries = Self::extract_queries(result);

        ResponseOutputItem::WebSearchCall {
            id: format!("ws_{tool_call_id}"),
            status: WebSearchCallStatus::Completed,
            action: WebSearchAction::Search {
                query: queries.first().cloned(),
                queries,
                sources,
            },
        }
    }

    /// Transform MCP code interpreter results to OpenAI code_interpreter_call format.
    fn to_code_interpreter_call(result: &Value, tool_call_id: &str) -> ResponseOutputItem {
        let obj = result.as_object();

        let container_id = obj
            .and_then(|o| o.get("container_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let code = obj
            .and_then(|o| o.get("code"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let outputs = Self::extract_code_outputs(result);

        ResponseOutputItem::CodeInterpreterCall {
            id: format!("ci_{tool_call_id}"),
            status: CodeInterpreterCallStatus::Completed,
            container_id,
            code,
            outputs: (!outputs.is_empty()).then_some(outputs),
        }
    }

    /// Transform MCP image generation results to OpenAI image_generation_call format.
    fn to_image_generation_call(result: &Value, tool_call_id: &str) -> ResponseOutputItem {
        let payload = Self::image_payload_from_wrapped_content(result)
            .unwrap_or_else(|| Value::String(Self::flatten_mcp_output(result)));
        let parsed_payload = payload
            .as_str()
            .and_then(|s| serde_json::from_str::<Value>(s).ok());
        let obj = payload
            .as_object()
            .or_else(|| parsed_payload.as_ref().and_then(|v| v.as_object()));

        let status = ImageGenerationCallStatus::Completed;
        let output_result = payload
            .as_object()
            .and_then(|obj| obj.get("result"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                parsed_payload
                    .as_ref()
                    .and_then(|v| v.as_object())
                    .and_then(|obj| obj.get("result"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .or_else(|| extract_image_generation_fallback_text(result))
            .or_else(|| payload.as_str().map(String::from))
            .or_else(|| result.as_str().map(String::from))
            .or_else(|| Some(payload.to_string()));

        ResponseOutputItem::ImageGenerationCall {
            id: format!("ig_{tool_call_id}"),
            status,
            result: output_result,
            revised_prompt: obj
                .and_then(|o| o.get("revised_prompt"))
                .and_then(|v| v.as_str())
                .map(String::from),
            background: obj
                .and_then(|o| o.get("background"))
                .and_then(|v| v.as_str())
                .map(String::from),
            output_format: obj
                .and_then(|o| o.get("output_format"))
                .and_then(|v| v.as_str())
                .map(String::from),
            quality: obj
                .and_then(|o| o.get("quality"))
                .and_then(|v| v.as_str())
                .map(String::from),
            size: obj
                .and_then(|o| o.get("size"))
                .and_then(|v| v.as_str())
                .map(String::from),
            action: obj
                .and_then(|o| o.get("action"))
                .and_then(|v| v.as_str())
                .map(String::from),
        }
    }

    /// Transform MCP file search results to OpenAI file_search_call format.
    fn to_file_search_call(result: &Value, tool_call_id: &str) -> ResponseOutputItem {
        let obj = result.as_object();

        let queries = Self::extract_queries(result);
        let results = Self::extract_file_results(result);

        ResponseOutputItem::FileSearchCall {
            id: format!("fs_{tool_call_id}"),
            status: FileSearchCallStatus::Completed,
            queries: if queries.is_empty() {
                obj.and_then(|o| o.get("query"))
                    .and_then(|v| v.as_str())
                    .map(|q| vec![q.to_string()])
                    .unwrap_or_default()
            } else {
                queries
            },
            results: (!results.is_empty()).then_some(results),
        }
    }

    /// Extract web sources from MCP result.
    fn extract_web_sources(result: &Value) -> Vec<WebSearchSource> {
        let maybe_array = result.as_array().or_else(|| {
            result
                .as_object()
                .and_then(|obj| obj.get("results"))
                .and_then(|v| v.as_array())
        });

        maybe_array
            .map(|arr| arr.iter().filter_map(Self::parse_web_source).collect())
            .unwrap_or_default()
    }

    /// Parse a single web source from JSON.
    fn parse_web_source(item: &Value) -> Option<WebSearchSource> {
        let obj = item.as_object()?;
        let url = obj.get("url").and_then(|v| v.as_str())?;
        Some(WebSearchSource {
            source_type: "url".to_string(),
            url: url.to_string(),
        })
    }

    /// Extract queries from MCP result.
    fn extract_queries(result: &Value) -> Vec<String> {
        result
            .as_object()
            .and_then(|obj| obj.get("queries"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract code interpreter outputs from MCP result.
    fn extract_code_outputs(result: &Value) -> Vec<CodeInterpreterOutput> {
        let mut outputs = Vec::new();

        if let Some(obj) = result.as_object() {
            // Check for logs/stdout
            if let Some(logs) = obj.get("logs").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Logs {
                    logs: logs.to_string(),
                });
            }
            if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Logs {
                    logs: stdout.to_string(),
                });
            }

            // Check for image outputs
            if let Some(image_url) = obj.get("image_url").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Image {
                    url: image_url.to_string(),
                });
            }

            // Check for outputs array
            if let Some(out_array) = obj.get("outputs").and_then(|v| v.as_array()) {
                outputs.extend(out_array.iter().filter_map(|item| {
                    let item_obj = item.as_object()?;
                    match item_obj.get("type").and_then(|v| v.as_str())? {
                        "logs" => item_obj.get("logs").and_then(|v| v.as_str()).map(|logs| {
                            CodeInterpreterOutput::Logs {
                                logs: logs.to_string(),
                            }
                        }),
                        "image" => item_obj.get("url").and_then(|v| v.as_str()).map(|url| {
                            CodeInterpreterOutput::Image {
                                url: url.to_string(),
                            }
                        }),
                        _ => None,
                    }
                }));
            }
        }

        outputs
    }

    /// Extract file search results from MCP result.
    fn extract_file_results(result: &Value) -> Vec<FileSearchResult> {
        result
            .as_object()
            .and_then(|obj| obj.get("results"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(Self::parse_file_result).collect())
            .unwrap_or_default()
    }

    /// Parse a file search result from JSON.
    fn parse_file_result(item: &Value) -> Option<FileSearchResult> {
        let obj = item.as_object()?;
        let file_id = obj.get("file_id").and_then(|v| v.as_str())?.to_string();
        let filename = obj.get("filename").and_then(|v| v.as_str())?.to_string();
        let text = obj.get("text").and_then(|v| v.as_str()).map(String::from);
        let score = obj.get("score").and_then(|v| v.as_f64()).map(|f| f as f32);

        Some(FileSearchResult {
            file_id,
            filename,
            text,
            score,
            attributes: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, to_value};

    use super::*;

    #[test]
    fn test_passthrough_transform() {
        let result = json!({"key": "value"});
        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "call_test-1",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { id, output, .. } => {
                assert_eq!(id, "mcp_test-1");
                assert!(output.contains("key"));
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_flattens_single_text_block() {
        let result = json!([
            {"type": "text", "text": "hello from mcp"}
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-2",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "hello from mcp");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_fc_id_to_mcp_prefix() {
        let result = json!({"key": "value"});
        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "fc_abc123",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { id, .. } => {
                assert_eq!(id, "mcp_abc123");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_flattens_multiple_text_blocks() {
        let result = json!([
            {"type": "text", "text": "first block"},
            {"type": "text", "text": "second block"}
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-3",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "first block\nsecond block");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_preserves_existing_mcp_prefix() {
        let result = json!({"key": "value"});
        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "mcp_existing",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { id, .. } => {
                assert_eq!(id, "mcp_existing");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_ignores_non_text_blocks() {
        let result = json!([
            {"type": "text", "text": "kept text"},
            {"type": "image", "url": "https://example.com/image.png"},
            {"type": "resource", "uri": "file:///tmp/test.txt"}
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-4",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "kept text");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_ignores_typed_non_text_blocks_with_text_fields() {
        let result = json!([
            {"type": "text", "text": "kept text"},
            {"type": "image", "text": "caption that should be ignored"}
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-4b",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "kept text");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_uses_error_message_for_structured_errors() {
        let result = json!({
            "error": {
                "code": "tool_failed",
                "message": "tool execution failed"
            }
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-5",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "tool execution failed");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_transform_keeps_content_when_error_has_no_text() {
        let result = json!([
            {"type": "text", "text": "hello"},
            {
                "error": {"code": "tool_failed"},
                "content": [
                    {"type": "text", "text": "important"}
                ]
            }
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-6",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { output, .. } => {
                assert_eq!(output, "hello\nimportant");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_web_search_transform() {
        let result = json!({
            "results": [
                {"url": "https://example.com", "title": "Example"},
                {"url": "https://rust-lang.org", "title": "Rust"}
            ]
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::WebSearchCall,
            "req-123",
            "server",
            "web_search",
            "{}",
        );

        match transformed {
            ResponseOutputItem::WebSearchCall { id, status, action } => {
                assert_eq!(id, "ws_req-123");
                assert_eq!(status, WebSearchCallStatus::Completed);
                match action {
                    WebSearchAction::Search { sources, .. } => {
                        assert_eq!(sources.len(), 2);
                        assert_eq!(sources[0].url, "https://example.com");
                    }
                    _ => panic!("Expected Search action"),
                }
            }
            _ => panic!("Expected WebSearchCall"),
        }
    }

    #[test]
    fn test_code_interpreter_transform() {
        let result = json!({
            "code": "print('hello')",
            "container_id": "cntr_abc123",
            "stdout": "hello\n"
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::CodeInterpreterCall,
            "req-456",
            "server",
            "code_interpreter",
            "{}",
        );

        match transformed {
            ResponseOutputItem::CodeInterpreterCall {
                id,
                status,
                code,
                outputs,
                ..
            } => {
                assert_eq!(id, "ci_req-456");
                assert_eq!(status, CodeInterpreterCallStatus::Completed);
                assert_eq!(code, Some("print('hello')".to_string()));
                assert!(outputs.is_some());
                assert_eq!(outputs.unwrap().len(), 1);
            }
            _ => panic!("Expected CodeInterpreterCall"),
        }
    }

    #[test]
    fn test_file_search_transform() {
        let result = json!({
            "query": "async patterns",
            "results": [
                {"file_id": "file_1", "filename": "async.md", "score": 0.95, "text": "..."},
                {"file_id": "file_2", "filename": "patterns.md", "score": 0.87}
            ]
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::FileSearchCall,
            "req-789",
            "server",
            "file_search",
            "{}",
        );

        match transformed {
            ResponseOutputItem::FileSearchCall {
                id,
                status,
                queries,
                results,
            } => {
                assert_eq!(id, "fs_req-789");
                assert_eq!(status, FileSearchCallStatus::Completed);
                assert_eq!(queries, vec!["async patterns"]);
                let results = results.unwrap();
                assert_eq!(results.len(), 2);
                assert_eq!(results[0].file_id, "file_1");
                assert_eq!(results[0].score, Some(0.95));
            }
            _ => panic!("Expected FileSearchCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_wrapped_content_extracts_metadata() {
        let result = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "{\"result\":\"ZmFrZV9iYXNlNjQ=\",\"status\":\"completed\",\"action\":\"generate\",\"background\":\"opaque\",\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"1024x1024\",\"revised_prompt\":\"rp\"}"
                    }
                ]
            }
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-1003",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall {
                id,
                status,
                result,
                action,
                background,
                output_format,
                quality,
                size,
                revised_prompt,
            } => {
                assert_eq!(id, "ig_req-1003");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
                assert_eq!(action.as_deref(), Some("generate"));
                assert_eq!(background.as_deref(), Some("opaque"));
                assert_eq!(output_format.as_deref(), Some("png"));
                assert_eq!(quality.as_deref(), Some("high"));
                assert_eq!(size.as_deref(), Some("1024x1024"));
                assert_eq!(revised_prompt.as_deref(), Some("rp"));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_output_shape_matches_dataplane() {
        let result = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "{\"result\":\"ZmFrZV9iYXNlNjQ=\",\"action\":\"generate\",\"background\":\"opaque\",\"output_format\":\"png\",\"quality\":\"high\"}"
                    }
                ],
                "isError": false
            }
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-shape",
            "server",
            "image_generation",
            "{}",
        );

        let item = to_value(&transformed).expect("image_generation_call should serialize");
        assert_eq!(item["type"], "image_generation_call");
        assert_eq!(item["id"], "ig_req-shape");
        assert_eq!(item["status"], "completed");
        assert_eq!(item["result"], "ZmFrZV9iYXNlNjQ=");
        assert_eq!(item["action"], "generate");
        assert_eq!(item["background"], "opaque");
        assert_eq!(item["output_format"], "png");
        assert_eq!(item["quality"], "high");
    }

    #[test]
    fn test_image_generation_transform_wrapped_content_skips_non_image_json_text() {
        let result = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "{\"foo\":\"bar\",\"trace_id\":\"abc\"}"
                    },
                    {
                        "type": "text",
                        "text": "{\"result\":\"ZmFrZV9iYXNlNjQ=\",\"status\":\"completed\"}"
                    }
                ]
            }
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-1004",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall {
                id, status, result, ..
            } => {
                assert_eq!(id, "ig_req-1004");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }
}
