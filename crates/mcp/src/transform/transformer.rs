//! Response transformer for MCP to OpenAI format conversion.

use openai_protocol::responses::{
    CodeInterpreterCallStatus, CodeInterpreterOutput, FileSearchCallStatus, FileSearchResult,
    ImageGenerationCallStatus, ResponseOutputItem, WebSearchAction, WebSearchCallStatus,
    WebSearchSource,
};
use serde_json::Value;

use super::ResponseFormat;

/// Transforms MCP CallToolResult to OpenAI Responses API output items.
pub struct ResponseTransformer;

impl ResponseTransformer {
    fn image_payload_from_wrapped_content(result: &Value) -> Option<Value> {
        // Already a direct object payload.
        if result.as_object().is_some_and(|obj| {
            obj.contains_key("result")
                || obj.contains_key("status")
                || obj.contains_key("error")
                || obj.contains_key("revised_prompt")
                || obj.contains_key("output_format")
                || obj.contains_key("background")
                || obj.contains_key("quality")
                || obj.contains_key("size")
                || obj.contains_key("action")
        }) {
            return Some(result.clone());
        }

        // Handle MCP CallToolResult-style wrapper:
        // [{"type":"text","text":"{...image_generation_call payload...}"}]
        if let Some(arr) = result.as_array() {
            for item in arr {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                if obj.get("type").and_then(|v| v.as_str()) != Some("text") {
                    continue;
                }
                let Some(text) = obj.get("text").and_then(|v| v.as_str()) else {
                    continue;
                };
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    if parsed.is_object() {
                        return Some(parsed);
                    }
                }
            }
        }

        // Sometimes payload comes as a JSON string.
        if let Some(text) = result.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                if parsed.is_object() {
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
            id: tool_call_id.to_string(),
            status: "completed".to_string(),
            approval_request_id: None,
            arguments: arguments.to_string(),
            error: None,
            name: tool_name.to_string(),
            output: result.to_string(),
            server_label: server_label.to_string(),
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
    fn to_code_interpreter_call(
        result: &Value,
        tool_call_id: &str,
    ) -> ResponseOutputItem {
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
    fn to_image_generation_call(
        result: &Value,
        tool_call_id: &str,
    ) -> ResponseOutputItem {
        let payload = Self::image_payload_from_wrapped_content(result).unwrap_or_else(|| result.clone());
        let obj = payload.as_object();
        let explicit_status = obj
            .and_then(|o| o.get("status"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let error_message = obj
            .and_then(|o| o.get("error"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let extracted = payload
            .as_object()
            .and_then(|obj| obj.get("result"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let fallback_text = payload
            .as_str()
            .map(String::from)
            .or_else(|| result.as_str().map(String::from))
            .unwrap_or_else(|| payload.to_string());

        let status = match explicit_status.as_deref() {
            Some("in_progress") | Some("in-progress") => ImageGenerationCallStatus::InProgress,
            Some("generating") => ImageGenerationCallStatus::Generating,
            Some("incomplete") | Some("partial") => ImageGenerationCallStatus::Failed,
            Some("completed") | Some("complete") | Some("success") | Some("succeeded") => {
                ImageGenerationCallStatus::Completed
            }
            Some("failed") | Some("error") => ImageGenerationCallStatus::Failed,
            _ if error_message.is_some() => ImageGenerationCallStatus::Failed,
            _ => ImageGenerationCallStatus::Completed,
        };

        let output_result = match status {
            ImageGenerationCallStatus::InProgress | ImageGenerationCallStatus::Generating => None,
            ImageGenerationCallStatus::Failed => {
                if let Some(err) = error_message {
                    Some(err)
                } else {
                    extracted.or(Some(fallback_text))
                }
            }
            ImageGenerationCallStatus::Completed => extracted.or(Some(fallback_text)),
        };

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
    use serde_json::json;

    use super::*;

    #[test]
    fn test_passthrough_transform() {
        let result = json!({"key": "value"});
        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "test-1",
            "server",
            "tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { id, output, .. } => {
                assert_eq!(id, "test-1");
                assert!(output.contains("key"));
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
    fn test_image_generation_transform() {
        let result = json!({
            "result": "ZmFrZV9iYXNlNjQ="
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-999",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall { id, status, result, .. } => {
                assert_eq!(id, "ig_req-999");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_direct_string() {
        let result = json!("ZmFrZV9iYXNlNjQ=");

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-1000",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall { id, status, result, .. } => {
                assert_eq!(id, "ig_req-1000");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_non_string_payload() {
        let result = json!(42);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-1001",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall { id, status, result, .. } => {
                assert_eq!(id, "ig_req-1001");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("42"));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_error_payload() {
        let result = json!({
            "error": "generation failed"
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::ImageGenerationCall,
            "req-1002",
            "server",
            "image_generation",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall { id, status, result, .. } => {
                assert_eq!(id, "ig_req-1002");
                assert_eq!(status, ImageGenerationCallStatus::Failed);
                assert_eq!(result.as_deref(), Some("generation failed"));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_image_generation_transform_wrapped_content_extracts_metadata() {
        let result = json!([
            {
                "type": "text",
                "text": "{\"result\":\"ZmFrZV9iYXNlNjQ=\",\"status\":\"completed\",\"action\":\"generate\",\"background\":\"opaque\",\"output_format\":\"png\",\"quality\":\"high\",\"size\":\"1024x1024\",\"revised_prompt\":\"rp\"}"
            }
        ]);

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
    fn test_passthrough_with_explicit_image_generation_payload_is_converted() {
        let result = json!([
            {
                "type": "text",
                "text": "{\"type\":\"image_generation_call\",\"result\":\"ZmFrZV9iYXNlNjQ=\",\"status\":\"completed\",\"action\":\"generate\"}"
            }
        ]);

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "req-1004",
            "server",
            "generate_image",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall {
                id,
                status,
                result,
                action,
                ..
            } => {
                assert_eq!(id, "ig_req-1004");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
                assert_eq!(action.as_deref(), Some("generate"));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }

    #[test]
    fn test_passthrough_with_generic_action_payload_stays_mcp_call() {
        let result = json!({
            "action": "run",
            "size": "small",
            "quality": "high"
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "req-1005",
            "server",
            "some_tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::McpCall { id, .. } => {
                assert_eq!(id, "req-1005");
            }
            _ => panic!("Expected McpCall"),
        }
    }

    #[test]
    fn test_passthrough_with_image_specific_fields_is_converted() {
        let result = json!({
            "result": "ZmFrZV9iYXNlNjQ=",
            "output_format": "png",
            "model": "openai.gpt-image-1.5"
        });

        let transformed = ResponseTransformer::transform(
            &result,
            &ResponseFormat::Passthrough,
            "req-1006",
            "server",
            "some_tool",
            "{}",
        );

        match transformed {
            ResponseOutputItem::ImageGenerationCall { id, status, result, .. } => {
                assert_eq!(id, "ig_req-1006");
                assert_eq!(status, ImageGenerationCallStatus::Completed);
                assert_eq!(result.as_deref(), Some("ZmFrZV9iYXNlNjQ="));
            }
            _ => panic!("Expected ImageGenerationCall"),
        }
    }
}
