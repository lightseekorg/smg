//! Response transformer for MCP to OpenAI format conversion.

use openai_protocol::responses::{
    CodeInterpreterCallStatus, CodeInterpreterOutput, FileSearchCallStatus, FileSearchResult,
    ResponseOutputItem, WebSearchAction, WebSearchCallStatus, WebSearchSource,
};

use super::ResponseFormat;

/// Transforms MCP CallToolResult to OpenAI Responses API output items.
pub struct ResponseTransformer;

impl ResponseTransformer {
    /// Transform an MCP result based on the configured response format.
    ///
    /// Returns a `ResponseOutputItem` from the protocols crate.
    pub fn transform(
        result: &serde_json::Value,
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
            ResponseFormat::FileSearchCall => Self::to_file_search_call(result, tool_call_id),
        }
    }

    /// Transform to mcp_call output (passthrough).
    fn to_mcp_call(
        result: &serde_json::Value,
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
    fn to_web_search_call(result: &serde_json::Value, tool_call_id: &str) -> ResponseOutputItem {
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
        result: &serde_json::Value,
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

    /// Transform MCP file search results to OpenAI file_search_call format.
    fn to_file_search_call(result: &serde_json::Value, tool_call_id: &str) -> ResponseOutputItem {
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
    fn extract_web_sources(result: &serde_json::Value) -> Vec<WebSearchSource> {
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
    fn parse_web_source(item: &serde_json::Value) -> Option<WebSearchSource> {
        let obj = item.as_object()?;
        let url = obj.get("url").and_then(|v| v.as_str())?;
        Some(WebSearchSource {
            source_type: "url".to_string(),
            url: url.to_string(),
        })
    }

    /// Extract queries from MCP result.
    fn extract_queries(result: &serde_json::Value) -> Vec<String> {
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
    fn extract_code_outputs(result: &serde_json::Value) -> Vec<CodeInterpreterOutput> {
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
    fn extract_file_results(result: &serde_json::Value) -> Vec<FileSearchResult> {
        result
            .as_object()
            .and_then(|obj| obj.get("results"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(Self::parse_file_result).collect())
            .unwrap_or_default()
    }

    /// Parse a file search result from JSON.
    fn parse_file_result(item: &serde_json::Value) -> Option<FileSearchResult> {
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
}
