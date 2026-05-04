use serde_json::{json, Value};

use super::GoogleProvider;
use crate::worker::Endpoint;

impl GoogleProvider {
    fn message_status_from_finish_reason(finish_reason: Option<&str>) -> &'static str {
        match finish_reason
            .unwrap_or_default()
            .to_ascii_uppercase()
            .as_str()
        {
            "SAFETY" | "MAX_TOKENS" | "RECITATION" => "incomplete",
            _ => "completed",
        }
    }

    fn completion_state(candidates: Option<&Vec<Value>>) -> (&'static str, Option<&'static str>) {
        let Some(candidates) = candidates else {
            return ("failed", None);
        };
        if candidates.is_empty() {
            return ("failed", None);
        }

        let mut has_content_filter = false;
        let mut has_max_tokens = false;
        for c in candidates {
            if let Some(finish_reason) = c.get("finishReason").and_then(|v| v.as_str()) {
                match finish_reason.to_ascii_uppercase().as_str() {
                    "MAX_TOKENS" => has_max_tokens = true,
                    "SAFETY" | "RECITATION" => has_content_filter = true,
                    _ => {}
                }
            }
        }

        if has_content_filter {
            ("incomplete", Some("content_filter"))
        } else if has_max_tokens {
            ("incomplete", Some("max_output_tokens"))
        } else {
            ("completed", None)
        }
    }
    pub(super) fn transform_response_impl(response: &mut Value, endpoint: Endpoint) {
        if endpoint != Endpoint::Responses {
            return;
        }

        let model = response
            .get("modelVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("gemini")
            .to_string();

        let mut output_items = Vec::new();
        let candidates = response.get("candidates").and_then(|v| v.as_array());
        let (status, incomplete_reason) = Self::completion_state(candidates);
        if let Some(candidates) = candidates {
            for c in candidates {
                let msg_status = Self::message_status_from_finish_reason(
                    c.get("finishReason").and_then(|v| v.as_str()),
                );
                let mut content_items = Vec::new();
                let mut reasoning_items = Vec::new();
                let mut extra_output = Vec::new();
                if let Some(parts) = c
                    .get("content")
                    .and_then(|v| v.get("parts"))
                    .and_then(|v| v.as_array())
                {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                            if p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false) {
                                reasoning_items.push(json!({"type":"summary_text","text":t}));
                            } else {
                                content_items.push(json!({"type":"output_text","text":t}));
                            }
                        }
                        if let Some(fc) = p.get("functionCall") {
                            let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                            extra_output.push(json!({
                                "id": format!("fc_{}", uuid::Uuid::now_v7().to_string().replace('-', "")),
                                "type": "function_call",
                                "status": msg_status,
                                "call_id": format!("call_{}", uuid::Uuid::now_v7().to_string().replace('-', "")),
                                "name": fc.get("name").and_then(|v| v.as_str()).unwrap_or("unknown_tool"),
                                "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
                            }));
                        }
                        if let Some(fr) = p.get("functionResponse") {
                            extra_output.push(json!({
                                "type": "function_call_output",
                                "status": msg_status,
                                "call_id": fr.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                "output": fr.get("response").cloned().unwrap_or_else(|| json!({}))
                            }));
                        }
                        if let Some(inline) = p.get("inlineData") {
                            let mime = inline.get("mimeType").and_then(|v| v.as_str());
                            let data = inline.get("data").and_then(|v| v.as_str());
                            if mime.map(|m| m.starts_with("image/")).unwrap_or(false)
                                && data.is_some()
                            {
                                content_items.push(json!({"type":"image","image_base64":data}));
                            } else {
                                let mut file = serde_json::Map::new();
                                if let Some(mime) = mime {
                                    file.insert("mime_type".to_string(), json!(mime));
                                }
                                if let Some(data) = data {
                                    file.insert("data".to_string(), json!(data));
                                }
                                content_items.push(
                                    json!({"type":"output_file","file": Value::Object(file)}),
                                );
                            }
                        }
                    }
                }
                if !reasoning_items.is_empty() {
                    output_items.push(json!({
                        "id": format!("rs_{}", uuid::Uuid::now_v7().to_string().replace('-', "")),
                        "type": "reasoning",
                        "summary": reasoning_items
                    }));
                }
                if !content_items.is_empty() {
                    let role = c
                        .get("content")
                        .and_then(|v| v.get("role"))
                        .and_then(|v| v.as_str())
                        .map(|r| {
                            if r.eq_ignore_ascii_case("model") {
                                "assistant"
                            } else {
                                r
                            }
                        })
                        .unwrap_or("assistant");
                    let mut message = json!({
                        "id": format!("msg_{}", uuid::Uuid::now_v7()),
                        "type": "message",
                        "role": role,
                        "status": msg_status,
                        "content": content_items,
                    });
                    if let Some(safety) = c.get("safetyRatings").and_then(|v| v.as_array()) {
                        let annotations: Vec<Value> = safety
                            .iter()
                            .map(|r| {
                                json!({
                                    "type": "safety",
                                    "category": r.get("category").and_then(|v| v.as_str()).unwrap_or(""),
                                    "blocked": r.get("blocked").and_then(|v| v.as_bool()).unwrap_or(false)
                                })
                            })
                            .collect();
                        message["annotations"] = json!(annotations);
                    }
                    output_items.push(message);
                }
                output_items.extend(extra_output);
            }
        }

        let usage = response
            .get("usageMetadata")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let input_tokens = usage
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + usage
                .get("thoughtsTokenCount")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
        let total_tokens = usage
            .get("totalTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(input_tokens + output_tokens);

        *response = json!({
            "id": response.get("responseId").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("resp_{}", uuid::Uuid::now_v7())),
            "object": "response",
            "created_at": response
                .get("createTime")
                .and_then(|v| v.as_str())
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|d| d.timestamp())
                .unwrap_or_else(|| chrono::Utc::now().timestamp()),
            "completed_at": response
                .get("createTime")
                .and_then(|v| v.as_str())
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|d| d.timestamp())
                .unwrap_or_else(|| chrono::Utc::now().timestamp()),
            "model": model,
            "background": false,
            "status": status,
            "incomplete_details": incomplete_reason.map(|r| json!({"reason": r})),
            "error": null,
            "reasoning": {"summary": null, "effort": null},
            "output": output_items,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "cached_tokens": usage.get("cachedContentTokenCount").and_then(|v| v.as_i64()).unwrap_or(0)
                },
                "output_tokens": output_tokens,
                "output_tokens_details": {
                    "reasoning_tokens": usage.get("thoughtsTokenCount").and_then(|v| v.as_i64()).unwrap_or(0)
                },
                "total_tokens": total_tokens,
            }
        });
    }
}
