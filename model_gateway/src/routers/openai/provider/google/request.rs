use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use super::{super::types::strip_sglang_fields, GoogleProvider};
use crate::worker::Endpoint;

impl GoogleProvider {
    pub(super) fn normalize_model(model: &str) -> String {
        model
            .trim_start_matches("google.")
            .trim_start_matches("google/")
            .trim_start_matches("models/")
            .trim()
            .to_string()
    }

    fn normalize_role(role: Option<&str>) -> String {
        let role = role.unwrap_or("user");
        if role.eq_ignore_ascii_case("assistant") {
            "model".to_string()
        } else if role.trim().is_empty() {
            "user".to_string()
        } else {
            role.to_string()
        }
    }

    fn map_tool_choice_mode(mode: &str) -> Option<&'static str> {
        match mode.to_ascii_lowercase().as_str() {
            "none" => Some("NONE"),
            "required" => Some("ANY"),
            "auto" => Some("AUTO"),
            _ => None,
        }
    }

    fn parse_data_uri(value: &str) -> Option<(Option<String>, String)> {
        if !value.starts_with("data:") {
            return None;
        }
        let comma = value.find(',')?;
        let meta = &value[5..comma];
        let data = value[comma + 1..].to_string();
        let mime = if meta.is_empty() {
            None
        } else {
            let mime_candidate = meta.split(';').next().unwrap_or_default().trim();
            if mime_candidate.is_empty() {
                None
            } else {
                Some(mime_candidate.to_string())
            }
        };
        Some((mime, data))
    }

    fn parse_json_or_value(input: &Value) -> Value {
        if let Some(s) = input.as_str() {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return json!({});
            }
            serde_json::from_str::<Value>(trimmed).unwrap_or_else(|_| Value::String(s.to_string()))
        } else if input.is_null() {
            json!({})
        } else {
            input.clone()
        }
    }

    fn normalize_response_object(input: Value) -> Value {
        if input.is_object() {
            input
        } else {
            json!({ "result": input })
        }
    }

    fn infer_mime_type(uri: &str) -> Option<&'static str> {
        let path = uri.split('?').next().unwrap_or(uri).to_ascii_lowercase();
        let ext = path.rsplit('.').next()?;
        match ext {
            "png" => Some("image/png"),
            "jpeg" | "jpg" => Some("image/jpeg"),
            "webp" => Some("image/webp"),
            "heic" => Some("image/heic"),
            "heif" => Some("image/heif"),
            "pdf" => Some("application/pdf"),
            "txt" => Some("text/plain"),
            "md" => Some("text/markdown"),
            "html" | "htm" => Some("text/html"),
            "xml" => Some("text/xml"),
            _ => None,
        }
    }

    fn extract_tool_choice_function_name(obj: &serde_json::Map<String, Value>) -> Option<String> {
        obj.get("name")
            .and_then(Value::as_str)
            .or_else(|| {
                obj.get("function")
                    .and_then(Value::as_object)
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
            })
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
    }

    fn sanitize_key_for_gemini(key: &str) -> Option<String> {
        let mut cleaned: String = key
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        cleaned = cleaned.trim_matches('_').to_string();
        cleaned = cleaned
            .trim_start_matches(|ch: char| ch.is_ascii_digit())
            .to_string();
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    fn normalize_type_node(type_node: &Value) -> Option<Value> {
        match type_node {
            Value::String(s) => Some(Value::String(s.to_ascii_uppercase())),
            Value::Array(arr) => {
                let vals: Vec<Value> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| Value::String(s.to_ascii_uppercase())))
                    .collect();
                if vals.is_empty() {
                    None
                } else {
                    Some(Value::Array(vals))
                }
            }
            _ => None,
        }
    }

    fn resolve_definition(
        key: &str,
        defs: &serde_json::Map<String, Value>,
        cache: &mut HashMap<String, Value>,
        active: &mut HashSet<String>,
    ) -> Option<Value> {
        if let Some(cached) = cache.get(key) {
            return Some(cached.clone());
        }
        if active.contains(key) {
            return None;
        }
        let definition = defs.get(key)?.as_object()?.clone();
        active.insert(key.to_string());
        let mut resolved = Value::Object(definition);
        Self::replace_refs(&mut resolved, defs, cache, active);
        active.remove(key);
        cache.insert(key.to_string(), resolved.clone());
        Some(resolved)
    }

    fn replace_refs(
        node: &mut Value,
        defs: &serde_json::Map<String, Value>,
        cache: &mut HashMap<String, Value>,
        active: &mut HashSet<String>,
    ) {
        match node {
            Value::Object(map) => {
                let ref_value = map
                    .get("$ref")
                    .or_else(|| map.get("_ref"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                if let Some(r) = ref_value {
                    let key = r
                        .strip_prefix("#/$defs/")
                        .or_else(|| r.strip_prefix("#/defs/"))
                        .map(str::to_string);
                    if let Some(key) = key {
                        if let Some(replacement) =
                            Self::resolve_definition(&key, defs, cache, active)
                        {
                            *node = replacement;
                            return;
                        }
                    }
                }
                for v in map.values_mut() {
                    Self::replace_refs(v, defs, cache, active);
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    Self::replace_refs(item, defs, cache, active);
                }
            }
            _ => {}
        }
    }

    fn flatten_refs_from_defs(schema: &Value) -> Value {
        let mut root = schema.clone();
        let defs = root
            .as_object()
            .and_then(|o| o.get("$defs").or_else(|| o.get("_defs")))
            .and_then(Value::as_object)
            .cloned();
        if let Some(defs) = defs {
            if let Some(obj) = root.as_object_mut() {
                obj.remove("$defs");
                obj.remove("_defs");
            }
            let mut cache = HashMap::new();
            let mut active = HashSet::new();
            Self::replace_refs(&mut root, &defs, &mut cache, &mut active);
        }
        root
    }

    fn sanitize_schema_node(node: &Value) -> Value {
        match node {
            Value::Null => Value::Null,
            Value::Array(arr) => Value::Array(
                arr.iter()
                    .map(Self::sanitize_schema_node)
                    .filter(|v| !v.is_null())
                    .collect(),
            ),
            Value::Object(obj) => {
                let mut out = serde_json::Map::new();
                for (k, v) in obj {
                    if k == "$defs" || k == "_defs" || k == "$ref" || k == "_ref" {
                        continue;
                    }
                    let out_key = match Self::sanitize_key_for_gemini(k) {
                        Some(s) => s,
                        None => continue,
                    };
                    let out_value = if k == "required" && v.is_array() {
                        let req_vals: Vec<Value> = v
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .filter_map(Value::as_str)
                            .filter_map(Self::sanitize_key_for_gemini)
                            .map(Value::String)
                            .collect();
                        Value::Array(req_vals)
                    } else {
                        Self::sanitize_schema_node(v)
                    };
                    if !out_value.is_null() {
                        out.insert(out_key, out_value);
                    }
                }
                Value::Object(out)
            }
            _ => node.clone(),
        }
    }

    fn to_gemini_schema(schema: &Value) -> Value {
        if !schema.is_object() {
            return json!({});
        }
        let flattened = Self::flatten_refs_from_defs(schema);
        let sanitized = Self::sanitize_schema_node(&flattened);
        if let Some(obj) = sanitized.as_object() {
            let mut out = serde_json::Map::new();
            if let Some(t) = obj.get("type") {
                if let Some(nt) = Self::normalize_type_node(t) {
                    out.insert("type".to_string(), nt);
                }
            }
            for key in [
                "properties",
                "items",
                "additionalProperties",
                "enum",
                "format",
                "title",
                "minimum",
                "maximum",
                "nullable",
                "prefixItems",
                "minItems",
                "maxItems",
                "description",
                "required",
            ] {
                if let Some(v) = obj.get(key) {
                    out.insert(key.to_string(), v.clone());
                }
            }
            Value::Object(out)
        } else {
            json!({})
        }
    }
    pub(super) fn transform_request_impl(payload: &mut Value, endpoint: Endpoint) {
        strip_sglang_fields(payload);
        if endpoint == Endpoint::Chat {
            if let Some(obj) = payload.as_object_mut() {
                if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                    obj.remove("logprobs");
                }
            }
            return;
        }
        if endpoint != Endpoint::Responses {
            return;
        }
        if let Some(input) = payload.get("input") {
            if payload.get("contents").is_some() {
                return;
            }

            let mut contents = Vec::new();
            let mut call_id_to_name = HashMap::<String, String>::new();

            let normalized_input: Vec<&Value> = match input {
                Value::Array(items) => items.iter().collect(),
                other => vec![other],
            };

            for item in normalized_input {
                if let Some(text) = item.as_str() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        contents.push(json!({
                            "role": "user",
                            "parts": [{"text": text}],
                        }));
                    }
                    continue;
                }

                if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                    if let (Some(call_id), Some(name)) = (
                        item.get("call_id").and_then(|v| v.as_str()),
                        item.get("name").and_then(|v| v.as_str()),
                    ) {
                        call_id_to_name.insert(call_id.to_string(), name.to_string());
                    }
                }

                if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown_tool");
                    let args = item
                        .get("arguments")
                        .map(Self::parse_json_or_value)
                        .unwrap_or_else(|| json!({}));
                    contents.push(json!({
                        "role": "model",
                        "parts": [{"functionCall": {"name": name, "args": args}}],
                    }));
                    continue;
                }

                if item.get("type").and_then(|v| v.as_str()) == Some("function_call_output") {
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            item.get("call_id")
                                .and_then(|v| v.as_str())
                                .and_then(|id| call_id_to_name.get(id).map(String::as_str))
                        })
                        .unwrap_or("unknown_tool");
                    let response_obj = item
                        .get("output")
                        .cloned()
                        .or_else(|| item.get("content").cloned())
                        .map(|v| Self::normalize_response_object(Self::parse_json_or_value(&v)))
                        .unwrap_or_else(|| json!({}));
                    contents.push(json!({
                        "role": "tool",
                        "parts": [{"functionResponse": {"name": name, "response": response_obj}}],
                    }));
                    continue;
                }

                if item.get("type").and_then(|v| v.as_str()) == Some("tool_response")
                    && item.get("role").and_then(|v| v.as_str()) == Some("tool")
                {
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown_tool");
                    let response_obj = item
                        .get("output")
                        .cloned()
                        .or_else(|| item.get("content").cloned())
                        .map(|v| Self::normalize_response_object(Self::parse_json_or_value(&v)))
                        .unwrap_or_else(|| json!({}));
                    contents.push(json!({
                        "role": "tool",
                        "parts": [{"functionResponse": {"name": name, "response": response_obj}}],
                    }));
                    continue;
                }

                let role = item.get("role").and_then(|v| v.as_str());
                let mut parts_out = Vec::<Value>::new();
                if let Some(content) = item.get("content") {
                    if let Some(s) = content.as_str() {
                        parts_out.push(json!({"text": s}));
                    } else if let Some(parts) = content.as_array() {
                        for p in parts {
                            match p.get("type").and_then(|v| v.as_str()) {
                                Some("input_text") | Some("output_text") => {
                                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                                        parts_out.push(json!({"text": t}));
                                    }
                                }
                                Some("input_image") => {
                                    if let Some(uri) = p.get("image_url").and_then(|v| v.as_str()) {
                                        if let Some((mime, data)) = Self::parse_data_uri(uri) {
                                            parts_out.push(json!({"inlineData": {"mimeType": mime.unwrap_or_else(|| "image/jpeg".to_string()), "data": data}}));
                                        } else {
                                            parts_out.push(json!({"fileData": {"fileUri": uri, "mimeType": Self::infer_mime_type(uri).unwrap_or("image/jpeg")}}));
                                        }
                                    }
                                }
                                Some("input_file") => {
                                    if let Some(uri) = p.get("file_url").and_then(|v| v.as_str()) {
                                        let mut fd = json!({"fileUri": uri});
                                        if let Some(mime) = Self::infer_mime_type(uri) {
                                            fd["mimeType"] = json!(mime);
                                        }
                                        parts_out.push(json!({"fileData": fd}));
                                    } else if let Some(data) =
                                        p.get("file_data").and_then(|v| v.as_str())
                                    {
                                        if data.trim().is_empty() {
                                            continue;
                                        }
                                        if let Some((mime, decoded)) = Self::parse_data_uri(data) {
                                            parts_out.push(json!({"inlineData": {"data": decoded, "mimeType": mime.unwrap_or_else(|| p.get("mime_type").and_then(|m| m.as_str()).unwrap_or("application/octet-stream").to_string())}}));
                                        } else {
                                            let mut mime = p
                                                .get("mime_type")
                                                .and_then(|m| m.as_str())
                                                .map(ToString::to_string);
                                            if mime
                                                .as_ref()
                                                .map(|s| s.trim().is_empty())
                                                .unwrap_or(true)
                                            {
                                                mime = p
                                                    .get("filename")
                                                    .and_then(|f| f.as_str())
                                                    .and_then(Self::infer_mime_type)
                                                    .map(ToString::to_string);
                                            }
                                            if let Some(mime) = mime {
                                                parts_out.push(
                                                    json!({"inlineData": {"data": data, "mimeType": mime}}),
                                                );
                                            } else {
                                                parts_out.push(json!({
                                                    "inlineData": {
                                                        "data": data,
                                                        "mimeType": "application/octet-stream"
                                                    }
                                                }));
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                if !parts_out.is_empty() {
                    contents.push(json!({
                        "role": Self::normalize_role(role),
                        "parts": parts_out,
                    }));
                }
            }

            let mut out = json!({
                "model": payload
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(Self::normalize_model)
                    .unwrap_or_else(|| "gemini".to_string()),
                "contents": contents,
            });

            if let Some(sys) = payload.get("instructions") {
                let mut parts = vec![];
                if let Some(text) = sys.as_str() {
                    parts.push(json!({"text": text}));
                } else if let Some(arr) = sys.as_array() {
                    for x in arr {
                        if let Some(s) = x.as_str() {
                            parts.push(json!({"text": s}));
                        }
                    }
                }
                if !parts.is_empty() {
                    out["system_instruction"] = json!({"parts": parts});
                }
            }

            if let Some(tools) = payload.get("tools").and_then(|v| v.as_array()) {
                let mut decls = Vec::new();
                for tool in tools {
                    if tool.get("type").and_then(|v| v.as_str()) == Some("function") {
                        let fn_node = tool.get("function").unwrap_or(tool);
                        if let Some(name) = fn_node.get("name").and_then(|v| v.as_str()) {
                            if !name.is_empty() {
                                let mut fd = json!({"name": name});
                                if let Some(desc) = fn_node.get("description") {
                                    fd["description"] = desc.clone();
                                }
                                if let Some(params) = fn_node.get("parameters") {
                                    let sanitized = if name.starts_with("mcp__hosted") {
                                        Self::to_gemini_schema(params)
                                    } else {
                                        params.clone()
                                    };
                                    fd["parameters"] = sanitized;
                                }
                                decls.push(fd);
                            }
                        }
                    }
                }
                if !decls.is_empty() {
                    out["tools"] = json!([{"functionDeclarations": decls}]);
                }
            }

            if payload
                .get("parallel_tool_calls")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                out["toolConfig"] = json!({
                    "functionCallingConfig": {
                        "mode": "ANY"
                    }
                });
            }

            if let Some(tool_choice) = payload.get("tool_choice") {
                let mut mode: Option<&'static str> = None;
                let mut allowed: Vec<String> = vec![];
                if let Some(s) = tool_choice.as_str() {
                    mode = Self::map_tool_choice_mode(s);
                } else if let Some(obj) = tool_choice.as_object() {
                    if let Some(t) = obj.get("type").and_then(|v| v.as_str()) {
                        if t.eq_ignore_ascii_case("function") {
                            mode = Some("ANY");
                            if let Some(name) = Self::extract_tool_choice_function_name(obj) {
                                allowed.push(name.to_string());
                            }
                        } else if t.eq_ignore_ascii_case("allowed_tools") {
                            mode = obj
                                .get("mode")
                                .and_then(|v| v.as_str())
                                .and_then(Self::map_tool_choice_mode);
                            if let Some(tools) = obj.get("tools").and_then(|v| v.as_array()) {
                                for t in tools {
                                    if t.get("type").and_then(|v| v.as_str()) == Some("function") {
                                        if let Some(obj) = t.as_object() {
                                            if let Some(name) =
                                                Self::extract_tool_choice_function_name(obj)
                                            {
                                                allowed.push(name);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if mode.is_none() {
                        mode = obj
                            .get("mode")
                            .and_then(|v| v.as_str())
                            .and_then(Self::map_tool_choice_mode);
                    }
                }

                if let Some(mode) = mode {
                    let mut cfg = json!({"mode": mode});
                    if mode == "ANY" && !allowed.is_empty() {
                        cfg["allowedFunctionNames"] = json!(allowed);
                    }
                    out["toolConfig"] = json!({"functionCallingConfig": cfg});
                }
            }

            let mut generation_config = serde_json::Map::new();
            if let Some(reasoning) = payload.get("reasoning").and_then(|v| v.as_object()) {
                let mut thinking = serde_json::Map::new();
                if let Some(effort) = reasoning.get("effort").and_then(|v| v.as_str()) {
                    let level = match effort.to_ascii_lowercase().as_str() {
                        "none" | "minimal" => Some("MINIMAL"),
                        "low" => Some("LOW"),
                        "medium" => Some("MEDIUM"),
                        "high" | "xhigh" => Some("HIGH"),
                        _ => None,
                    };
                    if let Some(level) = level {
                        thinking.insert("thinkingLevel".to_string(), json!(level));
                    }
                }
                if let Some(summary) = reasoning.get("summary").and_then(|v| v.as_str()) {
                    let normalized = summary.trim().to_ascii_lowercase();
                    if matches!(normalized.as_str(), "auto" | "concise" | "detailed") {
                        thinking.insert("includeThoughts".to_string(), json!(true));
                    }
                }
                if !thinking.is_empty() {
                    generation_config.insert("thinkingConfig".to_string(), Value::Object(thinking));
                }
            }
            if let Some(ftype) = payload
                .get("text")
                .and_then(|t| t.get("format"))
                .and_then(|f| f.get("type"))
                .and_then(Value::as_str)
                .map(str::to_ascii_lowercase)
            {
                if ftype == "json_object" || ftype == "json_schema" {
                    generation_config
                        .insert("responseMimeType".to_string(), json!("application/json"));
                }
                if ftype == "json_schema" {
                    if let Some(schema) = payload
                        .get("text")
                        .and_then(|t| t.get("format"))
                        .and_then(|f| f.get("schema"))
                    {
                        generation_config.insert(
                            "responseJsonSchema".to_string(),
                            Self::to_gemini_schema(schema),
                        );
                    }
                }
            }
            if !generation_config.is_empty() {
                out["generationConfig"] = Value::Object(generation_config);
            }
            *payload = out;
        }
    }
}
