use std::{collections::BTreeMap, io::Write};

use schemars::{schema::RootSchema, schema_for};
use serde::Serialize;
use serde_json::Value;

// ============================================================================
// OpenAPI 3.1 document structure (minimal, just what we need)
// ============================================================================

#[derive(Serialize)]
struct OpenApiDoc {
    openapi: String,
    info: Info,
    paths: BTreeMap<String, PathItem>,
    components: Components,
}

#[derive(Serialize)]
struct Info {
    title: String,
    version: String,
    description: String,
}

#[derive(Serialize)]
struct Components {
    schemas: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct PathItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    get: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    post: Option<Operation>,
}

#[derive(Serialize)]
struct Operation {
    #[serde(rename = "operationId")]
    operation_id: String,
    summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "requestBody")]
    request_body: Option<RequestBody>,
    responses: BTreeMap<String, Response>,
}

#[derive(Serialize)]
struct RequestBody {
    required: bool,
    content: BTreeMap<String, MediaType>,
}

#[derive(Serialize)]
struct MediaType {
    schema: SchemaRef,
}

#[derive(Serialize)]
struct Response {
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<BTreeMap<String, MediaType>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SchemaRef {
    #[serde(rename = "$ref")]
    schema_ref: String,
}

// ============================================================================
// Schema collection
// ============================================================================

/// Post-process a JSON value tree to fix schemars output for OpenAPI compatibility:
/// 1. Rewrite `#/definitions/X` → `#/components/schemas/X`
/// 2. Replace boolean `true` in anyOf/oneOf/allOf arrays with `{}` (empty = any)
/// 3. Replace top-level boolean `true` values with `{}`
fn fixup_schema(value: &mut Value) {
    match value {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix("#/definitions/") {
                *s = format!("#/components/schemas/{rest}");
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                if *item == Value::Bool(true) {
                    *item = Value::Object(serde_json::Map::new());
                } else {
                    fixup_schema(item);
                }
            }
        }
        Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                fixup_schema(v);
            }
        }
        _ => {}
    }
}

/// Collect a root schema and all its definitions into the schemas map.
/// Returns the top-level schema name.
fn collect_schema(
    root: &RootSchema,
    schemas: &mut BTreeMap<String, Value>,
) -> anyhow::Result<String> {
    let title = root
        .schema
        .metadata
        .as_ref()
        .and_then(|m| m.title.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    // Add all definitions (sub-schemas referenced by the root)
    for (name, schema) in &root.definitions {
        let mut value = serde_json::to_value(schema)?;
        fixup_schema(&mut value);
        schemas.insert(name.clone(), value);
    }

    // Add the root schema itself (strip definitions to avoid duplication)
    let mut root_value = serde_json::to_value(&root.schema)?;
    if let Value::Object(ref mut map) = root_value {
        map.remove("definitions");
    }
    fixup_schema(&mut root_value);
    schemas.insert(title.clone(), root_value);

    Ok(title)
}

fn schema_ref(name: &str) -> SchemaRef {
    SchemaRef {
        schema_ref: format!("#/components/schemas/{name}"),
    }
}

fn json_body(name: &str) -> RequestBody {
    let mut content = BTreeMap::new();
    content.insert(
        "application/json".to_string(),
        MediaType {
            schema: schema_ref(name),
        },
    );
    RequestBody {
        required: true,
        content,
    }
}

fn json_response(name: &str, desc: &str) -> BTreeMap<String, Response> {
    let mut content = BTreeMap::new();
    content.insert(
        "application/json".to_string(),
        MediaType {
            schema: schema_ref(name),
        },
    );
    let mut responses = BTreeMap::new();
    responses.insert(
        "200".to_string(),
        Response {
            description: desc.to_string(),
            content: Some(content),
        },
    );
    responses
}

fn post_endpoint(op_id: &str, summary: &str, req_name: &str, resp_name: &str) -> PathItem {
    PathItem {
        get: None,
        post: Some(Operation {
            operation_id: op_id.to_string(),
            summary: summary.to_string(),
            request_body: Some(json_body(req_name)),
            responses: json_response(resp_name, summary),
        }),
    }
}

fn get_endpoint(op_id: &str, summary: &str, resp_name: &str) -> PathItem {
    PathItem {
        get: Some(Operation {
            operation_id: op_id.to_string(),
            summary: summary.to_string(),
            request_body: None,
            responses: json_response(resp_name, summary),
        }),
        post: None,
    }
}

// ============================================================================
// Main: generate the OpenAPI spec
// ============================================================================

fn main() -> anyhow::Result<()> {
    let mut schemas = BTreeMap::new();
    let mut paths = BTreeMap::new();

    // ---- Chat Completions ----
    use openai_protocol::chat::*;
    let req_name = collect_schema(&schema_for!(ChatCompletionRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(ChatCompletionResponse), &mut schemas)?;
    collect_schema(&schema_for!(ChatCompletionStreamResponse), &mut schemas)?;
    paths.insert(
        "/v1/chat/completions".to_string(),
        post_endpoint(
            "createChatCompletion",
            "Create chat completion",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Completions ----
    use openai_protocol::completion::*;
    let req_name = collect_schema(&schema_for!(CompletionRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(CompletionResponse), &mut schemas)?;
    collect_schema(&schema_for!(CompletionStreamResponse), &mut schemas)?;
    paths.insert(
        "/v1/completions".to_string(),
        post_endpoint(
            "createCompletion",
            "Create completion",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Embeddings ----
    use openai_protocol::embedding::*;
    let req_name = collect_schema(&schema_for!(EmbeddingRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(EmbeddingResponse), &mut schemas)?;
    paths.insert(
        "/v1/embeddings".to_string(),
        post_endpoint("createEmbedding", "Create embedding", &req_name, &resp_name),
    );

    // ---- Rerank ----
    use openai_protocol::rerank::*;
    let req_name = collect_schema(&schema_for!(RerankRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(RerankResponse), &mut schemas)?;
    paths.insert(
        "/v1/rerank".to_string(),
        post_endpoint("createRerank", "Rerank documents", &req_name, &resp_name),
    );

    // ---- Messages (Anthropic) ----
    use openai_protocol::messages::*;
    let req_name = collect_schema(&schema_for!(CreateMessageRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(Message), &mut schemas)?;
    collect_schema(&schema_for!(MessageStreamEvent), &mut schemas)?;
    paths.insert(
        "/v1/messages".to_string(),
        post_endpoint(
            "createMessage",
            "Create message (Anthropic)",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Responses API ----
    use openai_protocol::responses::*;
    let req_name = collect_schema(&schema_for!(ResponsesRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(ResponsesResponse), &mut schemas)?;
    paths.insert(
        "/v1/responses".to_string(),
        post_endpoint("createResponse", "Create response", &req_name, &resp_name),
    );

    // ---- Generate (SGLang native) ----
    use openai_protocol::generate::*;
    let req_name = collect_schema(&schema_for!(GenerateRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(GenerateResponse), &mut schemas)?;
    paths.insert(
        "/generate".to_string(),
        post_endpoint(
            "generate",
            "Generate (SGLang native)",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Tokenize / Detokenize ----
    use openai_protocol::tokenize::*;
    let req_name = collect_schema(&schema_for!(TokenizeRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(TokenizeResponse), &mut schemas)?;
    paths.insert(
        "/v1/tokenize".to_string(),
        post_endpoint("tokenize", "Tokenize text", &req_name, &resp_name),
    );

    let req_name = collect_schema(&schema_for!(DetokenizeRequest), &mut schemas)?;
    let resp_name = collect_schema(&schema_for!(DetokenizeResponse), &mut schemas)?;
    paths.insert(
        "/v1/detokenize".to_string(),
        post_endpoint("detokenize", "Detokenize tokens", &req_name, &resp_name),
    );

    // ---- Models ----
    let resp_name = collect_schema(&schema_for!(ListModelsResponse), &mut schemas)?;
    paths.insert(
        "/v1/models".to_string(),
        get_endpoint("listModels", "List available models", &resp_name),
    );

    // ---- Shared types (not tied to a specific endpoint) ----
    use openai_protocol::common::{Detail, ErrorDetail, ErrorResponse};
    collect_schema(&schema_for!(ErrorResponse), &mut schemas)?;
    collect_schema(&schema_for!(ErrorDetail), &mut schemas)?;
    collect_schema(&schema_for!(Detail), &mut schemas)?;

    // ---- Assemble OpenAPI document ----
    let doc = OpenApiDoc {
        openapi: "3.1.0".to_string(),
        info: Info {
            title: "SMG (Shepherd Model Gateway) API".to_string(),
            version: "1.2.0".to_string(),
            description: "OpenAI-compatible API with Anthropic Messages and SGLang native support"
                .to_string(),
        },
        paths,
        components: Components { schemas },
    };

    // Output path: first CLI arg or default
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "clients/openapi/smg-openapi.yaml".to_string());

    let yaml = serde_yaml::to_string(&doc)?;

    if output_path == "-" {
        write!(std::io::stdout(), "{yaml}")?;
    } else {
        std::fs::write(&output_path, &yaml)?;
        writeln!(std::io::stderr(), "Wrote OpenAPI spec to {output_path}")?;
    }

    Ok(())
}
