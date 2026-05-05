//! Kimi-K2.5 tool-declaration encoder. See module-level docs.
//!
//! Mirrors `tool_declaration_ts.py` from the Kimi-K2.5 model snapshot
//! function-by-function. Output must be byte-equal to the Python reference;
//! gated by golden tests in `tests/kimi_k25_tools_encoder.rs`.

#![allow(
    unused_imports,
    dead_code,
    clippy::todo,
    clippy::disallowed_macros,
    clippy::unwrap_used
)]

use std::{collections::HashMap, fmt::Write};

use anyhow::Result;
use serde_json::Value;

use crate::chat_template::{ChatTemplateParams, ChatTemplateState};

const TS_INDENT: &str = "  ";
const TS_FIELD_DELIMITER: &str = ",\n";

pub fn encode_tools_to_typescript(tools: &[Value]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let function_strs: Vec<String> = tools
        .iter()
        .filter_map(|t| {
            if t.get("type").and_then(Value::as_str) != Some("function") {
                return None;
            }
            let func = t.get("function")?;
            Some(openai_function_to_typescript(func))
        })
        .collect();
    if function_strs.is_empty() {
        return None;
    }
    let mut out = String::from("# Tools\n\n## functions\nnamespace functions {\n");
    out.push_str(&function_strs.join("\n"));
    out.push('\n');
    out.push_str("}\n");
    Some(out)
}

/// Renderer for `Renderer::KimiK25Tools`. Computes `tools_ts_str` and merges
/// it into `template_kwargs`, then delegates to the standard minijinja path.
pub fn apply_kimi_k25_tools(
    chat_template: &ChatTemplateState,
    messages: &[Value],
    params: ChatTemplateParams,
) -> Result<String> {
    let _ = (chat_template, messages, params);
    todo!("apply_kimi_k25_tools — implemented in Task 7")
}

// ---------------------------------------------------------------------------
// Function-level encoding
// ---------------------------------------------------------------------------

fn openai_function_to_typescript(function: &Value) -> String {
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let mut registry = SchemaRegistry::default();
    let parsed = ParameterTypeObject::parse(&parameters, &mut registry);

    let mut interfaces: Vec<String> = Vec::new();
    let mut root_interface_name: Option<&str> = None;

    if registry.has_self_ref {
        root_interface_name = Some("parameters");
        let body = parsed
            .properties
            .iter()
            .map(|p| p.to_typescript(TS_INDENT, &registry))
            .collect::<Vec<_>>()
            .join(TS_FIELD_DELIMITER);
        let body = if body.is_empty() {
            String::new()
        } else {
            format!("\n{body}\n")
        };
        interfaces.push(format!("interface parameters {{{body}}}"));
    }

    let defs_clone: Vec<(String, Value)> = registry
        .definitions
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (name, schema) in defs_clone {
        let obj_type = parse_parameter_type(&schema, &mut registry);
        let body = obj_type.to_typescript("", &registry);
        let mut def_str = String::new();
        if let Some(desc) = schema.get("description").and_then(Value::as_str) {
            if !desc.is_empty() {
                def_str.push_str(&format_description(desc, ""));
                def_str.push('\n');
            }
        }
        write!(def_str, "interface {name} {body}").unwrap();
        interfaces.push(def_str);
    }

    let interface_str = interfaces.join("\n");
    let function_name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("function");
    let type_def = match root_interface_name {
        Some(n) => format!("type {function_name} = (_: {n}) => any;"),
        None => format!(
            "type {function_name} = (_: {}) => any;",
            parsed.to_typescript("", &registry)
        ),
    };
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    let desc_block = if description.is_empty() {
        String::new()
    } else {
        format_description(description, "")
    };

    [interface_str, desc_block, type_def]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Schema parsing
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SchemaRegistry {
    definitions: HashMap<String, Value>,
    has_self_ref: bool,
}

impl SchemaRegistry {
    fn register_definitions(&mut self, defs: &Value) {
        if let Some(map) = defs.as_object() {
            for (name, schema) in map {
                self.definitions.insert(name.clone(), schema.clone());
            }
        }
    }

    fn resolve_ref(&mut self, reference: &str) -> Option<Value> {
        if reference == "#" {
            self.has_self_ref = true;
            return Some(serde_json::json!({"$self_ref": true}));
        }
        if let Some(name) = reference.strip_prefix("#/$defs/") {
            return self.definitions.get(name).cloned();
        }
        None
    }
}

enum ParameterType {
    Scalar(ParameterTypeScalar),
    Object(ParameterTypeObject),
    Array(ParameterTypeArray),
}

impl ParameterType {
    fn format_docstring(&self, indent: &str) -> String {
        match self {
            ParameterType::Scalar(s) => s.base.format_docstring(indent),
            ParameterType::Object(o) => o.base.format_docstring(indent),
            ParameterType::Array(a) => a.base.format_docstring(indent),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        match self {
            ParameterType::Scalar(s) => s.to_typescript(),
            ParameterType::Object(o) => o.to_typescript(indent, registry),
            ParameterType::Array(a) => a.to_typescript(indent, registry),
        }
    }
}

#[derive(Default)]
struct BaseType {
    description: String,
    constraints: Vec<(String, Value)>,
}

impl BaseType {
    fn from_extra_props(props: &Value, allowed_keys: &[&str]) -> Self {
        let description = props
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut constraints: Vec<(String, Value)> = props
            .as_object()
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| allowed_keys.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();
        constraints.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            description,
            constraints,
        }
    }

    fn format_docstring(&self, indent: &str) -> String {
        let mut out = String::new();
        if !self.description.is_empty() {
            out.push_str(&format_description(&self.description, indent));
            out.push('\n');
        }
        if !self.constraints.is_empty() {
            let parts: Vec<String> = self
                .constraints
                .iter()
                .map(|(k, v)| format!("{k}: {}", json_inline(v)))
                .collect();
            writeln!(out, "{indent}// {}", parts.join(", ")).unwrap();
        }
        out
    }
}

struct ParameterTypeScalar {
    base: BaseType,
    typ: String,
}

impl ParameterTypeScalar {
    fn parse(typ: &str, props: &Value) -> Self {
        let allowed: &[&str] = match typ {
            "string" => &["maxLength", "minLength", "pattern"],
            "number" | "integer" => &["maximum", "minimum"],
            _ => &[],
        };
        Self {
            base: BaseType::from_extra_props(props, allowed),
            typ: typ.to_string(),
        }
    }

    fn any() -> Self {
        Self {
            base: BaseType::default(),
            typ: "any".to_string(),
        }
    }

    fn to_typescript(&self) -> String {
        match self.typ.as_str() {
            "integer" => "number".to_string(),
            other => other.to_string(),
        }
    }
}

struct Parameter {
    name: String,
    typ: ParameterType,
    optional: bool,
    default: Option<Value>,
}

impl Parameter {
    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut out = self.typ.format_docstring(indent);
        if let Some(d) = &self.default {
            let repr = match d {
                Value::Bool(_) | Value::Number(_) => d.to_string(),
                _ => serde_json::to_string(d).unwrap_or_else(|_| "null".to_string()),
            };
            writeln!(out, "{indent}// Default: {repr}").unwrap();
        }
        let opt_marker = if self.optional { "?" } else { "" };
        write!(
            out,
            "{indent}{}{opt_marker}: {}",
            self.name,
            self.typ.to_typescript(indent, registry)
        )
        .unwrap();
        out
    }
}

struct ParameterTypeObject {
    base: BaseType,
    properties: Vec<Parameter>,
    additional_properties: AdditionalProperties,
}

enum AdditionalProperties {
    None,
    True,
    False,
    Schema(Box<ParameterType>),
}

impl ParameterTypeObject {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let base = BaseType::from_extra_props(schema, &[]);
        if let Some(defs) = schema.get("$defs") {
            registry.register_definitions(defs);
        }

        let additional_properties = match schema.get("additionalProperties") {
            None => AdditionalProperties::None,
            Some(Value::Bool(true)) => AdditionalProperties::True,
            Some(Value::Bool(false)) => AdditionalProperties::False,
            Some(other) => {
                AdditionalProperties::Schema(Box::new(parse_parameter_type(other, registry)))
            }
        };

        let props_map = schema.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        let properties: Vec<Parameter> = props_map
            .map(|props| {
                props
                    .iter()
                    .map(|(name, prop_schema)| {
                        let optional = !required.contains(&name.as_str());
                        let default = prop_schema.get("default").cloned();
                        let typ = parse_parameter_type(prop_schema, registry);
                        Parameter {
                            name: name.clone(),
                            typ,
                            optional,
                            default,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            base,
            properties,
            additional_properties,
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut required: Vec<&Parameter> =
            self.properties.iter().filter(|p| !p.optional).collect();
        let mut optional: Vec<&Parameter> = self.properties.iter().filter(|p| p.optional).collect();
        required.sort_by(|a, b| a.name.cmp(&b.name));
        optional.sort_by(|a, b| a.name.cmp(&b.name));
        let inner_indent = format!("{indent}{TS_INDENT}");
        let mut parts: Vec<String> = required
            .into_iter()
            .chain(optional)
            .map(|p| p.to_typescript(&inner_indent, registry))
            .collect();

        match &self.additional_properties {
            AdditionalProperties::None => {}
            AdditionalProperties::True => parts.push(format!("{inner_indent}[k: string]: any")),
            AdditionalProperties::False => parts.push(format!("{inner_indent}[k: string]: never")),
            AdditionalProperties::Schema(inner) => {
                let ty = inner.to_typescript(&inner_indent, registry);
                parts.push(format!("{inner_indent}[k: string]: {ty}"));
            }
        }

        if parts.is_empty() {
            return "{}".to_string();
        }
        let body = parts.join(TS_FIELD_DELIMITER);
        let body = format!("\n{body}\n");
        format!("{{{body}{indent}}}")
    }
}

struct ParameterTypeArray {
    base: BaseType,
    item: Box<ParameterType>,
}

impl ParameterTypeArray {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let base = BaseType::from_extra_props(schema, &["minItems", "maxItems"]);
        let item = match schema.get("items") {
            Some(items) if !items.is_null() => parse_parameter_type(items, registry),
            _ => ParameterType::Scalar(ParameterTypeScalar::any()),
        };
        Self {
            base,
            item: Box::new(item),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let inner_indent = format!("{indent}{TS_INDENT}");
        let item_doc = self.item.format_docstring(&inner_indent);
        let item_ts = self.item.to_typescript(indent, registry);
        if item_doc.is_empty() {
            format!("Array<{item_ts}>")
        } else {
            format!("Array<\n{item_doc}{inner_indent}{item_ts}\n{indent}>")
        }
    }
}

fn parse_parameter_type(schema: &Value, registry: &mut SchemaRegistry) -> ParameterType {
    if schema.is_boolean() {
        return ParameterType::Scalar(ParameterTypeScalar {
            base: BaseType::default(),
            typ: if schema.as_bool() == Some(true) {
                "any"
            } else {
                "null"
            }
            .to_string(),
        });
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return ParameterType::Scalar(ParameterTypeScalar::any()),
    };

    if let Some(typ) = obj.get("type").and_then(Value::as_str) {
        return match typ {
            "object" => ParameterType::Object(ParameterTypeObject::parse(schema, registry)),
            "array" => ParameterType::Array(ParameterTypeArray::parse(schema, registry)),
            other => ParameterType::Scalar(ParameterTypeScalar::parse(other, schema)),
        };
    }
    if obj.is_empty() {
        return ParameterType::Scalar(ParameterTypeScalar::any());
    }
    // Fall-through for shapes covered in later tasks (anyOf, enum, type-list,
    // $ref). For now, degrade to `any` to keep the encoder permissive.
    ParameterType::Scalar(ParameterTypeScalar::any())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_description(description: &str, indent: &str) -> String {
    description
        .split('\n')
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{indent}// {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn json_inline(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}
