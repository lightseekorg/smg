---
title: Responses API
---

# Responses API Reference

The Responses API provides an OpenAI-compatible interface for agentic workflows with built-in support for multi-turn conversations, tool execution, and MCP (Model Context Protocol) integration.

---

## Overview

### Purpose vs Chat Completions API

The Responses API differs from the Chat Completions API in several key ways:

| Feature | Chat Completions | Responses API |
|---------|------------------|---------------|
| Conversation State | Stateless | Server-managed state |
| Tool Execution | Client-side | Server-side with MCP support |
| Multi-turn | Manual | Automatic with `previous_response_id` |
| Persistence | None | Built-in response/conversation storage |
| Agentic Workflows | Manual orchestration | Built-in tool loop execution |

### Agentic Workflow Concepts

The Responses API enables agentic workflows where the model can:

1. **Reason** about tasks using optional reasoning parameters
2. **Plan** tool usage with automatic tool selection
3. **Execute** tools via MCP servers or function calling
4. **Iterate** through multiple tool calls in a single request
5. **Persist** conversation history for multi-session workflows

---

## Base URL

```
http://localhost:30000/v1
```

---

## Create Response

Create a new response with optional tool execution and conversation management.

```
POST /v1/responses
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `input` | string or array | Yes | Input text or array of input items |
| `instructions` | string | No | System instructions for the model |
| `max_output_tokens` | integer | No | Maximum tokens to generate |
| `max_tool_calls` | integer | No | Maximum number of tool calls per request |
| `temperature` | number | No | Sampling temperature (0-2), default: 1.0 |
| `top_p` | number | No | Nucleus sampling parameter (0-1), default: 1.0 |
| `stream` | boolean | No | Enable streaming responses |
| `store` | boolean | No | Store response for later retrieval, default: true |
| `tools` | array | No | Available tools (function, mcp, web_search_preview, code_interpreter) |
| `tool_choice` | string/object | No | Tool selection behavior: `auto`, `none`, `required`, or specific tool |
| `parallel_tool_calls` | boolean | No | Allow parallel tool execution, default: true |
| `previous_response_id` | string | No | Continue from a previous response |
| `conversation` | string | No | Conversation ID (mutually exclusive with `previous_response_id`) |
| `reasoning` | object | No | Reasoning configuration |
| `text` | object | No | Text format for structured outputs |
| `metadata` | object | No | Custom metadata (max 16 properties) |
| `user` | string | No | End-user identifier |
| `background` | boolean | No | Run request in background (not with streaming) |

### Input Formats

**Simple text input:**

```json
{
  "input": "What is the capital of France?"
}
```

**Structured input items:**

```json
{
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    }
  ]
}
```

### Tool Configuration

**Function tools:**

```json
{
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      }
    }
  ]
}
```

**MCP tools:**

```json
{
  "tools": [
    {
      "type": "mcp",
      "server_url": "http://localhost:8080/mcp",
      "server_label": "my-mcp-server",
      "server_description": "My MCP server for data access",
      "require_approval": "never",
      "allowed_tools": ["query_database", "search_files"]
    }
  ]
}
```

### Reasoning Configuration

```json
{
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  }
}
```

Effort levels: `minimal`, `low`, `medium`, `high`

### Text Format (Structured Outputs)

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "user_info",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        }
      },
      "strict": true
    }
  }
}
```

### Example Request

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Search for the latest news about AI",
    "instructions": "Be concise and factual",
    "max_output_tokens": 500,
    "temperature": 0.7,
    "tools": [
      {
        "type": "mcp",
        "server_url": "http://localhost:8080/mcp",
        "server_label": "search"
      }
    ],
    "tool_choice": "auto"
  }'
```

### Response

```json
{
  "id": "resp_abc123def456",
  "object": "response",
  "created_at": 1705312345,
  "status": "completed",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "output": [
    {
      "type": "mcp_list_tools",
      "id": "mcp_list_001",
      "server_label": "search",
      "tools": [
        {
          "name": "web_search",
          "description": "Search the web",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
      ]
    },
    {
      "type": "mcp_call",
      "id": "mcp_call_001",
      "status": "completed",
      "name": "web_search",
      "arguments": "{\"query\": \"latest AI news\"}",
      "output": "{\"results\": [...]}",
      "server_label": "search"
    },
    {
      "type": "message",
      "id": "msg_001",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Based on my search, here are the latest AI developments..."
        }
      ],
      "status": "completed"
    }
  ],
  "usage": {
    "input_tokens": 50,
    "output_tokens": 150,
    "total_tokens": 200
  },
  "tools": [
    {
      "type": "mcp",
      "server_label": "search",
      "server_url": "http://localhost:8080/mcp"
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "store": true
}
```

### Streaming Response

With `"stream": true`, responses are sent as Server-Sent Events:

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Hello!",
    "stream": true
  }'
```

**Event sequence:**

```
event: response.created
data: {"type": "response.created", "sequence_number": 0, "response": {...}}

event: response.in_progress
data: {"type": "response.in_progress", "sequence_number": 1, "response": {...}}

event: response.output_item.added
data: {"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": {...}}

event: response.content_part.added
data: {"type": "response.content_part.added", "sequence_number": 3, "output_index": 0, "content_index": 0, "part": {...}}

event: response.output_text.delta
data: {"type": "response.output_text.delta", "sequence_number": 4, "output_index": 0, "content_index": 0, "delta": "Hello"}

event: response.output_text.done
data: {"type": "response.output_text.done", "sequence_number": 5, "output_index": 0, "content_index": 0, "text": "Hello! How can I help you?"}

event: response.output_item.done
data: {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": {...}}

event: response.completed
data: {"type": "response.completed", "sequence_number": 7, "response": {...}}

data: [DONE]
```

**MCP-specific streaming events:**

```
event: response.mcp_list_tools.in_progress
data: {"type": "response.mcp_list_tools.in_progress", "output_index": 0, "item_id": "mcp_list_001"}

event: response.mcp_list_tools.completed
data: {"type": "response.mcp_list_tools.completed", "output_index": 0, "item_id": "mcp_list_001"}

event: response.mcp_call.in_progress
data: {"type": "response.mcp_call.in_progress", "output_index": 1, "item_id": "mcp_call_001"}

event: response.mcp_call_arguments.delta
data: {"type": "response.mcp_call_arguments.delta", "output_index": 1, "item_id": "mcp_call_001", "delta": "{\"query\": \"..."}

event: response.mcp_call_arguments.done
data: {"type": "response.mcp_call_arguments.done", "output_index": 1, "item_id": "mcp_call_001", "arguments": "{\"query\": \"...\"}"}

event: response.output_item.done
data: {"type": "response.output_item.done", "output_index": 1, "item": {"type": "mcp_call", "output": "...", ...}}
```

---

## Get Response

Retrieve a previously stored response by ID.

```
GET /v1/responses/{response_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_id` | string | The response ID (e.g., `resp_abc123`) |

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | array | Additional fields to include |

### Example Request

```bash
curl http://localhost:30000/v1/responses/resp_abc123def456
```

### Response

Returns the full response object as shown in the Create Response section.

---

## Delete Response

Delete a stored response.

```
DELETE /v1/responses/{response_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_id` | string | The response ID to delete |

### Example Request

```bash
curl -X DELETE http://localhost:30000/v1/responses/resp_abc123def456
```

### Response

```json
{
  "id": "resp_abc123def456",
  "object": "response.deleted",
  "deleted": true
}
```

---

## List Response Input Items

List the input items that were sent with a response.

```
GET /v1/responses/{response_id}/input_items
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_id` | string | The response ID |

### Example Request

```bash
curl http://localhost:30000/v1/responses/resp_abc123def456/input_items
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "msg_input_001",
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    }
  ],
  "first_id": "msg_input_001",
  "last_id": "msg_input_001",
  "has_more": false
}
```

---

## Conversation Management

Conversations provide persistent storage for multi-turn interactions, enabling chat history to be maintained across multiple requests.

### Create Conversation

```
POST /v1/conversations
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metadata` | object | No | Custom metadata (max 16 properties) |

### Example Request

```bash
curl http://localhost:30000/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "project": "customer-support",
      "user_id": "user_123"
    }
  }'
```

### Response

```json
{
  "id": "conv_abc123def456",
  "object": "conversation",
  "created_at": 1705312345,
  "metadata": {
    "project": "customer-support",
    "user_id": "user_123"
  }
}
```

---

### Get Conversation

```
GET /v1/conversations/{conversation_id}
```

### Example Request

```bash
curl http://localhost:30000/v1/conversations/conv_abc123def456
```

### Response

```json
{
  "id": "conv_abc123def456",
  "object": "conversation",
  "created_at": 1705312345,
  "metadata": {
    "project": "customer-support"
  }
}
```

---

### Update Conversation

Update conversation metadata. Uses merge semantics - set a key to `null` to delete it.

```
POST /v1/conversations/{conversation_id}
```

### Request Body

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | object | Metadata to merge (null values delete keys) |

### Example Request

```bash
curl http://localhost:30000/v1/conversations/conv_abc123def456 \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "status": "resolved",
      "project": null
    }
  }'
```

### Response

Returns the updated conversation object.

---

### Delete Conversation

```
DELETE /v1/conversations/{conversation_id}
```

### Example Request

```bash
curl -X DELETE http://localhost:30000/v1/conversations/conv_abc123def456
```

### Response

```json
{
  "id": "conv_abc123def456",
  "object": "conversation.deleted",
  "deleted": true
}
```

---

### List Conversation Items

```
GET /v1/conversations/{conversation_id}/items
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Maximum items to return |
| `order` | string | `desc` | Sort order: `asc` or `desc` |
| `after` | string | - | Cursor for pagination |

### Example Request

```bash
curl "http://localhost:30000/v1/conversations/conv_abc123/items?limit=20&order=asc"
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "item_001",
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello"}],
      "status": "completed",
      "created_at": 1705312345
    },
    {
      "id": "item_002",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hi there!"}],
      "status": "completed",
      "created_at": 1705312346
    }
  ],
  "first_id": "item_001",
  "last_id": "item_002",
  "has_more": false
}
```

---

### Create Conversation Items

Add items to a conversation. Maximum 20 items per request.

```
POST /v1/conversations/{conversation_id}/items
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `items` | array | Yes | Array of items to add (max 20) |

### Supported Item Types

- `message` - User or assistant messages
- `reasoning` - Model reasoning content
- `mcp_list_tools` - MCP tool listing
- `mcp_call` - MCP tool invocation
- `item_reference` - Reference to an existing item
- `function_call` - Function tool call
- `function_call_output` - Function call result

### Example Request

```bash
curl http://localhost:30000/v1/conversations/conv_abc123/items \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What is 2+2?"}]
      },
      {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "2+2 equals 4."}]
      }
    ]
  }'
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "item_003",
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "What is 2+2?"}],
      "status": "completed"
    },
    {
      "id": "item_004",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "2+2 equals 4."}],
      "status": "completed"
    }
  ],
  "first_id": "item_003",
  "last_id": "item_004",
  "has_more": false
}
```

---

### Get Conversation Item

```
GET /v1/conversations/{conversation_id}/items/{item_id}
```

### Example Request

```bash
curl http://localhost:30000/v1/conversations/conv_abc123/items/item_001
```

### Response

Returns the item object.

---

### Delete Conversation Item

Remove an item from a conversation. This performs a soft delete - the item may still exist if referenced by other conversations.

```
DELETE /v1/conversations/{conversation_id}/items/{item_id}
```

### Example Request

```bash
curl -X DELETE http://localhost:30000/v1/conversations/conv_abc123/items/item_001
```

### Response

Returns the updated conversation object.

---

## Examples

### Simple Agentic Workflow

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="your-api-key"
)

# Create a response with MCP tools
response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Search for the weather in San Francisco and summarize it",
    tools=[
        {
            "type": "mcp",
            "server_url": "http://localhost:8080/mcp",
            "server_label": "weather-service"
        }
    ],
    tool_choice="auto"
)

# The response includes tool calls and final answer
for output in response.output:
    if output.type == "mcp_call":
        print(f"Tool called: {output.name}")
        print(f"Result: {output.output}")
    elif output.type == "message":
        for content in output.content:
            if content.type == "output_text":
                print(f"Answer: {content.text}")
```

### Multi-turn Conversation with Tools

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="your-api-key"
)

# Create a conversation
conversation = client.conversations.create(
    metadata={"session": "support-123"}
)

# First turn
response1 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="I need help with my order #12345",
    conversation=conversation.id,
    tools=[
        {
            "type": "mcp",
            "server_url": "http://localhost:8080/mcp",
            "server_label": "order-service"
        }
    ]
)
print(f"First response: {response1.id}")

# Second turn - continues the conversation
response2 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Can you also check if there are any discounts available?",
    conversation=conversation.id,
    tools=[
        {
            "type": "mcp",
            "server_url": "http://localhost:8080/mcp",
            "server_label": "order-service"
        }
    ]
)
print(f"Second response: {response2.id}")

# List conversation history
items = client.conversations.items.list(conversation.id)
for item in items.data:
    print(f"{item.role}: {item.content}")
```

### Streaming Response Handling

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="your-api-key"
)

# Stream a response
with client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Explain quantum computing",
    stream=True
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.mcp_call.in_progress":
            print(f"\n[Calling tool: {event.item_id}]")
        elif event.type == "response.completed":
            print(f"\n\nTokens used: {event.response.usage.total_tokens}")
```

### Using Previous Response ID

```python
# Alternative to conversations - chain responses directly
response1 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What are the main programming paradigms?",
    store=True
)

# Continue from previous response
response2 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Can you elaborate on functional programming?",
    previous_response_id=response1.id,
    store=True
)
```

---

## Error Responses

### Error Format

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "param": "field_name",
    "code": "error_code"
  }
}
```

### Common Errors

| HTTP Status | Type | Description |
|-------------|------|-------------|
| 400 | `invalid_request_error` | Malformed request or validation failure |
| 401 | `authentication_error` | Invalid or missing API key |
| 404 | `not_found_error` | Response, conversation, or item not found |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `internal_error` | Server error |
| 503 | `service_unavailable` | No healthy workers available |

### Validation Errors

```json
{
  "error": {
    "message": "Invalid 'conversation': 'invalid-id'. Expected an ID that begins with 'conv_'.",
    "type": "invalid_request_error",
    "param": "conversation",
    "code": "invalid_conversation_id"
  }
}
```

```json
{
  "error": {
    "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
    "type": "invalid_request_error",
    "code": "mutually_exclusive_parameters"
  }
}
```

---

## SGLang Extensions

The Responses API includes additional sampling parameters specific to SGLang:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | integer | -1 | Top-k sampling (-1 = disabled) |
| `min_p` | number | 0.0 | Min-p sampling threshold |
| `repetition_penalty` | number | 1.0 | Repetition penalty (1.0 = disabled) |
| `frequency_penalty` | number | - | OpenAI-compatible frequency penalty |
| `presence_penalty` | number | - | OpenAI-compatible presence penalty |
| `stop` | string/array | - | Stop sequences |

Example:

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "input": "Write a story",
  "top_k": 50,
  "min_p": 0.05,
  "repetition_penalty": 1.1
}
```
