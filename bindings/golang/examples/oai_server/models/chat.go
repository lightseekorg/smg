package models

// ChatRequest represents an OpenAI-compatible chat completion request
type ChatRequest struct {
	Model               string                   `json:"model" binding:"required"`
	Messages            []map[string]string      `json:"messages" binding:"required"`
	Stream              bool                     `json:"stream,omitempty"`
	StreamOptions       *StreamOptions           `json:"stream_options,omitempty"`
	Temperature         *float64                 `json:"temperature,omitempty"`
	TopP                *float64                 `json:"top_p,omitempty"`
	MaxTokens           *int                     `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`
	Tools               []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice          interface{}              `json:"tool_choice,omitempty"`
	IgnoreEos           bool                     `json:"ignore_eos,omitempty"`
	NoStopTrim          bool                     `json:"no_stop_trim,omitempty"`
	StopTokenIDs        []int                    `json:"stop_token_ids,omitempty"`
	Stop                interface{}              `json:"stop,omitempty"`
	FrequencyPenalty    *float64                 `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64                 `json:"presence_penalty,omitempty"`
	TopK                *int                     `json:"top_k,omitempty"`
	MinP                *float64                 `json:"min_p,omitempty"`
	RepetitionPenalty   *float64                 `json:"repetition_penalty,omitempty"`
}

// StreamOptions represents streaming options (e.g., include_usage)
type StreamOptions struct {
	IncludeUsage *bool `json:"include_usage,omitempty"`
}
