package config

import (
	"os"
)

// Config holds the application configuration
type Config struct {
	// Endpoints is a comma-separated list of gRPC endpoint URLs for multi-worker support
	// (e.g., "grpc://host1:20000,grpc://host2:20001")
	Endpoints     string
	TokenizerPath string
	Port          string
	LogDir        string
	LogLevel      string
	// PolicyName is the load balancing policy to use ("round_robin", "random", "cache_aware")
	// Defaults to "round_robin" if not specified
	PolicyName string
}

// Load loads configuration from environment variables with defaults
func Load() *Config {
	// Get tokenizer path from environment or use default
	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "../tokenizer"
	}

	// Get endpoints from environment or use default
	// Supports comma-separated list for multi-worker setups
	endpoints := os.Getenv("SGL_GRPC_ENDPOINTS")
	if endpoints == "" {
		// Fall back to legacy single endpoint
		endpoints = os.Getenv("SGL_GRPC_ENDPOINT")
		if endpoints == "" {
			endpoints = "grpc://localhost:20000"
		}
	}

	// Get load balancing policy from environment or use default
	policyName := os.Getenv("SGL_POLICY_NAME")
	if policyName == "" {
		policyName = "round_robin"
	}

	// Get port from environment or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Get log directory from environment or use default
	logDir := os.Getenv("LOG_DIR")
	if logDir == "" {
		logDir = "./logs"
	}

	// Get log level from environment or use default
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "info"
	}

	return &Config{
		Endpoints:     endpoints,
		TokenizerPath: tokenizerPath,
		Port:          port,
		LogDir:        logDir,
		LogLevel:      logLevel,
		PolicyName:    policyName,
	}
}
