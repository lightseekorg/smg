"""OpenAI Completions API E2E tests.

Tests for the OpenAI Completions API endpoints (/v1/completions) including:
- Basic non-streaming and streaming text completion
- Stop sequences, echo, and suffix handling
- Parallel sampling (n > 1)
- Usage statistics validation
"""
