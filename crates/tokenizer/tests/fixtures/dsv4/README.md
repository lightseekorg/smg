# DeepSeek-V4 encoding golden fixtures

Reference input/output vectors copied verbatim from the DeepSeek-V4 model
distribution (`encoding/tests/` in `deepseek-ai/DeepSeek-V4-Flash`, identical in
`-Pro`). `encoding_dsv4.py` produces each `test_output_N.txt` from the matching
`test_input_N.json`; vLLM serves from that same encoder.

`dsv4_encoding_parity.rs` renders these through SMG's Rust port and asserts
byte-for-byte equality, guarding against prompt drift between the SMG and vLLM
serving paths (e.g. JSON separator differences in embedded tool schemas).
