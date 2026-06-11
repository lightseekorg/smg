//! Runtime configuration for the mock worker fleet, parsed from CLI flags.

use std::time::Duration;

/// Configuration shared by every mocked HTTP and gRPC worker in the process.
#[derive(Debug, Clone)]
pub struct Config {
    /// Bind address for every listener.
    pub host: String,
    /// First HTTP port; `http_count` workers bind `[http_base_port, +count)`.
    pub http_base_port: u16,
    /// Number of HTTP workers to start.
    pub http_count: u16,
    /// First gRPC port; `grpc_count` workers bind `[grpc_base_port, +count)`.
    pub grpc_base_port: u16,
    /// Number of gRPC workers to start.
    pub grpc_count: u16,
    /// Model id advertised by every worker (one model, many replicas).
    pub model_id: String,
    /// Tokenizer path advertised by gRPC workers (for gateway autoload).
    pub tokenizer_path: String,
    /// Simulated per-request generation latency.
    pub gen_delay: Duration,
    /// Number of canned output tokens per generation.
    pub output_tokens: u32,
}

impl Config {
    /// Parse the configuration from `std::env::args`, falling back to defaults.
    pub fn from_args() -> Result<Self, String> {
        let mut cfg = Self {
            host: "127.0.0.1".to_string(),
            http_base_port: 9000,
            http_count: 0,
            grpc_base_port: 0,
            grpc_count: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: String::new(),
            gen_delay: Duration::from_millis(0),
            output_tokens: 8,
        };

        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--host" => cfg.host = value(&mut args, &flag)?,
                "--http-base-port" => cfg.http_base_port = parse(value(&mut args, &flag)?, &flag)?,
                "--http-count" => cfg.http_count = parse(value(&mut args, &flag)?, &flag)?,
                "--grpc-base-port" => cfg.grpc_base_port = parse(value(&mut args, &flag)?, &flag)?,
                "--grpc-count" => cfg.grpc_count = parse(value(&mut args, &flag)?, &flag)?,
                "--model" => cfg.model_id = value(&mut args, &flag)?,
                "--tokenizer" => cfg.tokenizer_path = value(&mut args, &flag)?,
                "--gen-ms" => {
                    cfg.gen_delay = Duration::from_millis(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--output-tokens" => cfg.output_tokens = parse(value(&mut args, &flag)?, &flag)?,
                "-h" | "--help" => return Err(usage()),
                other => return Err(format!("unknown flag: {other}\n\n{}", usage())),
            }
        }

        if cfg.tokenizer_path.is_empty() {
            cfg.tokenizer_path = cfg.model_id.clone();
        }
        if cfg.http_count == 0 && cfg.grpc_count == 0 {
            return Err(format!(
                "nothing to do: pass --http-count and/or --grpc-count\n\n{}",
                usage()
            ));
        }
        if cfg.grpc_count > 0 && cfg.grpc_base_port == 0 {
            return Err("--grpc-base-port is required when --grpc-count > 0".to_string());
        }
        Ok(cfg)
    }
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse<T: std::str::FromStr>(raw: String, flag: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn usage() -> String {
    "mock-worker — multi-port mock HTTP/gRPC inference workers for SMG scale testing\n\n\
     Flags:\n\
       --host <addr>            bind address (default 127.0.0.1)\n\
       --http-base-port <port>  first HTTP port (default 9000)\n\
       --http-count <n>         number of HTTP workers (default 0)\n\
       --grpc-base-port <port>  first gRPC port (required if --grpc-count > 0)\n\
       --grpc-count <n>         number of gRPC workers (default 0)\n\
       --model <id>             advertised model id (default mock-model)\n\
       --tokenizer <path>       tokenizer path for gRPC autoload (default = model)\n\
       --gen-ms <ms>            simulated generation latency (default 0)\n\
       --output-tokens <n>      canned output tokens per request (default 8)"
        .to_string()
}
