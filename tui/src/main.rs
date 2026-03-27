use std::{
    io,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use clap::Parser;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use smg_tui::{
    app::App,
    client::SmgClient,
    state::{spawn_poller, GatewayState},
};

#[derive(Parser)]
#[command(
    name = "smg-tui",
    about = "Terminal dashboard for Shepherd Model Gateway"
)]
struct Cli {
    /// SMG gateway base URL.
    #[arg(long, default_value = "http://localhost:30000")]
    gateway_url: String,

    /// Prometheus / metrics endpoint URL.
    #[arg(long, default_value = "http://localhost:29000")]
    metrics_url: String,

    /// Polling interval in seconds.
    #[arg(long, default_value_t = 3)]
    poll_interval: u64,

    /// API key for authenticated endpoints.
    /// Reads from: --api-key flag, SMG_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY env vars.
    #[arg(long, env = "SMG_API_KEY")]
    api_key: Option<String>,

    /// Automatically start the SMG gateway if not reachable.
    #[arg(long, default_value_t = false)]
    auto_start: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Tracing (file/stderr only — stdout is the TUI)
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter("smg_tui=info")
        .init();

    // Resolve API key: --api-key > SMG_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

    let client = SmgClient::new(cli.gateway_url.clone(), cli.metrics_url.clone(), api_key);

    // Auto-start gateway if requested and not reachable
    let _gateway_child = if cli.auto_start {
        match client.check_alive().await {
            Ok(()) => {
                tracing::info!("Gateway already running at {}", cli.gateway_url);
                None
            }
            Err(_) => {
                tracing::info!(
                    "Gateway not reachable at {}, starting automatically...",
                    cli.gateway_url
                );
                let port = extract_port(&cli.gateway_url).unwrap_or(30000);
                let metrics_port = extract_port(&cli.metrics_url).unwrap_or(29000);
                let launch_args = [
                    "launch",
                    "--port",
                    &port.to_string(),
                    "--prometheus-port",
                    &metrics_port.to_string(),
                    "--enable-igw",
                    "--policy",
                    "round_robin",
                ];
                tracing::info!("Running: smg {}", launch_args.join(" "));
                let log_file = std::fs::File::create("/tmp/smg-gateway.log")
                    .or_else(|_| std::fs::File::create("/dev/null"))?;
                let log_file2 = log_file
                    .try_clone()
                    .or_else(|_| std::fs::File::create("/dev/null"))?;
                tracing::info!("Gateway logs: /tmp/smg-gateway.log");
                let child = tokio::process::Command::new("smg")
                    .args(launch_args)
                    .stdout(std::process::Stdio::from(log_file))
                    .stderr(std::process::Stdio::from(log_file2))
                    .spawn();
                match child {
                    Ok(mut child) => {
                        tracing::info!(
                            "Gateway process started (pid {}), waiting for readiness...",
                            child.id().unwrap_or(0)
                        );
                        let deadline =
                            tokio::time::Instant::now() + tokio::time::Duration::from_secs(120);
                        loop {
                            if tokio::time::Instant::now() >= deadline {
                                tracing::warn!(
                                    "Gateway did not become ready within 120s, continuing anyway"
                                );
                                break;
                            }
                            // Check if the process exited early (crash/bad args)
                            match child.try_wait() {
                                Ok(Some(status)) => {
                                    tracing::error!("Gateway process exited with: {status}");
                                    tracing::error!("Gateway exited with {status}. Check that 'smg' is the Rust binary (not Python).");
                                    tracing::error!(
                                        "Install with: cargo install --path model_gateway"
                                    );
                                    return Err(anyhow::anyhow!("Gateway exited: {status}"));
                                }
                                Ok(None) => {} // still running
                                Err(e) => {
                                    tracing::error!("Failed to check gateway status: {e}");
                                }
                            }
                            if client.check_alive().await.is_ok() {
                                tracing::info!("Gateway is up and accepting connections");
                                break;
                            }
                            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        }
                        Some(child)
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to start gateway: {e}. \
                             Make sure 'smg' is in your PATH (cargo install --path model_gateway)."
                        );
                        return Err(e.into());
                    }
                }
            }
        }
    } else {
        None
    };

    let state = Arc::new(RwLock::new(GatewayState::default()));

    let _poller = spawn_poller(client.clone(), Arc::clone(&state), cli.poll_interval);

    // Terminal setup — use RAII guard so the terminal is restored on error or panic
    enable_raw_mode()?;

    // Install panic hook that restores the terminal
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        panic_hook(info);
    }));

    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        return Err(e.into());
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            return Err(e.into());
        }
    };

    // Run
    let mut app = App::new(state, client);
    let result = app.run(&mut terminal).await;

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Full shutdown (Ctrl+C×2): kill all spawned workers and auto-started gateway
    // Quit (q): leave everything running in the background
    if app.full_shutdown {
        tracing::info!("Full shutdown: stopping all services...");
        for (desc, mut child) in app.worker_children.drain(..) {
            tracing::info!("Stopping worker: {desc}");
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        if let Some(mut child) = _gateway_child {
            tracing::info!("Stopping auto-started gateway...");
            let _ = child.kill().await;
            let _ = child.wait().await;
        } else {
            // Gateway wasn't auto-started by this session — kill by port
            let port = extract_port(&cli.gateway_url).unwrap_or(30000);
            tracing::info!("Stopping gateway on port {port}...");
            kill_process_on_port(port).await;
        }
        tracing::info!("All services stopped.");
    }

    result
}

/// Kill the process listening on the given TCP port using lsof + kill.
async fn kill_process_on_port(port: u16) {
    let output = tokio::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{port}")])
        .output()
        .await;
    match output {
        Ok(out) if out.status.success() => {
            let pids = String::from_utf8_lossy(&out.stdout);
            for pid_str in pids.split_whitespace() {
                tracing::info!("Killing process {pid_str} on port {port}");
                let _ = tokio::process::Command::new("kill")
                    .args(["-TERM", pid_str.trim()])
                    .output()
                    .await;
            }
        }
        _ => {
            tracing::warn!("Could not find process on port {port}");
        }
    }
}

/// Extract port from a URL like "http://localhost:30000" or "http://localhost:30000/health".
fn extract_port(url: &str) -> Option<u16> {
    url.rsplit(':').next()?.split('/').next()?.parse().ok()
}
