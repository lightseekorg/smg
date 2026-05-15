use std::collections::VecDeque;

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use openai_protocol::worker::{ProviderType, RuntimeType, WorkerSpec};
use tokio::sync::mpsc;

use crate::{
    chat::{ChatEndpoint, ChatMessage},
    client::SmgClient,
    event::{AppEvent, EventHandler},
    state::SharedState,
    types::{ActionMenuItem, AddMenuState, InputMode, ProviderPreset, View},
    ui,
};

/// Top-level application state.
pub struct App {
    pub view: View,
    pub input_mode: InputMode,
    pub input_buffer: String,
    pub active_filter: Option<String>,
    pub should_quit: bool,
    /// True when user wants full shutdown (kill workers + gateway). False = quit TUI only.
    pub full_shutdown: bool,
    /// Tracks first Ctrl+C press for double-Ctrl+C full shutdown.
    ctrl_c_at: Option<std::time::Instant>,
    pub selected_index: usize,
    pub state: SharedState,
    pub client: SmgClient,
    pub status_message: Option<String>,
    pub show_help: bool,
    /// (worker_id, worker_url) pending confirmation
    pub confirm_delete: Option<(String, String)>,
    pub show_detail: bool,
    pub show_action_menu: bool,
    pub action_menu_index: usize,
    pub add_menu_state: Option<AddMenuState>,
    pub confirm_flush: Option<(String, String)>,

    // Playground state
    pub chat_messages: Vec<ChatMessage>,
    pub chat_input: String,
    pub chat_model: String,
    pub chat_streaming: bool,
    pub chat_scroll: u16,
    pub chat_endpoint: ChatEndpoint,
    pub chat_previous_response_id: Option<String>,
    chat_stream_rx: Option<mpsc::UnboundedReceiver<String>>,

    // Logs
    pub log_entries: VecDeque<LogEntry>,
    pub log_scroll: u16,
    pub log_auto_scroll: bool,
    pub log_sub_tab: LogSubTab,

    // Spawned local worker processes
    pub worker_children: Vec<(String, tokio::process::Child)>, // (description, child)
    // GPUs claimed by spawned workers, keyed by worker URL (to avoid double-allocation)
    claimed_gpus: std::collections::HashMap<String, Vec<u32>>,

    status_clear_at: Option<std::time::Instant>,
}

const MAX_LOG_ENTRIES: usize = 1000;

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogSubTab {
    Tui,
    Gateway,
    Worker(String), // port number as string
}

impl LogSubTab {
    pub fn label(&self) -> String {
        match self {
            Self::Tui => "a:TUI".to_string(),
            Self::Gateway => "b:SMG".to_string(),
            Self::Worker(port) => format!("w:{port}"),
        }
    }
}

impl App {
    pub fn new(state: SharedState, client: SmgClient) -> Self {
        Self {
            view: View::Pulse,
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            active_filter: None,
            should_quit: false,
            full_shutdown: false,
            ctrl_c_at: None,
            selected_index: 0,
            state,
            client,
            status_message: None,
            show_help: false,
            confirm_delete: None,
            show_detail: false,
            show_action_menu: false,
            action_menu_index: 0,
            add_menu_state: None,
            confirm_flush: None,
            chat_messages: Vec::new(),
            chat_input: String::new(),
            chat_model: String::new(),
            chat_streaming: false,
            chat_scroll: 0,
            chat_endpoint: ChatEndpoint::default(),
            chat_previous_response_id: None,
            chat_stream_rx: None,
            log_entries: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            log_scroll: u16::MAX,
            log_auto_scroll: true,
            log_sub_tab: LogSubTab::Tui,
            worker_children: Vec::new(),
            claimed_gpus: std::collections::HashMap::new(),
            status_clear_at: None,
        }
    }

    pub async fn run(
        &mut self,
        terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    ) -> Result<()> {
        let mut events = EventHandler::new(250);

        loop {
            terminal.draw(|f| ui::render(f, self))?;

            match events.next().await {
                Some(AppEvent::Key(key)) => self.handle_key(key).await,
                Some(AppEvent::Tick) => self.on_tick(),
                Some(AppEvent::Resize(_, _)) => {} // ratatui handles resize on next draw
                None => break,
            }

            if self.should_quit {
                break;
            }
        }

        // Only kill spawned workers on full shutdown (Ctrl+C×2).
        // Note: main() also handles this after terminal cleanup, but we start
        // killing here so processes begin dying while the terminal restores.
        if self.full_shutdown {
            for (desc, child) in &mut self.worker_children {
                if let Err(e) = child.start_kill() {
                    tracing::warn!("Failed to kill worker {desc}: {e}");
                }
            }
        }

        Ok(())
    }

    fn on_tick(&mut self) {
        // Auto-clear status message after 5 seconds
        if let Some(deadline) = self.status_clear_at {
            if std::time::Instant::now() >= deadline {
                self.status_message = None;
                self.status_clear_at = None;
            }
        }

        // Log poller errors
        {
            let err_msg = {
                // Safety: RwLock is not poisoned — no panics while holding the lock
                #[expect(clippy::unwrap_used)]
                let state = self.state.read().unwrap();
                state.last_error.clone()
            };
            if let Some(err) = err_msg {
                let already_logged = self
                    .log_entries
                    .back()
                    .map(|e| e.message.contains(&err))
                    .unwrap_or(false);
                if !already_logged {
                    self.add_log(LogLevel::Warn, &format!("Gateway: {err}"));
                }
            }
        }

        // Drain streaming tokens from playground
        if let Some(ref mut rx) = self.chat_stream_rx {
            let mut got_data = false;
            loop {
                match rx.try_recv() {
                    Ok(token) => {
                        got_data = true;
                        if token == "\n[DONE]" {
                            self.chat_streaming = false;
                            self.chat_stream_rx = None;
                            break;
                        } else if token.starts_with("\n[RESPONSE_ID]") {
                            let id = token.trim_start_matches("\n[RESPONSE_ID]").to_string();
                            self.chat_previous_response_id = Some(id);
                            continue;
                        } else if token.starts_with("\n[ERROR]") {
                            let err = token.trim_start_matches("\n[ERROR]").to_string();
                            if let Some(msg) = self.chat_messages.last_mut() {
                                msg.content.push_str(&format!("\n[Error: {err}]"));
                            }
                            self.chat_streaming = false;
                            self.chat_stream_rx = None;
                            break;
                        } else if let Some(msg) = self.chat_messages.last_mut() {
                            msg.content.push_str(&token);
                        }
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        self.chat_streaming = false;
                        self.chat_stream_rx = None;
                        break;
                    }
                }
            }
            if got_data {
                // Auto-scroll to bottom
                self.chat_scroll = u16::MAX;
            }
        }
    }

    async fn handle_key(&mut self, key: KeyEvent) {
        // Ctrl+C: double-press for full shutdown, single press warns
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if let Some(first) = self.ctrl_c_at {
                if first.elapsed() < std::time::Duration::from_secs(3) {
                    // Second Ctrl+C within 3s → full shutdown
                    self.should_quit = true;
                    self.full_shutdown = true;
                    return;
                }
            }
            // First (or expired) Ctrl+C → warn and record time
            self.ctrl_c_at = Some(std::time::Instant::now());
            self.set_status(
                "Press Ctrl+C again to stop all services, or q to quit TUI only".to_string(),
            );
            return;
        }

        // Delete confirmation dialog takes priority
        if self.confirm_delete.is_some() {
            self.handle_delete_confirm(key).await;
            return;
        }

        if self.show_action_menu {
            self.handle_action_menu_key(key).await;
            return;
        }
        if self.add_menu_state.is_some() {
            self.handle_add_menu_key(key).await;
            return;
        }
        if let Some((ref id, ref _url)) = self.confirm_flush.clone() {
            match key.code {
                KeyCode::Char('y') => {
                    let id = id.clone();
                    self.confirm_flush = None;
                    match self.client.flush_worker_cache(&id).await {
                        Ok(_) => self.set_status("Cache flushed".to_string()),
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                }
                _ => {
                    self.confirm_flush = None;
                }
            }
            return;
        }

        // Playground has its own input handling
        if self.view == View::Chat && self.input_mode == InputMode::Normal {
            self.handle_chat_key(key);
            return;
        }

        match self.input_mode {
            InputMode::Normal => self.handle_normal(key),
            InputMode::Filter => self.handle_filter_input(key),
            InputMode::Command => self.handle_command_input(key).await,
        }
    }

    fn handle_normal(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,

            // View switching
            code @ (KeyCode::Char('1')
            | KeyCode::Char('2')
            | KeyCode::Char('3')
            | KeyCode::Char('4')
            | KeyCode::Char('5')
            | KeyCode::Char('6')
            | KeyCode::Char('7')
            | KeyCode::Char('8')) => {
                if let Some(v) = View::from_key(code) {
                    self.view = v;
                    self.selected_index = 0;
                }
            }

            // Navigation
            KeyCode::Char('j') | KeyCode::Down => {
                if self.view == View::Logs {
                    // If at auto-scroll bottom, don't move further down
                    if self.log_auto_scroll {
                        return;
                    }
                    self.log_scroll = self.log_scroll.saturating_add(1);
                } else {
                    self.selected_index = self.selected_index.saturating_add(1);
                    self.clamp_selection();
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.view == View::Logs {
                    self.log_scroll = self.log_scroll.saturating_sub(1);
                    self.log_auto_scroll = false;
                } else {
                    self.selected_index = self.selected_index.saturating_sub(1);
                }
            }
            // Logs: G to jump to bottom
            KeyCode::Char('G') if self.view == View::Logs => {
                self.log_scroll = u16::MAX;
                self.log_auto_scroll = true;
            }
            // Logs: a/b/c to switch sub-tabs
            KeyCode::Char('a') if self.view == View::Logs => {
                self.log_sub_tab = LogSubTab::Tui;
                self.log_scroll = u16::MAX;
            }
            KeyCode::Char('b') if self.view == View::Logs => {
                self.log_sub_tab = LogSubTab::Gateway;
                self.log_scroll = u16::MAX;
            }
            KeyCode::Char('w') if self.view == View::Logs => {
                // Cycle through worker logs
                let tabs = self.worker_log_tabs();
                if tabs.is_empty() {
                    self.set_status("No worker logs available".to_string());
                } else {
                    let current_port = match &self.log_sub_tab {
                        LogSubTab::Worker(p) => Some(p.clone()),
                        _ => None,
                    };
                    let next = if let Some(cur) = current_port {
                        let idx = tabs.iter().position(|(_, p)| p == &cur).unwrap_or(0);
                        (idx + 1) % tabs.len()
                    } else {
                        0
                    };
                    self.log_sub_tab = LogSubTab::Worker(tabs[next].1.clone());
                    self.log_scroll = u16::MAX;
                }
            }

            // Input modes
            KeyCode::Char('/') => {
                self.input_mode = InputMode::Filter;
                self.input_buffer.clear();
            }
            KeyCode::Char(':') => {
                self.input_mode = InputMode::Command;
                self.input_buffer.clear();
            }

            // Help
            KeyCode::Char('?') => self.show_help = !self.show_help,

            KeyCode::Enter if self.view == View::Workers => {
                self.show_detail = !self.show_detail;
            }

            // Esc clears overlays/filters
            KeyCode::Esc => {
                if self.show_detail {
                    self.show_detail = false;
                } else if self.show_help {
                    self.show_help = false;
                } else {
                    self.active_filter = None;
                }
            }

            // Workers-only keys
            KeyCode::Char('d') if self.view == View::Workers => {
                self.start_delete();
            }
            KeyCode::Char('e') if self.view == View::Workers => {
                self.show_action_menu = true;
                self.action_menu_index = 0;
            }
            KeyCode::Char('a') if self.view == View::Workers => {
                self.add_menu_state = Some(AddMenuState::SelectCategory);
            }

            _ => {}
        }
    }

    fn handle_filter_input(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.active_filter = if self.input_buffer.is_empty() {
                    None
                } else {
                    Some(self.input_buffer.clone())
                };
                self.input_mode = InputMode::Normal;
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
                self.input_buffer.clear();
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            _ => {}
        }
    }

    async fn handle_command_input(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let cmd = self.input_buffer.clone();
                self.input_mode = InputMode::Normal;
                self.input_buffer.clear();
                self.execute_command(&cmd).await;
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
                self.input_buffer.clear();
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            _ => {}
        }
    }

    async fn execute_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.trim().splitn(2, ' ').collect();
        match parts.first().copied() {
            Some("quit" | "q") => self.should_quit = true,
            Some("add") => self.cmd_add(parts.get(1).copied()).await,
            Some("delete") => {
                if let Some(id) = parts.get(1) {
                    let id = id.trim().to_string();
                    // Look up the worker's URL by ID from state before deleting
                    let worker_url = {
                        #[expect(clippy::unwrap_used)]
                        let state = self.state.read().unwrap();
                        state.workers.as_ref().and_then(|w| {
                            w.workers
                                .iter()
                                .find(|wi| wi.id == id)
                                .map(|wi| wi.url.clone())
                        })
                    };
                    match self.client.delete_worker(&id).await {
                        Ok(_) => {
                            if let Some(url) = worker_url {
                                self.claimed_gpus.remove(&url);
                            }
                            self.set_status(format!("Worker {id} deleted"));
                        }
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                } else {
                    self.set_status("Usage: delete <id>".into());
                }
            }
            Some("priority") => {
                let args = parts.get(1).copied();
                if let Some(val) = args.and_then(|a| a.parse::<u32>().ok()) {
                    if let Some(id) = self.selected_worker_id() {
                        let update = openai_protocol::worker::WorkerUpdateRequest {
                            priority: Some(val),
                            cost: None,
                            labels: None,
                            api_key: None,
                            health: None,
                        };
                        match self.client.update_worker(&id, &update).await {
                            Ok(_) => self.set_status(format!("Priority set to {val}")),
                            Err(e) => self.set_status(format!("Error: {e}")),
                        }
                    } else {
                        self.set_status("No worker selected".to_string());
                    }
                } else {
                    self.set_status("Usage: :priority <number>".to_string());
                }
            }
            Some("cost") => {
                let args = parts.get(1).copied();
                if let Some(val) = args.and_then(|a| a.parse::<f32>().ok()) {
                    if let Some(id) = self.selected_worker_id() {
                        let update = openai_protocol::worker::WorkerUpdateRequest {
                            cost: Some(val),
                            priority: None,
                            labels: None,
                            api_key: None,
                            health: None,
                        };
                        match self.client.update_worker(&id, &update).await {
                            Ok(_) => self.set_status(format!("Cost set to {val}")),
                            Err(e) => self.set_status(format!("Error: {e}")),
                        }
                    } else {
                        self.set_status("No worker selected".to_string());
                    }
                } else {
                    self.set_status("Usage: :cost <number>".to_string());
                }
            }
            Some("api-key") => {
                if let Some(key) = parts.get(1) {
                    if let Some(id) = self.selected_worker_id() {
                        let update = openai_protocol::worker::WorkerUpdateRequest {
                            priority: None,
                            cost: None,
                            labels: None,
                            api_key: Some(key.trim().to_string()),
                            health: None,
                        };
                        match self.client.update_worker(&id, &update).await {
                            Ok(_) => self.set_status("API key updated".to_string()),
                            Err(e) => self.set_status(format!("Error: {e}")),
                        }
                    } else {
                        self.set_status("No worker selected".to_string());
                    }
                } else {
                    self.set_status("Usage: :api-key <key>".to_string());
                }
            }
            Some("flush-cache") => {
                if let Some(id) = self.selected_worker_id() {
                    let url = self.selected_worker_url().unwrap_or_default();
                    self.confirm_flush = Some((id, url));
                } else {
                    self.set_status("No worker selected".to_string());
                }
            }
            Some("toggle-health") => {
                if let Some(worker) = self.selected_worker() {
                    // Toggle: if healthy, disable health check; if unhealthy, re-enable
                    let disable = worker.is_healthy;
                    let update = openai_protocol::worker::WorkerUpdateRequest {
                        health: Some(openai_protocol::worker::HealthCheckUpdate {
                            disable_health_check: Some(disable),
                            timeout_secs: None,
                            check_interval_secs: None,
                            success_threshold: None,
                            failure_threshold: None,
                            drain_settle_secs: None,
                        }),
                        priority: None,
                        cost: None,
                        labels: None,
                        api_key: None,
                    };
                    let action = if disable { "disabled" } else { "enabled" };
                    match self.client.update_worker(&worker.id, &update).await {
                        Ok(_) => self.set_status(format!("Health check {action}")),
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                } else {
                    self.set_status("No worker selected".to_string());
                }
            }
            Some("add-openai") => {
                self.add_menu_state = Some(AddMenuState::EnterApiKey {
                    provider: ProviderPreset::OpenAI,
                    input: String::new(),
                });
            }
            Some("add-anthropic") => {
                self.add_menu_state = Some(AddMenuState::EnterApiKey {
                    provider: ProviderPreset::Anthropic,
                    input: String::new(),
                });
            }
            Some("add-xai") => {
                self.add_menu_state = Some(AddMenuState::EnterApiKey {
                    provider: ProviderPreset::Xai,
                    input: String::new(),
                });
            }
            Some("add-gemini") => {
                self.add_menu_state = Some(AddMenuState::EnterApiKey {
                    provider: ProviderPreset::Gemini,
                    input: String::new(),
                });
            }
            _ => self.set_status(format!("Unknown command: {cmd}")),
        }
    }

    /// Parse `:add <url> [--provider <p>] [--runtime <r>]`
    async fn cmd_add(&mut self, args: Option<&str>) {
        let Some(args) = args else {
            self.set_status(
                "Usage: add <url> [--provider openai|anthropic|gemini|xai] [--runtime external|sglang|vllm|trtllm]".into(),
            );
            return;
        };

        let tokens: Vec<&str> = args.split_whitespace().collect();
        if tokens.is_empty() {
            self.set_status("Usage: add <url> [--provider <p>] [--runtime <r>]".into());
            return;
        }

        let url = tokens[0].to_string();
        let mut spec = WorkerSpec::new(url);

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "--provider" | "-p" => {
                    i += 1;
                    if i < tokens.len() {
                        spec.provider = Some(parse_provider(tokens[i]));
                        // Auto-set runtime to external when provider is specified
                        spec.runtime_type = RuntimeType::External;
                    }
                }
                "--runtime" | "-r" => {
                    i += 1;
                    if i < tokens.len() {
                        spec.runtime_type = match tokens[i].parse() {
                            Ok(rt) => rt,
                            Err(_) => {
                                self.set_status(format!("Error: Invalid runtime '{}'", tokens[i]));
                                return;
                            }
                        };
                    }
                }
                _ => {} // skip unknown flags
            }
            i += 1;
        }

        match self.client.add_worker(&spec).await {
            Ok(_) => self.set_status("Worker added".into()),
            Err(e) => self.set_status(format!("Error: {e}")),
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') if !self.chat_streaming && self.chat_input.is_empty() => {
                self.should_quit = true;
            }
            // Tab to cycle model, Backtab (Shift+Tab) to cycle endpoint
            KeyCode::Tab if !self.chat_streaming => {
                self.cycle_chat_model();
            }
            KeyCode::BackTab if !self.chat_streaming => {
                self.chat_endpoint = self.chat_endpoint.cycle();
                self.chat_previous_response_id = None;
                self.chat_messages.clear(); // clear conversation on endpoint switch
                self.set_status(format!("Endpoint: /v1/{}", self.chat_endpoint.label()));
            }
            // Number keys for view switching (only when not typing)
            code @ (KeyCode::Char('1')
            | KeyCode::Char('2')
            | KeyCode::Char('4')
            | KeyCode::Char('5')
            | KeyCode::Char('6')
            | KeyCode::Char('7'))
                if self.chat_input.is_empty() && !self.chat_streaming =>
            {
                if let Some(v) = View::from_key(code) {
                    self.view = v;
                    self.selected_index = 0;
                }
            }
            KeyCode::Char('?') if self.chat_input.is_empty() && !self.chat_streaming => {
                self.show_help = !self.show_help;
            }
            // Enter sends the message
            KeyCode::Enter => {
                if self.chat_streaming {
                    return; // Don't send while streaming
                }
                let text = self.chat_input.trim().to_string();
                if text.is_empty() {
                    return;
                }
                // Auto-select first model if none set
                if self.chat_model.is_empty() {
                    self.cycle_chat_model();
                    if self.chat_model.is_empty() {
                        self.set_status("No models available — add a worker first".to_string());
                        return;
                    }
                }
                self.chat_input.clear();
                self.chat_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: text,
                });
                // Add empty assistant message that will be filled by streaming
                self.chat_messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                });
                self.chat_streaming = true;
                self.chat_scroll = u16::MAX;

                // Build messages for API
                let api_messages: Vec<serde_json::Value> = self
                    .chat_messages
                    .iter()
                    .filter(|m| !m.content.is_empty())
                    .map(|m| {
                        serde_json::json!({
                            "role": m.role,
                            "content": m.content,
                        })
                    })
                    .collect();

                let (tx, rx) = mpsc::unbounded_channel();
                self.chat_stream_rx = Some(rx);

                let client = self.client.clone();
                let model = self.chat_model.clone();
                let endpoint = self.chat_endpoint;
                let prev_id = self.chat_previous_response_id.clone();
                // Safety: fire-and-forget streaming task; tokens are sent via channel
                #[expect(clippy::disallowed_methods)]
                tokio::spawn(async move {
                    crate::chat::stream_chat(&client, &model, &api_messages, endpoint, prev_id, tx)
                        .await;
                });
            }
            // Esc cancels input or stops streaming
            KeyCode::Esc => {
                if self.chat_streaming {
                    self.chat_streaming = false;
                    self.chat_stream_rx = None;
                    if let Some(msg) = self.chat_messages.last_mut() {
                        if msg.content.is_empty() {
                            self.chat_messages.pop();
                        }
                    }
                } else if !self.chat_input.is_empty() {
                    self.chat_input.clear();
                }
            }
            KeyCode::Backspace => {
                self.chat_input.pop();
            }
            KeyCode::Char(c) => {
                self.chat_input.push(c);
            }
            KeyCode::Up => {
                self.chat_scroll = self.chat_scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                self.chat_scroll = self.chat_scroll.saturating_add(1);
            }
            _ => {}
        }
    }

    fn cycle_chat_model(&mut self) {
        // Safety: RwLock is not poisoned — no panics while holding the lock
        #[expect(clippy::unwrap_used)]
        let state = self.state.read().unwrap();

        // Collect chat-capable models from workers.
        // OpenAI has too many models — only show gpt-5.4* to keep the list manageable.
        // All other providers/local workers show all chat models.
        let mut models: Vec<String> = Vec::new();
        if let Some(ref workers) = state.workers {
            for w in &workers.workers {
                let is_openai = w.url.contains("openai.com");
                for m in &w.models {
                    let is_chat = m.model_type.iter().any(|t| t == "chat");
                    if !is_chat {
                        continue;
                    }
                    if is_openai && !m.id.starts_with("gpt-5.4") {
                        continue;
                    }
                    if !models.contains(&m.id) {
                        models.push(m.id.clone());
                    }
                }
            }
        }
        drop(state);

        if models.is_empty() {
            self.set_status("No models available".to_string());
            return;
        }

        let current_idx = models.iter().position(|m| m == &self.chat_model);
        let next_idx = match current_idx {
            Some(i) => (i + 1) % models.len(),
            None => 0,
        };
        self.chat_model.clone_from(&models[next_idx]);
        self.set_status(format!("Model: {}", self.chat_model));
    }

    fn start_delete(&mut self) {
        // Safety: RwLock is not poisoned — no panics while holding the lock
        #[expect(clippy::unwrap_used)]
        let state = self.state.read().unwrap();
        if let Some(ref wl) = state.workers {
            let filtered: Vec<_> = wl
                .workers
                .iter()
                .filter(|w| {
                    self.active_filter.as_ref().is_none_or(|f| {
                        w.id.to_lowercase().contains(&f.to_lowercase())
                            || w.url.to_lowercase().contains(&f.to_lowercase())
                    })
                })
                .collect();

            if let Some(worker) = filtered.get(self.selected_index) {
                self.confirm_delete = Some((worker.id.clone(), worker.url.clone()));
            }
        }
    }

    async fn handle_delete_confirm(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some((ref id, ref url)) = self.confirm_delete {
                    let id = id.clone();
                    let url = url.clone();
                    match self.client.delete_worker(&id).await {
                        Ok(_) => {
                            // Release claimed GPUs for this worker
                            if self.claimed_gpus.remove(&url).is_some() {
                                self.add_log(
                                    LogLevel::Info,
                                    &format!("Released GPU claim for {url}"),
                                );
                            }
                            // Kill the backend process if it was spawned by the TUI
                            if let Some(port) = url.rsplit(':').next() {
                                let port = port.trim_matches('/').to_string();
                                let match_str = format!("port {port}");
                                let mut kill_result = None;
                                for (desc, child) in &mut self.worker_children {
                                    if desc.contains(&match_str) {
                                        kill_result = Some((desc.clone(), child.kill().await));
                                        break;
                                    }
                                }
                                if let Some((desc, result)) = kill_result {
                                    match result {
                                        Ok(()) => self.add_log(
                                            LogLevel::Info,
                                            &format!("Killed backend: {desc}"),
                                        ),
                                        Err(e) => self.add_log(
                                            LogLevel::Warn,
                                            &format!("Failed to kill backend: {e}"),
                                        ),
                                    }
                                    self.worker_children
                                        .retain(|(d, _)| !d.contains(&match_str));
                                }
                            }
                            self.set_status(format!("Worker {id} deleted"));
                        }
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                }
                self.confirm_delete = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.confirm_delete = None;
            }
            _ => {}
        }
    }

    async fn handle_action_menu_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.show_action_menu = false;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                let max = ActionMenuItem::all().len().saturating_sub(1);
                self.action_menu_index = (self.action_menu_index + 1).min(max);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.action_menu_index = self.action_menu_index.saturating_sub(1);
            }
            KeyCode::Enter => {
                let item = ActionMenuItem::all()[self.action_menu_index];
                self.show_action_menu = false;
                match item {
                    ActionMenuItem::UpdatePriority => {
                        self.input_mode = InputMode::Command;
                        self.input_buffer = "priority ".to_string();
                    }
                    ActionMenuItem::UpdateCost => {
                        self.input_mode = InputMode::Command;
                        self.input_buffer = "cost ".to_string();
                    }
                    ActionMenuItem::UpdateApiKey => {
                        self.input_mode = InputMode::Command;
                        self.input_buffer = "api-key ".to_string();
                    }
                    ActionMenuItem::FlushCache => {
                        if let Some(id) = self.selected_worker_id() {
                            let url = self.selected_worker_url().unwrap_or_default();
                            self.confirm_flush = Some((id, url));
                        }
                    }
                    ActionMenuItem::ToggleHealthCheck => {
                        if let Some(worker) = self.selected_worker() {
                            let disable = worker.is_healthy;
                            let action = if disable { "disabled" } else { "enabled" };
                            match self
                                .client
                                .update_worker(
                                    &worker.id,
                                    &openai_protocol::worker::WorkerUpdateRequest {
                                        priority: None,
                                        cost: None,
                                        labels: None,
                                        api_key: None,
                                        health: Some(openai_protocol::worker::HealthCheckUpdate {
                                            disable_health_check: Some(disable),
                                            timeout_secs: None,
                                            check_interval_secs: None,
                                            success_threshold: None,
                                            failure_threshold: None,
                                            drain_settle_secs: None,
                                        }),
                                    },
                                )
                                .await
                            {
                                Ok(_) => {
                                    self.set_status(format!("Health check {action}"));
                                }
                                Err(e) => self.set_status(format!("Error: {e}")),
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    async fn handle_add_menu_key(&mut self, key: KeyEvent) {
        use crate::types::{LocalModelPreset, LocalRuntime};

        let state_clone = self.add_menu_state.clone();
        match &state_clone {
            Some(AddMenuState::SelectCategory) => match key.code {
                KeyCode::Esc => self.add_menu_state = None,
                KeyCode::Char('1') => self.add_menu_state = Some(AddMenuState::SelectProvider),
                KeyCode::Char('2') => self.add_menu_state = Some(AddMenuState::SelectRuntime),
                KeyCode::Char('3') => {
                    self.add_menu_state = Some(AddMenuState::EnterCustomUrl {
                        input: String::new(),
                    });
                }
                _ => {}
            },
            Some(AddMenuState::SelectProvider) => match key.code {
                KeyCode::Esc => self.add_menu_state = Some(AddMenuState::SelectCategory),
                KeyCode::Char('1') => {
                    self.add_menu_state = Some(AddMenuState::EnterApiKey {
                        provider: ProviderPreset::OpenAI,
                        input: String::new(),
                    });
                }
                KeyCode::Char('2') => {
                    self.add_menu_state = Some(AddMenuState::EnterApiKey {
                        provider: ProviderPreset::Anthropic,
                        input: String::new(),
                    });
                }
                KeyCode::Char('3') => {
                    self.add_menu_state = Some(AddMenuState::EnterApiKey {
                        provider: ProviderPreset::Xai,
                        input: String::new(),
                    });
                }
                KeyCode::Char('4') => {
                    self.add_menu_state = Some(AddMenuState::EnterApiKey {
                        provider: ProviderPreset::Gemini,
                        input: String::new(),
                    });
                }
                _ => {}
            },
            Some(AddMenuState::EnterApiKey { provider, input }) => match key.code {
                KeyCode::Esc => self.add_menu_state = Some(AddMenuState::SelectProvider),
                KeyCode::Enter => {
                    let provider = *provider;
                    // Use entered key, or fall back to env var
                    let api_key = if input.is_empty() {
                        std::env::var(provider.env_key()).unwrap_or_default()
                    } else {
                        input.clone()
                    };
                    if api_key.is_empty() {
                        self.set_status("No API key provided".to_string());
                        return;
                    }
                    self.add_menu_state = None;
                    let mut spec = WorkerSpec::new(provider.url());
                    spec.provider = Some(provider.provider_type());
                    spec.runtime_type = provider.runtime_type();
                    spec.api_key = Some(api_key);
                    match self.client.add_worker(&spec).await {
                        Ok(_) => self.set_status(format!("Added {} worker", provider.label())),
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                }
                KeyCode::Backspace => {
                    if let Some(AddMenuState::EnterApiKey { ref mut input, .. }) =
                        self.add_menu_state
                    {
                        input.pop();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(AddMenuState::EnterApiKey { ref mut input, .. }) =
                        self.add_menu_state
                    {
                        input.push(c);
                    }
                }
                _ => {}
            },
            Some(AddMenuState::SelectRuntime) => match key.code {
                KeyCode::Esc => self.add_menu_state = Some(AddMenuState::SelectCategory),
                KeyCode::Char('1') => {
                    self.add_menu_state = Some(AddMenuState::SelectConnection {
                        runtime: LocalRuntime::Sglang,
                    });
                }
                KeyCode::Char('2') => {
                    self.add_menu_state = Some(AddMenuState::SelectConnection {
                        runtime: LocalRuntime::Vllm,
                    });
                }
                _ => {}
            },
            Some(AddMenuState::SelectConnection { runtime }) => match key.code {
                KeyCode::Esc => self.add_menu_state = Some(AddMenuState::SelectRuntime),
                KeyCode::Char('1') => {
                    self.add_menu_state = Some(AddMenuState::SelectModel {
                        runtime: *runtime,
                        grpc: false,
                    });
                }
                KeyCode::Char('2') => {
                    self.add_menu_state = Some(AddMenuState::SelectModel {
                        runtime: *runtime,
                        grpc: true,
                    });
                }
                _ => {}
            },
            Some(AddMenuState::SelectModel { runtime, grpc }) => {
                let presets = LocalModelPreset::all();
                let custom_idx = presets.len() + 1;
                match key.code {
                    KeyCode::Esc => {
                        self.add_menu_state =
                            Some(AddMenuState::SelectConnection { runtime: *runtime });
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        let idx = c.to_digit(10).unwrap_or(0) as usize;
                        if idx >= 1 && idx <= presets.len() {
                            let model = presets[idx - 1].clone();
                            let runtime = *runtime;
                            let grpc = *grpc;
                            self.add_menu_state = None;
                            self.spawn_local_worker_with_args(runtime, model, grpc, "")
                                .await;
                        } else if idx == custom_idx {
                            self.add_menu_state = Some(AddMenuState::EnterCustomModel {
                                runtime: *runtime,
                                grpc: *grpc,
                                field: 0,
                                model_id: String::new(),
                                tp: "1".to_string(),
                                extra_args: String::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Some(AddMenuState::EnterCustomUrl { input }) => match key.code {
                KeyCode::Esc => self.add_menu_state = Some(AddMenuState::SelectCategory),
                KeyCode::Enter => {
                    let url = input.clone();
                    self.add_menu_state = None;
                    let spec = WorkerSpec::new(url);
                    match self.client.add_worker(&spec).await {
                        Ok(_) => self.set_status("Added custom worker".to_string()),
                        Err(e) => self.set_status(format!("Error: {e}")),
                    }
                }
                KeyCode::Backspace => {
                    if let Some(AddMenuState::EnterCustomUrl { ref mut input }) =
                        self.add_menu_state
                    {
                        input.pop();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(AddMenuState::EnterCustomUrl { ref mut input }) =
                        self.add_menu_state
                    {
                        input.push(c);
                    }
                }
                _ => {}
            },
            Some(AddMenuState::EnterCustomModel {
                runtime,
                grpc,
                field: _,
                model_id,
                tp,
                extra_args,
            }) => {
                let runtime = *runtime;
                let grpc = *grpc;
                match key.code {
                    KeyCode::Esc => {
                        self.add_menu_state = Some(AddMenuState::SelectModel { runtime, grpc });
                    }
                    KeyCode::Tab => {
                        // Cycle to next field
                        if let Some(AddMenuState::EnterCustomModel { ref mut field, .. }) =
                            self.add_menu_state
                        {
                            *field = (*field + 1) % 3;
                        }
                    }
                    KeyCode::BackTab => {
                        if let Some(AddMenuState::EnterCustomModel { ref mut field, .. }) =
                            self.add_menu_state
                        {
                            *field = if *field == 0 { 2 } else { *field - 1 };
                        }
                    }
                    KeyCode::Enter => {
                        let model_id = model_id.clone();
                        let tp_str = tp.clone();
                        let extra = extra_args.clone();
                        self.add_menu_state = None;
                        if model_id.is_empty() {
                            self.set_status("Model ID is required".to_string());
                            return;
                        }
                        let tp_val: u32 = tp_str.parse().unwrap_or(1);
                        let model = LocalModelPreset::Custom {
                            model_id,
                            tp: tp_val,
                        };
                        self.spawn_local_worker_with_args(runtime, model, grpc, &extra)
                            .await;
                    }
                    KeyCode::Backspace => {
                        if let Some(AddMenuState::EnterCustomModel {
                            ref mut model_id,
                            ref mut tp,
                            ref mut extra_args,
                            field,
                            ..
                        }) = self.add_menu_state
                        {
                            match field {
                                0 => {
                                    model_id.pop();
                                }
                                1 => {
                                    tp.pop();
                                }
                                _ => {
                                    extra_args.pop();
                                }
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        if let Some(AddMenuState::EnterCustomModel {
                            ref mut model_id,
                            ref mut tp,
                            ref mut extra_args,
                            field,
                            ..
                        }) = self.add_menu_state
                        {
                            match field {
                                0 => model_id.push(c),
                                1 => {
                                    if c.is_ascii_digit() {
                                        tp.push(c);
                                    }
                                }
                                _ => extra_args.push(c),
                            }
                        }
                    }
                    _ => {}
                }
            }
            None => {}
        }
    }

    /// Get the selected worker from the filtered list (matching the UI's filter).
    fn selected_worker(&self) -> Option<crate::client::WorkerInfo> {
        // Safety: RwLock is not poisoned — no panics while holding the lock
        #[expect(clippy::unwrap_used)]
        let state = self.state.read().unwrap();
        state.workers.as_ref().and_then(|w| {
            let filtered: Vec<_> = w
                .workers
                .iter()
                .filter(|w| {
                    self.active_filter.as_ref().is_none_or(|f| {
                        f.is_empty()
                            || w.id.to_lowercase().contains(&f.to_lowercase())
                            || w.url.to_lowercase().contains(&f.to_lowercase())
                            || w.runtime_type.to_lowercase().contains(&f.to_lowercase())
                    })
                })
                .collect();
            filtered.get(self.selected_index).copied().cloned()
        })
    }

    fn selected_worker_id(&self) -> Option<String> {
        self.selected_worker().map(|w| w.id)
    }

    fn selected_worker_url(&self) -> Option<String> {
        self.selected_worker().map(|w| w.url)
    }

    fn set_status(&mut self, msg: String) {
        let level = if msg.starts_with("Error") {
            LogLevel::Error
        } else {
            LogLevel::Info
        };
        self.add_log(level, &msg);
        self.status_message = Some(msg);
        self.status_clear_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
    }

    async fn spawn_local_worker_with_args(
        &mut self,
        runtime: crate::types::LocalRuntime,
        model: crate::types::LocalModelPreset,
        grpc: bool,
        extra_args: &str,
    ) {
        // Find an available port
        let port = match std::net::TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => match listener.local_addr() {
                Ok(addr) => addr.port(),
                Err(e) => {
                    self.set_status(format!("Failed to get local address: {e}"));
                    return;
                }
            },
            Err(e) => {
                self.set_status(format!("Failed to find available port: {e}"));
                return;
            }
        };

        // Find GPUs with enough free memory, excluding already-claimed GPUs
        let all_claimed: std::collections::HashSet<u32> =
            self.claimed_gpus.values().flatten().copied().collect();
        let free_gpus: Vec<u32> = find_free_gpus()
            .await
            .into_iter()
            .filter(|g| !all_claimed.contains(g))
            .collect();
        let tp = model.tp();
        if free_gpus.is_empty() {
            self.set_status(
                "No GPUs available (all in use or claimed by pending workers)".to_string(),
            );
            return;
        }
        if (tp as usize) > free_gpus.len() {
            self.set_status(format!(
                "Not enough free GPUs: model requires TP={tp} but only {} unclaimed GPU(s) available",
                free_gpus.len()
            ));
            return;
        }
        // Pick the first `tp` free GPUs
        let selected_gpus: Vec<u32> = free_gpus.into_iter().take(tp as usize).collect();
        let cuda_devices = selected_gpus
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let model_id = model.model_id().to_string();
        let conn_label = if grpc { "grpc" } else { "http" };
        let (cmd, mut args) = runtime.launch_args(&model_id, tp, port, grpc);
        // Append user-provided extra arguments (e.g. --max-model-len 16384)
        for arg in extra_args.split_whitespace() {
            args.push(arg.to_string());
        }
        let desc = format!(
            "{} {} {} (port {port})",
            runtime.label(),
            conn_label,
            model.label()
        );

        self.add_log(
            LogLevel::Info,
            &format!("Starting {desc} on GPU [{cuda_devices}]..."),
        );
        self.add_log(
            LogLevel::Info,
            &format!(
                "Command: CUDA_VISIBLE_DEVICES={cuda_devices} {cmd} {}",
                args.join(" ")
            ),
        );

        let log_path = format!("/tmp/smg-worker-{port}.log");
        let log_file = match std::fs::File::create(&log_path) {
            Ok(f) => f,
            Err(e) => {
                self.set_status(format!("Failed to create log file: {e}"));
                return;
            }
        };
        let log_file2 = match log_file.try_clone() {
            Ok(f) => f,
            Err(e) => {
                self.set_status(format!("Failed to clone log file handle: {e}"));
                return;
            }
        };

        match tokio::process::Command::new(&cmd)
            .args(&args)
            .env("CUDA_VISIBLE_DEVICES", &cuda_devices)
            .stdout(std::process::Stdio::from(log_file))
            .stderr(std::process::Stdio::from(log_file2))
            .spawn()
        {
            Ok(child) => {
                self.add_log(
                    LogLevel::Info,
                    &format!(
                        "Worker started (pid {}, log: {log_path})",
                        child.id().unwrap_or(0)
                    ),
                );
                self.worker_children.push((desc.clone(), child));

                // Register with gateway immediately — SMG handles health checks itself
                let url = if grpc {
                    format!("grpc://127.0.0.1:{port}")
                } else {
                    format!("http://127.0.0.1:{port}")
                };
                // Track claimed GPUs so next worker won't pick the same ones
                self.claimed_gpus.insert(url.clone(), selected_gpus);
                let runtime_type = runtime.runtime_type();
                let connection_mode = if grpc {
                    openai_protocol::worker::ConnectionMode::Grpc
                } else {
                    openai_protocol::worker::ConnectionMode::Http
                };
                // Register with gateway immediately — SMG handles health checks
                let mut spec = WorkerSpec::new(&url);
                spec.runtime_type = runtime_type;
                spec.connection_mode = connection_mode;
                match self.client.add_worker(&spec).await {
                    Ok(_) => {
                        self.add_log(
                            LogLevel::Info,
                            &format!("Registered worker {url} with gateway"),
                        );
                        self.set_status(format!("Started {desc} — registered with gateway"));
                    }
                    Err(e) => {
                        self.add_log(
                            LogLevel::Error,
                            &format!("Failed to register worker {url}: {e}"),
                        );
                        // Roll back: kill the spawned process and release claimed GPUs
                        if let Some(pos) = self.worker_children.iter().position(|(d, _)| d == &desc)
                        {
                            let (_, mut child) = self.worker_children.remove(pos);
                            let _ = child.kill().await;
                            self.add_log(
                                LogLevel::Info,
                                &format!("Rolled back spawned worker: {desc}"),
                            );
                        }
                        self.claimed_gpus.remove(&url);
                        self.set_status(format!("Registration failed, worker rolled back: {e}"));
                    }
                }
            }
            Err(e) => {
                self.set_status(format!("Failed to start worker: {e}"));
            }
        }
    }

    /// Get available worker log files as (label, port) pairs.
    pub fn worker_log_tabs(&self) -> Vec<(String, String)> {
        let mut tabs = Vec::new();
        for (desc, _child) in &self.worker_children {
            // desc is like "sglang http Llama-3.2-1B (TP=1) (port 37595)"
            if let Some(port_start) = desc.rfind("port ") {
                let port = desc[port_start + 5..].trim_end_matches(')').to_string();
                // Extract short model name (first 5 chars of model label)
                let short = desc
                    .split_whitespace()
                    .nth(2)
                    .unwrap_or("work")
                    .chars()
                    .take(5)
                    .collect::<String>();
                tabs.push((format!("{short}-{port}"), port));
            }
        }
        tabs
    }

    fn add_log(&mut self, level: LogLevel, message: &str) {
        if self.log_entries.len() >= MAX_LOG_ENTRIES {
            self.log_entries.pop_front();
        }
        self.log_entries.push_back(LogEntry {
            timestamp: chrono::Utc::now(),
            level,
            message: message.to_string(),
        });
        if self.log_auto_scroll {
            self.log_scroll = u16::MAX;
        }
    }

    fn clamp_selection(&mut self) {
        // Clamp against filtered list size (matches what the UI renders)
        #[expect(clippy::unwrap_used)]
        let state = self.state.read().unwrap();
        if let Some(ref wl) = state.workers {
            let count = wl
                .workers
                .iter()
                .filter(|w| {
                    self.active_filter.as_ref().is_none_or(|f| {
                        f.is_empty()
                            || w.id.to_lowercase().contains(&f.to_lowercase())
                            || w.url.to_lowercase().contains(&f.to_lowercase())
                            || w.runtime_type.to_lowercase().contains(&f.to_lowercase())
                    })
                })
                .count();
            if count > 0 {
                self.selected_index = self.selected_index.min(count - 1);
            }
        }
    }
}

/// Find GPUs with <10% VRAM utilization via nvidia-smi. Returns empty if unavailable.
async fn find_free_gpus() -> Vec<u32> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        let idx = parts[0].parse::<u32>().ok()?;
                        let used_mb = parts[1].parse::<f64>().ok()?;
                        let total_mb = parts[2].parse::<f64>().ok()?;
                        // Only consider GPUs with <10% VRAM utilization
                        if total_mb > 0.0 && (used_mb / total_mb) < 0.10 {
                            Some(idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

fn parse_provider(s: &str) -> ProviderType {
    match s.to_lowercase().as_str() {
        "openai" => ProviderType::OpenAI,
        "anthropic" | "claude" => ProviderType::Anthropic,
        "gemini" | "google" => ProviderType::Gemini,
        "xai" | "grok" => ProviderType::XAI,
        other => ProviderType::Custom(other.to_string()),
    }
}
