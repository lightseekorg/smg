use crossterm::event::KeyCode;

/// Active view/tab in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum View {
    #[default]
    Pulse,
    Workers,
    Chat,
    Logs,
    Benchmark,
    Traffic,
    Mesh,
}

impl View {
    /// Map a number key to a view.
    pub fn from_key(code: KeyCode) -> Option<Self> {
        match code {
            KeyCode::Char('1') => Some(Self::Pulse),
            KeyCode::Char('2') => Some(Self::Workers),
            KeyCode::Char('3') => Some(Self::Chat),
            KeyCode::Char('4') => Some(Self::Logs),
            KeyCode::Char('5') => Some(Self::Benchmark),
            KeyCode::Char('6') => Some(Self::Traffic),
            KeyCode::Char('7') => Some(Self::Mesh),
            _ => None,
        }
    }

    /// Human-readable label for the tab bar.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pulse => "Pulse",
            Self::Workers => "Workers",
            Self::Chat => "Chat",
            Self::Logs => "Logs",
            Self::Benchmark => "Benchmark",
            Self::Traffic => "Traffic",
            Self::Mesh => "Mesh",
        }
    }

    /// All views in order.
    pub fn all() -> &'static [View] {
        &[
            Self::Pulse,
            Self::Workers,
            Self::Chat,
            Self::Logs,
            Self::Benchmark,
            Self::Traffic,
            Self::Mesh,
        ]
    }

    /// 1-based index for display.
    pub fn index(self) -> usize {
        match self {
            Self::Pulse => 1,
            Self::Workers => 2,
            Self::Chat => 3,
            Self::Logs => 4,
            Self::Benchmark => 5,
            Self::Traffic => 6,
            Self::Mesh => 7,
        }
    }
}

/// Input mode determines how keystrokes are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    /// Normal navigation mode — keys are shortcuts.
    #[default]
    Normal,
    /// Filter mode — typing populates the filter bar (prefix: `/`).
    Filter,
    /// Command mode — typing populates the command bar (prefix: `:`).
    Command,
}

/// State machine for the Add Worker menu flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddMenuState {
    /// Top-level: 1. External  2. Local  3. Custom URL
    SelectCategory,
    /// External: pick provider
    SelectProvider,
    /// External: enter API key
    EnterApiKey {
        provider: ProviderPreset,
        input: String,
    },
    /// Local: pick runtime (sglang/vllm)
    SelectRuntime,
    /// Local: pick connection (http/grpc)
    SelectConnection { runtime: LocalRuntime },
    /// Local: pick model preset
    SelectModel { runtime: LocalRuntime, grpc: bool },
    /// Local: custom model — multi-field form
    EnterCustomModel {
        runtime: LocalRuntime,
        grpc: bool,
        /// 0=model_id, 1=tp, 2=extra_args
        field: u8,
        model_id: String,
        tp: String,
        extra_args: String,
    },
    /// Custom: enter URL
    EnterCustomUrl { input: String },
}

impl AddMenuState {
    pub fn get_input(&self) -> Option<String> {
        match self {
            Self::EnterApiKey { input, .. } | Self::EnterCustomUrl { input } => Some(input.clone()),
            Self::EnterCustomModel {
                field,
                model_id,
                tp,
                extra_args,
                ..
            } => Some(
                match field {
                    0 => model_id,
                    1 => tp,
                    _ => extra_args,
                }
                .clone(),
            ),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalRuntime {
    Sglang,
    Vllm,
}

impl LocalRuntime {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }

    pub fn runtime_type(self) -> openai_protocol::worker::RuntimeType {
        match self {
            Self::Sglang => openai_protocol::worker::RuntimeType::Sglang,
            Self::Vllm => openai_protocol::worker::RuntimeType::Vllm,
        }
    }

    /// Build the command and args to launch a worker.
    pub fn launch_args(
        self,
        model_id: &str,
        tp: u32,
        port: u16,
        grpc: bool,
    ) -> (String, Vec<String>) {
        match self {
            Self::Sglang => {
                let mut args = vec![
                    "-m".to_string(),
                    "sglang.launch_server".to_string(),
                    "--model-path".to_string(),
                    model_id.to_string(),
                    "--tp-size".to_string(),
                    tp.to_string(),
                    "--port".to_string(),
                    port.to_string(),
                    "--host".to_string(),
                    "0.0.0.0".to_string(),
                ];
                if grpc {
                    args.push("--grpc-mode".to_string());
                }
                ("python3".to_string(), args)
            }
            Self::Vllm => {
                let entrypoint = if grpc {
                    "vllm.entrypoints.grpc_server"
                } else {
                    "vllm.entrypoints.openai.api_server"
                };
                let args = vec![
                    "-m".to_string(),
                    entrypoint.to_string(),
                    "--model".to_string(),
                    model_id.to_string(),
                    "--tensor-parallel-size".to_string(),
                    tp.to_string(),
                    "--port".to_string(),
                    port.to_string(),
                    "--host".to_string(),
                    "0.0.0.0".to_string(),
                    "--max-model-len".to_string(),
                    "16384".to_string(),
                    "--gpu-memory-utilization".to_string(),
                    "0.9".to_string(),
                ];
                ("python3".to_string(), args)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalModelPreset {
    Preset {
        name: &'static str,
        model_id: &'static str,
        tp: u32,
    },
    Custom {
        model_id: String,
        tp: u32,
    },
}

impl LocalModelPreset {
    pub fn all() -> Vec<LocalModelPreset> {
        vec![
            Self::Preset {
                name: "Llama-3.2-1B",
                model_id: "meta-llama/Llama-3.2-1B-Instruct",
                tp: 1,
            },
            Self::Preset {
                name: "Llama-3.1-8B",
                model_id: "meta-llama/Llama-3.1-8B-Instruct",
                tp: 1,
            },
            Self::Preset {
                name: "Qwen2.5-7B",
                model_id: "Qwen/Qwen2.5-7B-Instruct",
                tp: 1,
            },
            Self::Preset {
                name: "Qwen2.5-14B",
                model_id: "Qwen/Qwen2.5-14B-Instruct",
                tp: 2,
            },
            Self::Preset {
                name: "DeepSeek-R1-7B",
                model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                tp: 1,
            },
            Self::Preset {
                name: "Mistral-7B",
                model_id: "mistralai/Mistral-7B-Instruct-v0.3",
                tp: 1,
            },
        ]
    }

    pub fn label(&self) -> String {
        match self {
            Self::Preset { name, tp, .. } => format!("{name} (TP={tp})"),
            Self::Custom { model_id, tp } => format!("{model_id} (TP={tp})"),
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::Preset { model_id, .. } => model_id,
            Self::Custom { model_id, .. } => model_id,
        }
    }

    pub fn tp(&self) -> u32 {
        match self {
            Self::Preset { tp, .. } | Self::Custom { tp, .. } => *tp,
        }
    }
}

/// Preset provider for quick-add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPreset {
    OpenAI,
    Anthropic,
    Xai,
    Gemini,
}

impl ProviderPreset {
    pub fn url(self) -> &'static str {
        match self {
            Self::OpenAI => "https://api.openai.com",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Xai => "https://api.x.ai",
            Self::Gemini => "https://generativelanguage.googleapis.com",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OpenAI => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::Xai => "xAI",
            Self::Gemini => "Gemini",
        }
    }

    pub fn provider_type(self) -> openai_protocol::worker::ProviderType {
        match self {
            Self::OpenAI => openai_protocol::worker::ProviderType::OpenAI,
            Self::Anthropic => openai_protocol::worker::ProviderType::Anthropic,
            Self::Xai => openai_protocol::worker::ProviderType::XAI,
            Self::Gemini => openai_protocol::worker::ProviderType::Gemini,
        }
    }

    /// Environment variable name for this provider's API key.
    pub fn env_key(self) -> &'static str {
        match self {
            Self::OpenAI => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Xai => "XAI_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
        }
    }

    #[expect(clippy::unused_self)] // Part of consistent per-preset API pattern
    pub fn runtime_type(self) -> openai_protocol::worker::RuntimeType {
        openai_protocol::worker::RuntimeType::External
    }

    pub fn all() -> &'static [ProviderPreset] {
        &[Self::OpenAI, Self::Anthropic, Self::Xai, Self::Gemini]
    }
}

/// Items in the worker action menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMenuItem {
    UpdatePriority,
    UpdateCost,
    UpdateApiKey,
    FlushCache,
    ToggleHealthCheck,
}

impl ActionMenuItem {
    pub fn label(self) -> &'static str {
        match self {
            Self::UpdatePriority => "Update priority",
            Self::UpdateCost => "Update cost",
            Self::UpdateApiKey => "Update API key",
            Self::FlushCache => "Flush cache",
            Self::ToggleHealthCheck => "Toggle health check",
        }
    }

    pub fn all() -> &'static [ActionMenuItem] {
        &[
            Self::UpdatePriority,
            Self::UpdateCost,
            Self::UpdateApiKey,
            Self::FlushCache,
            Self::ToggleHealthCheck,
        ]
    }
}
