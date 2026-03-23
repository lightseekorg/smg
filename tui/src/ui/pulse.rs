use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::{sparkline, theme};
use crate::app::App;

pub fn render_pulse(f: &mut Frame, app: &App, area: Rect) {
    let width = area.width;
    // Safety: RwLock is not poisoned — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let state = app.state.read().unwrap();

    let has_cluster = state.cluster.is_some();
    let has_gpus = state.gpus.is_some();
    let has_node_panel = has_cluster || has_gpus;

    if width < 80 {
        // Narrow: single column
        let mut constraints: Vec<Constraint> = vec![Constraint::Fill(1)]; // Worker Health
        if has_node_panel {
            constraints.push(Constraint::Fill(1));
        }
        constraints.push(Constraint::Fill(1)); // Throughput
        constraints.push(Constraint::Fill(1)); // Request Stats

        let rows = Layout::vertical(constraints).split(area);
        let mut i = 0;
        render_worker_health(f, &state, rows[i]);
        i += 1;
        if has_node_panel {
            render_node_status(f, &state, rows[i]);
            i += 1;
        }
        render_throughput_compact(f, &state, rows[i]);
        i += 1;
        render_request_stats(f, &state, rows[i]);
        return;
    }

    let columns =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);

    // Left column: Worker Health + Node/GPU (if available)
    let left = if has_node_panel {
        Layout::vertical([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(columns[0])
    } else {
        Layout::vertical([Constraint::Fill(1)]).split(columns[0])
    };

    // Right column: Throughput sparkline + Request Stats
    let right =
        Layout::vertical([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(columns[1]);

    // Left panels
    render_worker_health(f, &state, left[0]);
    if has_node_panel {
        render_node_status(f, &state, left[1]);
    }

    // Right panels
    if width < 100 {
        render_throughput_compact(f, &state, right[0]);
    } else {
        render_throughput(f, &state, right[0]);
    }
    render_request_stats(f, &state, right[1]);
}

fn render_worker_health(f: &mut Frame, state: &crate::state::GatewayState, area: Rect) {
    let block = theme::panel(" WORKER HEALTH ");

    let lines = if let Some(ref w) = state.workers {
        // Group workers: external by provider name, local by model name
        let mut groups: Vec<(String, Vec<&crate::client::WorkerInfo>)> = Vec::new();

        for worker in &w.workers {
            let group_name = if worker.runtime_type == "external" {
                // Use provider name from URL (e.g. "OpenAI" from "api.openai.com")
                let host = worker
                    .url
                    .trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .split('/')
                    .next()
                    .unwrap_or("");
                if host.contains("openai") {
                    "OpenAI".to_string()
                } else if host.contains("anthropic") {
                    "Anthropic".to_string()
                } else if host.contains("x.ai") {
                    "xAI".to_string()
                } else if host.contains("googleapis") {
                    "Gemini".to_string()
                } else {
                    format!("External ({host})")
                }
            } else {
                // Local: group by first model name
                worker
                    .models
                    .first()
                    .map(|m| m.id.clone())
                    .unwrap_or_else(|| "No model".to_string())
            };

            if let Some(group) = groups.iter_mut().find(|(name, _)| name == &group_name) {
                group.1.push(worker);
            } else {
                groups.push((group_name, vec![worker]));
            }
        }

        let mut lines: Vec<Line> = Vec::new();
        for (group_name, workers) in groups {
            // Group header
            lines.push(Line::from(Span::styled(
                group_name,
                ratatui::style::Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            )));

            // Worker details indented
            for worker in workers {
                let (dot_color, status) = if worker.is_healthy {
                    (theme::GREEN, "healthy")
                } else {
                    (theme::RED, "unhealthy")
                };
                let status_style = ratatui::style::Style::default()
                    .fg(dot_color)
                    .add_modifier(ratatui::style::Modifier::BOLD);

                let host = worker
                    .url
                    .trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .trim_start_matches("grpc://")
                    .trim_end_matches('/')
                    .split('/')
                    .next()
                    .unwrap_or(&worker.url);

                let rt = if worker.runtime_type.is_empty() {
                    "unknown"
                } else {
                    &worker.runtime_type
                };
                let conn = if worker.connection_mode.is_empty() {
                    ""
                } else {
                    &worker.connection_mode
                };

                lines.push(Line::from(vec![
                    Span::styled("  ● ", ratatui::style::Style::default().fg(dot_color)),
                    Span::styled(format!("{host} "), theme::text()),
                    Span::styled(format!("{rt} "), theme::label()),
                    if conn.is_empty() {
                        Span::raw("")
                    } else {
                        Span::styled(format!("{conn} "), theme::label())
                    },
                    Span::styled(status, status_style),
                ]));
            }
        }
        if lines.is_empty() {
            vec![Line::styled("No workers", theme::label())]
        } else {
            lines
        }
    } else {
        vec![Line::styled("No data", theme::label())]
    };

    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// Renders either GPU status (single-node) or cluster info (multi-node).
fn render_node_status(f: &mut Frame, state: &crate::state::GatewayState, area: Rect) {
    if let Some(ref gpus) = state.gpus {
        render_gpu_status(f, gpus, area);
    } else if let Some(ref c) = state.cluster {
        render_cluster(f, c, area);
    }
}

fn render_gpu_status(f: &mut Frame, gpus: &[crate::state::GpuInfo], area: Rect) {
    let block = theme::panel(" GPUs ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines: Vec<Line> = gpus
        .iter()
        .map(|gpu| {
            let mem_ratio = if gpu.memory_total_mb > 0 {
                gpu.memory_used_mb as f64 / gpu.memory_total_mb as f64
            } else {
                0.0
            };
            let util_ratio = gpu.utilization_pct as f64 / 100.0;

            let color = theme::severity(util_ratio);
            let bar_width = (inner.width / 4).max(6) as usize;
            let (filled, empty, _) = sparkline::gauge_bar(mem_ratio, bar_width);

            let mem_gb_used = gpu.memory_used_mb as f64 / 1024.0;
            let mem_gb_total = gpu.memory_total_mb as f64 / 1024.0;

            // Shorten GPU name (e.g. "NVIDIA A100-SXM4-80GB" → "A100-80GB")
            let short_name = shorten_gpu_name(&gpu.name);

            Line::from(vec![
                Span::styled(format!("GPU{} ", gpu.index), theme::label()),
                Span::styled(format!("{short_name:<12} "), theme::text()),
                Span::styled(filled, ratatui::style::Style::default().fg(color)),
                Span::styled(
                    empty,
                    ratatui::style::Style::default().fg(theme::TEXT_MUTED),
                ),
                Span::styled(
                    format!(" {mem_gb_used:.1}/{mem_gb_total:.0}G "),
                    theme::label(),
                ),
                Span::styled(
                    format!("{}% ", gpu.utilization_pct),
                    ratatui::style::Style::default().fg(color),
                ),
                Span::styled(format!("{}°C", gpu.temperature_c), theme::label()),
            ])
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

fn shorten_gpu_name(name: &str) -> String {
    // Strip common prefixes
    let name = name
        .trim_start_matches("NVIDIA ")
        .trim_start_matches("Tesla ")
        .trim_start_matches("GeForce ");
    // Truncate if too long (char-safe)
    if name.chars().count() > 12 {
        name.chars().take(12).collect()
    } else {
        name.to_string()
    }
}

fn render_cluster(f: &mut Frame, c: &crate::client::ClusterStatusResponse, area: Rect) {
    let block = theme::panel(" CLUSTER ");

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Node:  ", theme::label()),
            Span::styled(c.node_name.as_deref().unwrap_or("unknown"), theme::text()),
        ]),
        Line::from(vec![
            Span::styled("Size:  ", theme::label()),
            Span::styled(
                c.cluster_size
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".into()),
                theme::text(),
            ),
        ]),
    ];

    if let Some(ref stores) = c.stores {
        lines.push(Line::from(""));
        lines.push(Line::styled("Stores:", theme::label()));
        for store in stores {
            let color = if store.healthy {
                theme::GREEN
            } else {
                theme::RED
            };
            lines.push(Line::from(vec![
                Span::styled("  ● ", ratatui::style::Style::default().fg(color)),
                Span::styled(&store.name, theme::text()),
            ]));
        }
    }

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_throughput(f: &mut Frame, state: &crate::state::GatewayState, area: Rect) {
    let block = theme::panel(" THROUGHPUT ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.throughput_history.is_empty() && state.requests_per_sec_history.is_empty() {
        f.render_widget(
            Paragraph::new(Line::styled("No data", theme::label())),
            inner,
        );
        return;
    }

    // Header: latest values
    let rps = state
        .requests_per_sec_history
        .back()
        .copied()
        .unwrap_or(0.0);
    let in_tps = state.input_tps_history.back().copied().unwrap_or(0.0);
    let out_tps = state.output_tps_history.back().copied().unwrap_or(0.0);
    let total_tps = in_tps + out_tps;

    let mut header_spans = vec![
        Span::styled("req/s: ", theme::label()),
        Span::styled(format!("{rps:.1}"), theme::text().fg(theme::GREEN)),
    ];
    if total_tps > 0.0 {
        header_spans.extend([
            Span::styled("  tok/s: ", theme::label()),
            Span::styled(format!("{total_tps:.0}"), theme::text().fg(theme::GREEN)),
            Span::styled(
                format!("  (in:{in_tps:.0} out:{out_tps:.0})"),
                theme::label(),
            ),
        ]);
    }

    let header_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: 1,
    };
    f.render_widget(Paragraph::new(Line::from(header_spans)), header_area);
    let header_height = 1u16;

    // Sparkline area (total tok/s)
    let sparkline_start = inner.y + header_height;
    let remaining = inner.height.saturating_sub(header_height + 1);
    if remaining > 0 {
        let sparkline_area = Rect {
            x: inner.x,
            y: sparkline_start,
            width: inner.width,
            height: remaining,
        };
        sparkline::render_sparkline(f, &state.throughput_history, theme::GREEN, sparkline_area);
    }

    // Time labels
    if inner.height >= header_height + 2 {
        let label_area = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(1),
            width: inner.width,
            height: 1,
        };
        let padding = " ".repeat(label_area.width.saturating_sub(7) as usize);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("-60s", theme::label()),
                Span::raw(padding),
                Span::styled("now", theme::label()),
            ])),
            label_area,
        );
    }
}

fn render_throughput_compact(f: &mut Frame, state: &crate::state::GatewayState, area: Rect) {
    let block = theme::panel(" THROUGHPUT ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rps = state
        .requests_per_sec_history
        .back()
        .copied()
        .unwrap_or(0.0);
    if state.requests_per_sec_history.is_empty() && state.throughput_history.is_empty() {
        f.render_widget(
            Paragraph::new(Line::styled("No data", theme::label())),
            inner,
        );
        return;
    }

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Latest: ", theme::label()),
            Span::styled(format!("{rps:.1} req/s"), theme::text().fg(theme::GREEN)),
        ])),
        inner,
    );
}

fn render_request_stats(f: &mut Frame, state: &crate::state::GatewayState, area: Rect) {
    let block = theme::panel(" REQUEST STATS ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let avg_latency = state.avg_latency_history.back().copied().unwrap_or(0.0);
    let connections = state.active_connections;
    let inflight = state.inflight_requests;

    // Format latency nicely
    let latency_str = if avg_latency >= 1.0 {
        format!("{avg_latency:.2}s")
    } else {
        format!("{:.0}ms", avg_latency * 1000.0)
    };
    let latency_color = if avg_latency > 5.0 {
        theme::RED
    } else if avg_latency > 1.0 {
        theme::YELLOW
    } else {
        theme::GREEN
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Avg Latency    ", theme::label()),
            Span::styled(
                &latency_str,
                ratatui::style::Style::default().fg(latency_color),
            ),
        ]),
        Line::from(vec![
            Span::styled("Connections    ", theme::label()),
            Span::styled(connections.to_string(), theme::text()),
        ]),
        Line::from(vec![
            Span::styled("In-flight      ", theme::label()),
            Span::styled(inflight.to_string(), theme::text()),
        ]),
    ];

    // Latency sparkline if we have history
    if state.avg_latency_history.len() > 1 && inner.height > 5 {
        lines.push(Line::from(""));
        let sparkline_area = Rect {
            x: inner.x,
            y: inner.y + 4,
            width: inner.width,
            height: inner.height.saturating_sub(5),
        };
        f.render_widget(Paragraph::new(lines), inner);
        sparkline::render_sparkline(f, &state.avg_latency_history, theme::YELLOW, sparkline_area);

        // Time label
        if inner.height > 5 {
            let label_area = Rect {
                x: inner.x,
                y: inner.y + inner.height.saturating_sub(1),
                width: inner.width,
                height: 1,
            };
            let padding = " ".repeat(label_area.width.saturating_sub(7) as usize);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("-60s", theme::label()),
                    Span::raw(padding),
                    Span::styled("now", theme::label()),
                ])),
                label_area,
            );
        }
    } else {
        f.render_widget(Paragraph::new(lines), inner);
    }
}
