use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Cell, Row, Table, TableState},
    Frame,
};

use super::{detail, theme};
use crate::{app::App, client::WorkerInfo};

pub fn render_workers(f: &mut Frame, app: &App, area: Rect) {
    let block = theme::panel(" Workers ");

    // Split layout when detail panel is visible
    let (table_area, detail_area) = if app.show_detail {
        let split =
            Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).split(area);
        (split[0], Some(split[1]))
    } else {
        (area, None)
    };

    // Safety: RwLock is not poisoned — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let state = app.state.read().unwrap();

    let width = table_area.width;

    let filtered: Vec<WorkerInfo> = if let Some(ref wl) = state.workers {
        wl.workers
            .iter()
            .filter(|w| matches_filter(w, app.active_filter.as_ref()))
            .cloned()
            .collect()
    } else {
        vec![]
    };

    // Build per-worker load lookup from /get_loads
    let worker_loads: std::collections::HashMap<String, &crate::client::WorkerLoad> = state
        .loads
        .as_ref()
        .map(|l| l.workers.iter().map(|wl| (wl.worker.clone(), wl)).collect())
        .unwrap_or_default();
    let worker_rps = &state.worker_rps;

    // Build rows and table based on terminal width
    let (header, rows, widths): (Row, Vec<Row>, Vec<Constraint>) = if width < 80 {
        // Narrow: ID, Health, Running (3 columns)
        let header_cells = ["ID", "Health", "Running"].iter().map(|h| {
            Cell::from(*h).style(
                Style::default()
                    .fg(theme::ACCENT)
                    .bg(theme::PANEL_BG)
                    .add_modifier(Modifier::BOLD),
            )
        });
        let header = Row::new(header_cells).height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .map(|w| {
                let health_style = if w.is_healthy {
                    Style::default().fg(theme::GREEN)
                } else {
                    Style::default().fg(theme::RED)
                };
                let health_text = if w.is_healthy { "healthy" } else { "unhealthy" };

                let (running, _) = get_worker_load_info(&worker_loads, worker_rps, &w.url);
                Row::new(vec![
                    Cell::from(truncate(&w.id, 12)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(health_text).style(health_style),
                    Cell::from(running).style(Style::default().fg(theme::TEXT)),
                ])
                .style(Style::default().bg(theme::BG))
            })
            .collect();

        let widths = vec![
            Constraint::Fill(1),
            Constraint::Length(10),
            Constraint::Length(8),
        ];

        (header, rows, widths)
    } else if width < 100 {
        // Compact: ID, URL, Health, Running (4 columns)
        let header_cells = ["ID", "URL", "Health", "Running"].iter().map(|h| {
            Cell::from(*h).style(
                Style::default()
                    .fg(theme::ACCENT)
                    .bg(theme::PANEL_BG)
                    .add_modifier(Modifier::BOLD),
            )
        });
        let header = Row::new(header_cells).height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .map(|w| {
                let health_style = if w.is_healthy {
                    Style::default().fg(theme::GREEN)
                } else {
                    Style::default().fg(theme::RED)
                };
                let health_text = if w.is_healthy { "healthy" } else { "unhealthy" };

                let (running, _) = get_worker_load_info(&worker_loads, worker_rps, &w.url);
                Row::new(vec![
                    Cell::from(truncate(&w.id, 12)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(shorten_url(&w.url)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(health_text).style(health_style),
                    Cell::from(running).style(Style::default().fg(theme::TEXT)),
                ])
                .style(Style::default().bg(theme::BG))
            })
            .collect();

        let widths = vec![
            Constraint::Length(14),
            Constraint::Fill(1),
            Constraint::Length(10),
            Constraint::Length(8),
        ];

        (header, rows, widths)
    } else if width < 120 {
        // Medium: ID, URL, Runtime, Health, Running, Token Usage (6 columns)
        let header_cells = ["ID", "URL", "Runtime", "Health", "Running", "Tok Usage"]
            .iter()
            .map(|h| {
                Cell::from(*h).style(
                    Style::default()
                        .fg(theme::ACCENT)
                        .bg(theme::PANEL_BG)
                        .add_modifier(Modifier::BOLD),
                )
            });
        let header = Row::new(header_cells).height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .map(|w| {
                let health_style = if w.is_healthy {
                    Style::default().fg(theme::GREEN)
                } else {
                    Style::default().fg(theme::RED)
                };
                let health_text = if w.is_healthy { "healthy" } else { "unhealthy" };

                let (running, token_usage) =
                    get_worker_load_info(&worker_loads, worker_rps, &w.url);

                Row::new(vec![
                    Cell::from(truncate(&w.id, 12)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(shorten_url(&w.url)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(w.runtime_type.as_str())
                        .style(Style::default().fg(theme::TEXT_MUTED)),
                    Cell::from(health_text).style(health_style),
                    Cell::from(running).style(Style::default().fg(theme::TEXT)),
                    Cell::from(token_usage).style(Style::default().fg(theme::TEXT)),
                ])
                .style(Style::default().bg(theme::BG))
            })
            .collect();

        let widths = vec![
            Constraint::Length(14),
            Constraint::Fill(1),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
        ];

        (header, rows, widths)
    } else {
        // Full: all 8 columns
        let header_cells = [
            "ID",
            "URL",
            "Mode",
            "Runtime",
            "Models",
            "Health",
            "Running",
            "Tok Usage",
        ]
        .iter()
        .map(|h| {
            Cell::from(*h).style(
                Style::default()
                    .fg(theme::ACCENT)
                    .bg(theme::PANEL_BG)
                    .add_modifier(Modifier::BOLD),
            )
        });
        let header = Row::new(header_cells).height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .map(|w| {
                let health_style = if w.is_healthy {
                    Style::default().fg(theme::GREEN)
                } else {
                    Style::default().fg(theme::RED)
                };
                let health_text = if w.is_healthy { "healthy" } else { "unhealthy" };

                let model_names: String = w
                    .models
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let models_display = if model_names.is_empty() {
                    "*".to_string()
                } else {
                    model_names
                };

                let (running, token_usage) =
                    get_worker_load_info(&worker_loads, worker_rps, &w.url);

                Row::new(vec![
                    Cell::from(truncate(&w.id, 12)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(shorten_url(&w.url)).style(Style::default().fg(theme::TEXT)),
                    Cell::from(w.connection_mode.as_str())
                        .style(Style::default().fg(theme::TEXT_MUTED)),
                    Cell::from(w.runtime_type.as_str())
                        .style(Style::default().fg(theme::TEXT_MUTED)),
                    Cell::from(truncate(&models_display, 20))
                        .style(Style::default().fg(theme::TEXT)),
                    Cell::from(health_text).style(health_style),
                    Cell::from(running).style(Style::default().fg(theme::TEXT)),
                    Cell::from(token_usage).style(Style::default().fg(theme::TEXT)),
                ])
                .style(Style::default().bg(theme::BG))
            })
            .collect();

        let widths = vec![
            Constraint::Length(14),
            Constraint::Length(22),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(22),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(10),
        ];

        (header, rows, widths)
    };

    let row_count = rows.len();

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        );

    // Build TableState from app's selected_index
    let mut table_state = TableState::default();
    if row_count > 0 {
        table_state.select(Some(app.selected_index.min(row_count.saturating_sub(1))));
    }

    f.render_stateful_widget(table, table_area, &mut table_state);

    // Drop state before calling detail render (which re-acquires it)
    drop(state);

    if let Some(detail_area) = detail_area {
        let clamped = if row_count > 0 {
            app.selected_index.min(row_count.saturating_sub(1))
        } else {
            0
        };
        if let Some(worker) = filtered.get(clamped) {
            detail::render_detail(f, app, worker, detail_area);
        }
    }
}

fn matches_filter(worker: &WorkerInfo, filter: Option<&String>) -> bool {
    let Some(f) = filter else { return true };
    if f.is_empty() {
        return true;
    }
    let f_lower = f.to_lowercase();
    worker.id.to_lowercase().contains(&f_lower)
        || worker.url.to_lowercase().contains(&f_lower)
        || worker.worker_type.to_lowercase().contains(&f_lower)
        || worker.runtime_type.to_lowercase().contains(&f_lower)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

/// Shorten a URL: strip scheme, keep host:port
fn shorten_url(url: &str) -> String {
    url.trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("grpc://")
        .trim_end_matches('/')
        .split('/')
        .next()
        .unwrap_or(url)
        .to_string()
}

/// Get running reqs and token usage for a worker from /get_loads data.
/// Returns (running_reqs_str, token_usage_str).
fn get_worker_load_info(
    loads: &std::collections::HashMap<String, &crate::client::WorkerLoad>,
    worker_rps: &std::collections::HashMap<String, f64>,
    worker_url: &str,
) -> (String, String) {
    // HTTP sglang/vllm workers: aggregate across all loads (TP>1 workers have multiple entries)
    if let Some(wl) = loads.get(worker_url) {
        if let Some(ref details) = wl.details {
            if !details.loads.is_empty() {
                let total_running: i32 = details.loads.iter().map(|l| l.num_running_reqs).sum();
                let avg_usage: f64 = details.loads.iter().map(|l| l.token_usage).sum::<f64>()
                    / details.loads.len() as f64;
                let running = format!("{total_running}");
                let usage = format!("{:.1}%", avg_usage * 100.0);
                return (running, usage);
            }
        }
    }
    // gRPC/local workers: show req/s from Prometheus per-worker counts
    if worker_url.starts_with("grpc://") {
        let rps = worker_rps.get(worker_url).copied().unwrap_or(0.0);
        return (format!("{rps:.1} r/s"), "N/A".to_string());
    }
    // External workers
    if worker_url.starts_with("https://") {
        let rps = worker_rps.get(worker_url).copied().unwrap_or(0.0);
        return (format!("{rps:.1} r/s"), "N/A".to_string());
    }
    ("0".to_string(), "0.0%".to_string())
}
