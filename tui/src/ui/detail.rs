use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::{sparkline, theme};
use crate::{app::App, client::WorkerInfo};

pub fn render_detail(f: &mut Frame, app: &App, worker: &WorkerInfo, area: Rect) {
    let title = format!(" Worker: {} ", truncate_str(&worker.id, 20));
    let block = theme::panel(&title);
    f.render_widget(block, area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    if inner.width < 3 || inner.height < 1 {
        return;
    }

    // 2-column layout: config (left) + stats (right)
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(inner);

    // Safety: RwLock is not poisoned — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let state = app.state.read().unwrap();

    // Left: Config + Models
    render_config(f, worker, left_area);

    // Right: Live stats
    let worker_load = state
        .loads
        .as_ref()
        .and_then(|l| l.workers.iter().find(|wl| wl.worker == worker.url));
    let worker_rps = state.worker_rps.get(&worker.url).copied().unwrap_or(0.0);
    let cb_has_open = state.circuit_breakers.open > 0;
    render_stats(f, worker, worker_load, worker_rps, cb_has_open, right_area);
}

fn render_config(f: &mut Frame, worker: &WorkerInfo, area: Rect) {
    if area.height == 0 {
        return;
    }

    let health_color = if worker.is_healthy {
        theme::GREEN
    } else {
        theme::RED
    };
    let health_text = if worker.is_healthy {
        "healthy"
    } else {
        "unhealthy"
    };

    let short_url = worker
        .url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("grpc://")
        .trim_end_matches('/');

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled("Config", Style::default().fg(theme::ACCENT))),
        line_kv("URL", short_url),
        line_kv("Runtime", &worker.runtime_type),
        line_kv("Mode", &worker.connection_mode),
        Line::from(vec![
            Span::styled("Health: ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(health_text, Style::default().fg(health_color)),
        ]),
    ];

    // Models
    if !worker.models.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("Models ({})", worker.models.len()),
            Style::default().fg(theme::ACCENT),
        )));
        let max_models = area.height.saturating_sub(lines.len() as u16) as usize;
        for (i, model) in worker.models.iter().enumerate() {
            if i >= max_models {
                lines.push(Line::from(Span::styled(
                    format!("  +{} more", worker.models.len() - i),
                    Style::default().fg(theme::TEXT_MUTED),
                )));
                break;
            }
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(&model.id, Style::default().fg(theme::TEXT)),
            ]));
        }
    }

    f.render_widget(Paragraph::new(lines), area);
}

fn render_stats(
    f: &mut Frame,
    worker: &WorkerInfo,
    worker_load: Option<&crate::client::WorkerLoad>,
    worker_rps: f64,
    cb_has_open: bool,
    area: Rect,
) {
    if area.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        "Stats",
        Style::default().fg(theme::ACCENT),
    ))];

    let is_http_local = worker.connection_mode == "http"
        && (worker.runtime_type == "sglang" || worker.runtime_type == "vllm");

    // Req/s from Prometheus (all workers)
    lines.push(line_kv("Req/s", &format!("{worker_rps:.1}")));

    // Circuit breaker — use actual breaker state from Prometheus metrics
    // If worker is unhealthy, show open; if there are open breakers globally and worker unhealthy, show open
    let cb_is_open = !worker.is_healthy && cb_has_open;
    let cb_label = if cb_is_open { "open" } else { "closed" };
    let cb_color = if cb_is_open { theme::RED } else { theme::GREEN };
    lines.push(Line::from(vec![
        Span::styled(
            "Circuit: ".to_string(),
            Style::default().fg(theme::TEXT_MUTED),
        ),
        Span::styled(cb_label.to_string(), Style::default().fg(cb_color)),
    ]));

    // HTTP sglang/vllm: show detailed load stats
    if is_http_local {
        if let Some(wl) = worker_load {
            if let Some(ref details) = wl.details {
                let bar_width = (area.width / 3).max(6) as usize;
                for snap in &details.loads {
                    lines.push(Line::from(""));

                    // Running requests
                    let running_ratio = if snap.max_running_requests > 0 {
                        snap.num_running_reqs as f64 / snap.max_running_requests as f64
                    } else {
                        0.0
                    }
                    .clamp(0.0, 1.0);
                    let run_color = theme::severity(running_ratio);
                    let (filled, empty, pct) = sparkline::gauge_bar(running_ratio, bar_width);
                    lines.push(line_kv(
                        "Running",
                        &format!(
                            "{} / {} max",
                            snap.num_running_reqs, snap.max_running_requests
                        ),
                    ));
                    lines.push(Line::from(vec![
                        Span::styled("  ".to_string(), Style::default()),
                        Span::styled(filled, Style::default().fg(run_color)),
                        Span::styled(empty, Style::default().fg(theme::TEXT_MUTED)),
                        Span::styled(format!(" {pct}%"), Style::default().fg(theme::TEXT_MUTED)),
                    ]));

                    // Waiting
                    lines.push(line_kv("Waiting", &snap.num_waiting_reqs.to_string()));

                    // KV Cache usage
                    let token_ratio = snap.token_usage.clamp(0.0, 1.0);
                    let tok_color = theme::severity(token_ratio);
                    let (filled, empty, pct) = sparkline::gauge_bar(token_ratio, bar_width);
                    lines.push(line_kv(
                        "KV Cache",
                        &format!(
                            "{} / {} ({:.1}%)",
                            snap.num_used_tokens,
                            snap.max_total_num_tokens,
                            snap.token_usage * 100.0
                        ),
                    ));
                    lines.push(Line::from(vec![
                        Span::styled("  ".to_string(), Style::default()),
                        Span::styled(filled, Style::default().fg(tok_color)),
                        Span::styled(empty, Style::default().fg(theme::TEXT_MUTED)),
                        Span::styled(format!(" {pct}%"), Style::default().fg(theme::TEXT_MUTED)),
                    ]));

                    if snap.gen_throughput > 0.0 {
                        lines.push(line_kv(
                            "Gen Throughput",
                            &format!("{:.1} tok/s", snap.gen_throughput),
                        ));
                    }
                }
            }
        }
    } else if let Some(wl) = worker_load {
        // gRPC/external: show load value
        if wl.load >= 0 {
            lines.push(line_kv("Load", &format!("{}", wl.load)));
        }
    }

    f.render_widget(Paragraph::new(lines), area);
}

fn line_kv(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key}: "), Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(value.to_string(), Style::default().fg(theme::TEXT)),
    ])
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}
