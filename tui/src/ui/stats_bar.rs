use ratatui::{
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::{sparkline, theme};
use crate::state::GatewayState;

pub fn render_stats_bar(f: &mut Frame, state: &GatewayState, area: Rect) {
    let bg = Style::default().bg(theme::STATS_BG);
    f.render_widget(ratatui::widgets::Block::default().style(bg), area);

    // Row 0: logo (left) + connection status (right)  — 1 line
    // Row 1–3: stats cards                             — 3 lines
    // Row 4: separator line                            — 1 line
    let rows = Layout::vertical([
        Constraint::Length(1), // logo row
        Constraint::Length(3), // stats cards
        Constraint::Length(1), // separator
    ])
    .split(area);

    render_logo_row(f, state, rows[0], bg);
    render_stats_cards(f, state, rows[1], bg);
    render_separator(f, rows[2]);
}

fn render_logo_row(f: &mut Frame, state: &GatewayState, area: Rect, bg: Style) {
    let cols = Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)]).split(area);

    // Left: logo
    let logo = Line::from(vec![
        Span::styled("⎔ ", theme::label()),
        Span::styled("SMG", theme::title()),
    ]);
    f.render_widget(Paragraph::new(logo).style(bg), cols[0]);

    // Right: connection status
    let conn = if state.connected {
        Line::from(Span::styled(
            "● connected",
            Style::default().fg(theme::GREEN),
        ))
    } else {
        Line::from(Span::styled(
            "● disconnected",
            Style::default().fg(theme::RED),
        ))
    };
    f.render_widget(
        Paragraph::new(conn).alignment(Alignment::Right).style(bg),
        cols[1],
    );
}

fn render_stats_cards(f: &mut Frame, state: &GatewayState, area: Rect, bg: Style) {
    let cols = Layout::horizontal([
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
    ])
    .split(area);

    // Card 1: Workers
    let (total, healthy) = state
        .workers
        .as_ref()
        .map(|w| {
            let h = w.workers.iter().filter(|w| w.is_healthy).count();
            (w.total, h)
        })
        .unwrap_or((0, 0));

    let unhealthy = total.saturating_sub(healthy);
    let health_text = if !state.connected {
        ("--".to_string(), theme::TEXT_MUTED)
    } else if total == 0 {
        ("no workers".to_string(), theme::TEXT_MUTED)
    } else if unhealthy == 0 {
        ("all healthy".to_string(), theme::GREEN)
    } else {
        (format!("{unhealthy} unhealthy"), theme::RED)
    };

    let workers_value = if state.connected {
        total.to_string()
    } else {
        "--".to_string()
    };

    render_card(
        f,
        cols[0],
        bg,
        "WORKERS",
        &workers_value,
        Some((&health_text.0, health_text.1)),
    );

    // Card 2: Circuit Breakers
    let cb = &state.circuit_breakers;
    let cb_total = cb.closed + cb.open;
    let cb_value = if state.connected {
        cb_total.to_string()
    } else {
        "--".to_string()
    };

    let cb_detail = if !state.connected {
        ("--".to_string(), theme::TEXT_MUTED)
    } else if cb.open > 0 {
        (format!("{} open", cb.open), theme::RED)
    } else if cb.closed > 0 {
        ("all closed".to_string(), theme::GREEN)
    } else {
        ("--".to_string(), theme::TEXT_MUTED)
    };

    render_card(
        f,
        cols[1],
        bg,
        "CIRCUIT BREAKERS",
        &cb_value,
        Some((&cb_detail.0, cb_detail.1)),
    );

    // Card 3: REQ/S
    let rps_value = if state.connected {
        let rps = state
            .requests_per_sec_history
            .back()
            .copied()
            .unwrap_or(0.0);
        format!("{rps:.1}")
    } else {
        "--".to_string()
    };
    let inflight = if state.connected {
        format!("{} in-flight", state.inflight_requests)
    } else {
        "--".to_string()
    };
    render_card(
        f,
        cols[2],
        bg,
        "REQ/S",
        &rps_value,
        Some((&inflight, theme::TEXT_MUTED)),
    );

    // Card 4: AVG LATENCY with gauge bar
    let avg_latency = state.avg_latency_history.back().copied().unwrap_or(0.0);
    let (latency_value, latency_color) = if state.connected {
        let latency_str = if avg_latency >= 1.0 {
            format!("{avg_latency:.2}s")
        } else {
            format!("{:.0}ms", avg_latency * 1000.0)
        };
        let color = if avg_latency > 5.0 {
            theme::RED
        } else if avg_latency > 1.0 {
            theme::YELLOW
        } else {
            theme::GREEN
        };
        (latency_str, color)
    } else {
        ("--".to_string(), theme::TEXT_MUTED)
    };

    render_latency_card(
        f,
        cols[3],
        bg,
        &latency_value,
        latency_color,
        avg_latency,
        state.connected,
    );
}

fn render_card(
    f: &mut Frame,
    area: Rect,
    bg: Style,
    label: &str,
    value: &str,
    detail: Option<(&str, ratatui::style::Color)>,
) {
    // 3 rows: label, big number, detail
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    // Label (centered, muted)
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(label, theme::label())))
            .alignment(Alignment::Center)
            .style(bg),
        rows[0],
    );

    // Big number (centered, bold)
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            value,
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center)
        .style(bg),
        rows[1],
    );

    // Detail line (centered)
    if let Some((text, color)) = detail {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(text, Style::default().fg(color))))
                .alignment(Alignment::Center)
                .style(bg),
            rows[2],
        );
    }
}

fn render_latency_card(
    f: &mut Frame,
    area: Rect,
    bg: Style,
    latency_value: &str,
    latency_color: ratatui::style::Color,
    avg_latency_secs: f64,
    connected: bool,
) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    // Label
    f.render_widget(
        Paragraph::new(Line::from(Span::styled("AVG LATENCY", theme::label())))
            .alignment(Alignment::Center)
            .style(bg),
        rows[0],
    );

    // Big value
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            latency_value,
            Style::default()
                .fg(latency_color)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center)
        .style(bg),
        rows[1],
    );

    // Gauge bar (0-5s scale)
    if connected {
        let ratio = (avg_latency_secs / 5.0).clamp(0.0, 1.0);
        let bar_width = (area.width / 3).max(6) as usize;
        let (filled, empty, _pct) = sparkline::gauge_bar(ratio, bar_width);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(filled, Style::default().fg(latency_color)),
                Span::styled(empty, Style::default().fg(theme::TEXT_MUTED)),
            ]))
            .alignment(Alignment::Center)
            .style(bg),
            rows[2],
        );
    }
}

fn render_separator(f: &mut Frame, area: Rect) {
    let line = "─".repeat(area.width as usize);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            line,
            Style::default().fg(theme::BORDER),
        ))),
        area,
    );
}
