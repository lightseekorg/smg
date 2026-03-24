use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::theme;
use crate::app::{App, LogLevel, LogSubTab};

pub fn render_logs(f: &mut Frame, app: &App, area: Rect) {
    // Split: content (fill) + footer (1 line)
    let [content_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);

    // Render content based on active sub-tab
    match &app.log_sub_tab {
        LogSubTab::Tui => render_tui_logs(f, app, content_area),
        LogSubTab::Gateway => {
            render_file_log(f, "/tmp/smg-gateway.log", "SMG Gateway", app, content_area);
        }
        LogSubTab::Worker(port) => {
            let path = format!("/tmp/smg-worker-{port}.log");
            let label = worker_tab_label(app, port);
            render_file_log(f, &path, &label, app, content_area);
        }
    }

    // Render footer: sub-tabs (left) + key hints (right)
    render_footer(f, app, footer_area);
}

fn render_footer(f: &mut Frame, app: &App, area: Rect) {
    let [tabs_area, hints_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(30)]).areas(area);

    // Left: sub-tab selector
    let mut spans = vec![
        Span::styled(" ", Style::default()),
        tab_span("a:TUI", matches!(app.log_sub_tab, LogSubTab::Tui)),
        Span::styled("  ", Style::default()),
        tab_span("b:SMG", matches!(app.log_sub_tab, LogSubTab::Gateway)),
    ];

    let worker_tabs = app.worker_log_tabs();
    let worker_labels: Vec<(String, bool)> = worker_tabs
        .iter()
        .map(|(label, port)| {
            let active = matches!(&app.log_sub_tab, LogSubTab::Worker(p) if p == port);
            (format!("w:{label}"), active)
        })
        .collect();
    for (label, active) in &worker_labels {
        spans.push(Span::styled("  ", Style::default()));
        spans.push(tab_span(label, *active));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), tabs_area);

    // Right: key hints
    let hints = Line::from(vec![
        Span::styled("j/k", Style::default().fg(theme::ACCENT)),
        Span::styled(" scroll  ", theme::label()),
        Span::styled("G", Style::default().fg(theme::ACCENT)),
        Span::styled(" bottom  ", theme::label()),
        Span::styled("w", Style::default().fg(theme::ACCENT)),
        Span::styled(" worker", theme::label()),
    ]);
    f.render_widget(
        Paragraph::new(hints).alignment(ratatui::layout::Alignment::Right),
        hints_area,
    );
}

fn tab_span(label: &str, active: bool) -> Span<'_> {
    if active {
        Span::styled(
            label,
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )
    } else {
        Span::styled(label, theme::label())
    }
}

fn worker_tab_label(app: &App, port: &str) -> String {
    for (label, p) in app.worker_log_tabs() {
        if p == port {
            return label;
        }
    }
    format!("worker-{port}")
}

fn render_tui_logs(f: &mut Frame, app: &App, area: Rect) {
    let total = app.log_entries.len();
    let title = format!(" TUI Logs ({total}) ");
    let block = Block::default()
        .title(title)
        .title_style(theme::title())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::BG));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.log_entries.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No log entries yet.",
                theme::label(),
            ))),
            inner,
        );
        return;
    }

    let lines: Vec<Line> = app
        .log_entries
        .iter()
        .map(|entry| {
            let time = entry.timestamp.format("%H:%M:%S").to_string();
            let (level_str, level_color) = match entry.level {
                LogLevel::Info => ("INFO", theme::GREEN),
                LogLevel::Warn => ("WARN", theme::YELLOW),
                LogLevel::Error => ("ERR ", theme::RED),
            };

            Line::from(vec![
                Span::styled(format!("{time} "), Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    format!("{level_str} "),
                    Style::default()
                        .fg(level_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(&entry.message, theme::text()),
            ])
        })
        .collect();

    let total_lines = lines.len() as u16;
    let visible = inner.height;
    let max_scroll = total_lines.saturating_sub(visible);
    let scroll = if app.log_scroll >= max_scroll {
        max_scroll
    } else {
        app.log_scroll
    };

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        inner,
    );
}

fn render_file_log(f: &mut Frame, path: &str, label: &str, app: &App, area: Rect) {
    // Read only the tail of the file to avoid blocking the render loop on large files
    use std::io::{Read, Seek, SeekFrom};
    let content = match std::fs::File::open(path) {
        Ok(mut file) => {
            // Read at most the last 64KB to avoid blocking on large files
            const TAIL_BYTES: u64 = 64 * 1024;
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);
            if len > TAIL_BYTES {
                let _ = file.seek(SeekFrom::Start(len - TAIL_BYTES));
            }
            let mut raw = Vec::new();
            if file.read_to_end(&mut raw).is_err() {
                raw.clear();
            }
            let buf = String::from_utf8_lossy(&raw).into_owned();
            // If we seeked into the middle, skip the first partial line
            if len > TAIL_BYTES {
                buf.split_once('\n')
                    .map(|(_, rest)| rest.to_string())
                    .unwrap_or(buf)
            } else {
                buf
            }
        }
        Err(_) => {
            let block = Block::default()
                .title(format!(" {label} "))
                .title_style(theme::title())
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BORDER))
                .style(Style::default().bg(theme::BG));
            let inner = block.inner(area);
            f.render_widget(block, area);
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("Log file not found: {path}"),
                    theme::label(),
                ))),
                inner,
            );
            return;
        }
    };

    // Take last N lines
    let all_lines: Vec<&str> = content.lines().collect();
    let max_lines = 500;
    let start = all_lines.len().saturating_sub(max_lines);
    let lines: Vec<Line> = all_lines[start..]
        .iter()
        .map(|line| {
            // Strip ANSI escape codes for cleaner display
            let clean = strip_ansi(line);
            let color = if clean.contains("ERROR") || clean.contains("Error") {
                theme::RED
            } else if clean.contains("WARN") || clean.contains("Warning") {
                theme::YELLOW
            } else if clean.contains("INFO") {
                theme::GREEN
            } else {
                theme::TEXT
            };
            Line::from(Span::styled(clean, Style::default().fg(color)))
        })
        .collect();

    let total_count = lines.len();
    let title = format!(" {label} ({total_count} lines) ");
    let block = Block::default()
        .title(title)
        .title_style(theme::title())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::BG));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let total_lines = lines.len() as u16;
    let visible = inner.height;
    let max_scroll = total_lines.saturating_sub(visible);
    let scroll = if app.log_scroll >= max_scroll {
        max_scroll
    } else {
        app.log_scroll
    };

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        inner,
    );
}

/// Strip ANSI escape codes from a string.
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until we find the end of the escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if nc.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}
