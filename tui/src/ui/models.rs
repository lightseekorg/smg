use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Cell, Row, Table, TableState},
    Frame,
};

use super::theme;
use crate::app::App;

pub fn render_models(f: &mut Frame, app: &App, area: Rect) {
    // Safety: RwLock is not poisoned — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let state = app.state.read().unwrap();

    let block = theme::panel(" Models ");

    let header = Row::new(vec!["ID", "NAME", "WORKERS", "CREATED"]).style(
        Style::default()
            .fg(theme::ACCENT)
            .bg(theme::PANEL_BG)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = if let Some(ref models) = state.models {
        models
            .data
            .iter()
            .map(|m| {
                // Count workers serving this model
                let worker_count = state
                    .workers
                    .as_ref()
                    .map(|w| {
                        w.workers
                            .iter()
                            .filter(|wi| wi.models.iter().any(|mr| mr.id == m.id))
                            .count()
                    })
                    .unwrap_or(0);

                // created_at is already a human-readable string
                let created = if m.created_at.is_empty() {
                    "--".to_string()
                } else {
                    m.created_at.clone()
                };

                Row::new(vec![
                    Cell::from(m.id.clone()).style(theme::text()),
                    Cell::from(m.display_name.clone()).style(theme::label()),
                    Cell::from(worker_count.to_string()).style(theme::text()),
                    Cell::from(created).style(theme::label()),
                ])
                .style(Style::default().bg(theme::BG))
            })
            .collect()
    } else {
        vec![]
    };

    let row_count = rows.len();

    let widths = [
        ratatui::layout::Constraint::Percentage(40),
        ratatui::layout::Constraint::Percentage(25),
        ratatui::layout::Constraint::Percentage(15),
        ratatui::layout::Constraint::Percentage(20),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(
            Style::default()
                .fg(theme::TEXT)
                .bg(theme::BORDER)
                .add_modifier(Modifier::BOLD),
        );

    let mut table_state = TableState::default();
    if row_count > 0 {
        table_state.select(Some(app.selected_index.min(row_count.saturating_sub(1))));
    }

    f.render_stateful_widget(table, area, &mut table_state);
    drop(state);
}
