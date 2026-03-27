use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::theme;
use crate::{
    app::App,
    types::{InputMode, View},
};

pub fn render_footer(f: &mut Frame, app: &App, area: Rect) {
    let [hints_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    // Key hints — context-sensitive
    let hints = match app.input_mode {
        InputMode::Normal => match app.view {
            View::Workers => {
                vec![
                    hint("q", "quit"),
                    hint("1-7", "view"),
                    hint("j/k", "nav"),
                    hint("/", "filter"),
                    hint(":", "cmd"),
                    hint("a", "add"),
                    hint("d", "delete"),
                    hint("?", "help"),
                ]
            }
            _ => {
                vec![
                    hint("q", "quit"),
                    hint("1-7", "view"),
                    hint("/", "filter"),
                    hint(":", "cmd"),
                    hint("?", "help"),
                ]
            }
        },
        InputMode::Filter | InputMode::Command => {
            vec![hint("Enter", "submit"), hint("Esc", "cancel")]
        }
    };

    let line = Line::from(
        hints
            .into_iter()
            .flat_map(|(key, desc)| {
                vec![
                    Span::styled(
                        format!(" {key} "),
                        Style::default().fg(theme::ACCENT).bg(theme::PANEL_BG),
                    ),
                    Span::styled(format!("{desc} "), Style::default().fg(theme::TEXT_MUTED)),
                ]
            })
            .collect::<Vec<_>>(),
    );
    f.render_widget(Paragraph::new(line), hints_area);

    // Status message
    if let Some(msg) = &app.status_message {
        let style = if msg.starts_with("Error") {
            Style::default().fg(theme::RED)
        } else {
            Style::default().fg(theme::GREEN)
        };
        f.render_widget(Paragraph::new(msg.as_str()).style(style), status_area);
    }
}

fn hint(key: &str, desc: &str) -> (String, String) {
    (key.to_string(), desc.to_string())
}
