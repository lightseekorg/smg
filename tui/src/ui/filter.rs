use ratatui::{layout::Rect, style::Style, widgets::Paragraph, Frame};

use super::theme;
use crate::{app::App, types::InputMode};

/// Render the filter/command input bar over the footer area.
pub fn render_filter(f: &mut Frame, app: &App, footer_area: Rect) {
    match app.input_mode {
        InputMode::Filter => {
            let text = format!("/{}", app.input_buffer);
            let paragraph =
                Paragraph::new(text).style(Style::default().fg(theme::YELLOW).bg(theme::BG));
            f.render_widget(paragraph, footer_area);
        }
        InputMode::Command => {
            let text = format!(":{}", app.input_buffer);
            let paragraph =
                Paragraph::new(text).style(Style::default().fg(theme::ACCENT).bg(theme::BG));
            f.render_widget(paragraph, footer_area);
        }
        InputMode::Normal => {}
    }
}
