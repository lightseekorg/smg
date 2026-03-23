use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::theme;
use crate::types::View;

pub fn render_tabs(f: &mut Frame, active: View, area: Rect) {
    let bg = Style::default().bg(theme::PANEL_BG);
    let tabs: Vec<Span> = View::all()
        .iter()
        .flat_map(|view| {
            let num = format!("{}", view.index());
            let label = view.label();
            let style = if *view == active {
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            } else {
                Style::default().fg(theme::TEXT_MUTED)
            };
            vec![
                Span::styled(format!("{num}:{label}"), style),
                Span::raw("  "),
            ]
        })
        .collect();
    f.render_widget(Paragraph::new(Line::from(tabs)).style(bg), area);
}
