use ratatui::{
    layout::{Constraint, Flex, Layout},
    style::Style,
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use super::theme;
use crate::app::App;

/// Render the delete-confirmation popup.
pub fn render_delete_dialog(f: &mut Frame, app: &App) {
    let Some(ref info) = app.confirm_delete else {
        return;
    };

    let area = f.area();

    // Center a 50×8 popup
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(8),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(50)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Confirm Delete ")
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::RED))
        .style(Style::default().bg(theme::PANEL_BG));

    let text = format!(
        "Delete worker?\n\nID:  {}\nURL: {}\n\n[y] confirm  [n/Esc] cancel",
        info.0, info.1,
    );

    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(theme::TEXT))
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, popup);
}

/// Render the flush-cache confirmation popup.
pub fn render_flush_dialog(f: &mut Frame, app: &App) {
    let Some(ref info) = app.confirm_flush else {
        return;
    };

    let area = f.area();

    // Center a 50×8 popup
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(8),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(50)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Flush Cache? ")
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let text = format!(
        "Flush cache for worker?\n\nID:  {}\nURL: {}\n\n[y] confirm  [n/Esc] cancel",
        info.0, info.1,
    );

    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(theme::TEXT))
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, popup);
}
