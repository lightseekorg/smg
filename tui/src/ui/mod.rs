pub mod action_menu;
mod chat;
pub mod detail;
mod dialog;
mod filter;
mod footer;
mod help;
mod logs;
pub mod models;
mod pulse;
pub mod sparkline;
pub mod stats_bar;
pub mod tabs;
pub mod theme;
mod workers;

use ratatui::{
    layout::{Constraint, Layout},
    Frame,
};

use crate::{app::App, types::View};

/// Root render function — called once per frame.
pub fn render(f: &mut Frame, app: &App) {
    // Safety: RwLock is not poisoned — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let state = app.state.read().unwrap();

    // Layout: stats_bar (5) + tabs (1) + content (fill) + footer (2)
    let [stats_area, tabs_area, content_area, footer_area] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(2),
    ])
    .areas(f.area());

    // Background
    f.render_widget(
        ratatui::widgets::Block::default().style(ratatui::style::Style::default().bg(theme::BG)),
        f.area(),
    );

    stats_bar::render_stats_bar(f, &state, stats_area);
    tabs::render_tabs(f, app.view, tabs_area);

    drop(state);

    match app.view {
        View::Pulse => pulse::render_pulse(f, app, content_area),
        View::Workers => workers::render_workers(f, app, content_area),
        View::Chat => chat::render_chat(f, app, content_area),
        View::Logs => logs::render_logs(f, app, content_area),
        View::Benchmark | View::Traffic | View::Mesh => {
            render_placeholder(f, app.view, content_area);
        }
    }

    footer::render_footer(f, app, footer_area);

    // Overlays (rendered last)
    if app.show_help {
        help::render_help(f, app.view);
    }
    if app.confirm_delete.is_some() {
        dialog::render_delete_dialog(f, app);
    }
    if app.show_action_menu {
        action_menu::render_action_menu(f, app);
    }
    if app.add_menu_state.is_some() {
        action_menu::render_add_menu(f, app);
    }
    if app.confirm_flush.is_some() {
        dialog::render_flush_dialog(f, app);
    }
    filter::render_filter(f, app, footer_area);
}

fn render_placeholder(f: &mut Frame, view: View, area: ratatui::layout::Rect) {
    use ratatui::widgets::{Block, Borders, Paragraph};

    let text = format!("{} — coming soon", view.label());
    let block = Block::default()
        .borders(Borders::ALL)
        .title(view.label())
        .title_style(theme::title())
        .border_style(ratatui::style::Style::default().fg(theme::BORDER))
        .style(ratatui::style::Style::default().bg(theme::BG));
    let paragraph = Paragraph::new(text).style(theme::label()).block(block);
    f.render_widget(paragraph, area);
}
