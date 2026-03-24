use ratatui::{
    layout::{Constraint, Flex, Layout, Rect},
    style::Style,
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use super::theme;
use crate::types::View;

fn help_text(view: View) -> String {
    let mut text = String::from(
        "\
Navigation
  1-7          Switch view (Pulse/Workers/Chat/Logs/Bench/Traffic/Mesh)
  q            Quit TUI (services keep running)
  Ctrl+C ×2    Full shutdown (stop all services)
  ?            Toggle this help
  /            Filter
  :            Command mode
  Esc          Close overlay / clear filter

",
    );

    match view {
        View::Workers => {
            text.push_str(
                "\
Workers View
  j / Down     Move selection down
  k / Up       Move selection up
  Enter        Toggle worker detail panel
  e            Action menu (priority, cost, flush cache, ...)
  a            Add worker (enters command mode)
  d            Delete selected worker

",
            );
        }
        _ => {
            text.push_str(
                "\
General
  j / Down     Move selection down
  k / Up       Move selection up

",
            );
        }
    }

    text.push_str(
        "\
Commands
  :add <url> [--provider <p>] [--runtime <r>]
                   Add a worker
  :delete <id>     Delete a worker by ID
  :priority <id> <n>   Set worker priority
  :cost <id> <n>       Set worker cost
  :flush-cache <id>    Flush worker cache
  :toggle-health <id>  Toggle health check
  :api-key <key>       Update API key
  :add-openai          Quick-add OpenAI worker
  :quit                Quit the TUI

Providers: openai, anthropic, gemini, xai
Runtimes:  sglang (default), vllm, trtllm, external
",
    );

    text
}

pub fn render_help(f: &mut Frame, view: View) {
    let popup = centered_rect(60, 70, f.area());
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Help ")
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let paragraph = Paragraph::new(help_text(view))
        .style(Style::default().fg(theme::TEXT))
        .block(block)
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, popup);
}

/// Return a centered `Rect` occupying `percent_x`% width and `percent_y`% height.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [_, vert, _] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);

    let [horiz] = Layout::horizontal([Constraint::Percentage(percent_x)])
        .flex(Flex::Center)
        .areas(vert);

    horiz
}
