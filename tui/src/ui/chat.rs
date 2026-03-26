use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::theme;
use crate::app::App;

pub fn render_chat(f: &mut Frame, app: &App, area: Rect) {
    // Layout: messages (fill) + input (3 lines)
    let [messages_area, input_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);

    render_messages(f, app, messages_area);
    render_input(f, app, input_area);
}

fn render_messages(f: &mut Frame, app: &App, area: Rect) {
    let mode_hint = match app.chat_endpoint {
        crate::chat::ChatEndpoint::Chat => "full history",
        crate::chat::ChatEndpoint::Responses => {
            if app.chat_previous_response_id.is_some() {
                "prev_response_id"
            } else {
                "first turn"
            }
        }
    };
    let title = format!(
        " Chat — {} — /v1/{} ({}) ",
        app.chat_model,
        app.chat_endpoint.label(),
        mode_hint,
    );
    let block = Block::default()
        .title(title)
        .title_style(theme::title())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::BG));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.chat_messages.is_empty() {
        let help = vec![
            Line::from(""),
            Line::from(Span::styled(
                "Type a message and press Enter to chat.",
                theme::label(),
            )),
            Line::from(Span::styled(
                "Tab: cycle models  Shift+Tab: cycle endpoint  Esc: cancel",
                theme::label(),
            )),
        ];
        f.render_widget(Paragraph::new(help), inner);
        return;
    }

    // Build all lines from messages
    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.chat_messages {
        let (prefix, prefix_style) = match msg.role.as_str() {
            "user" => (
                "You: ",
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            "assistant" => (
                "SMG: ",
                Style::default()
                    .fg(theme::GREEN)
                    .add_modifier(Modifier::BOLD),
            ),
            _ => (
                "",
                Style::default()
                    .fg(theme::YELLOW)
                    .add_modifier(Modifier::BOLD),
            ),
        };

        // First line gets the role prefix
        let is_assistant = msg.role == "assistant";
        let content_lines: Vec<&str> = msg.content.split('\n').collect();
        let mut in_code_block = false;

        for (i, content_line) in content_lines.iter().enumerate() {
            // Track fenced code blocks
            if content_line.starts_with("```") {
                in_code_block = !in_code_block;
                let mut spans = Vec::new();
                if i == 0 {
                    spans.push(Span::styled(prefix, prefix_style));
                } else {
                    spans.push(Span::raw(" ".repeat(prefix.len())));
                }
                spans.push(Span::styled(
                    *content_line,
                    Style::default().fg(theme::YELLOW),
                ));
                lines.push(Line::from(spans));
                continue;
            }

            let mut spans = Vec::new();
            if i == 0 {
                spans.push(Span::styled(prefix, prefix_style));
            } else {
                spans.push(Span::raw(" ".repeat(prefix.len())));
            }

            if in_code_block {
                // Inside code block — render as-is with code style
                spans.push(Span::styled(
                    *content_line,
                    Style::default().fg(theme::YELLOW),
                ));
            } else if is_assistant {
                spans.extend(parse_markdown_spans(content_line));
            } else {
                spans.push(Span::styled(*content_line, theme::text()));
            }

            lines.push(Line::from(spans));
        }

        // Show streaming cursor
        if msg.role == "assistant"
            && app.chat_streaming
            && app
                .chat_messages
                .last()
                .is_some_and(|last| std::ptr::eq(msg, last))
        {
            if let Some(last_line) = lines.last_mut() {
                last_line
                    .spans
                    .push(Span::styled("▊", Style::default().fg(theme::ACCENT)));
            }
        }

        lines.push(Line::from("")); // blank line between messages
    }

    // Calculate wrapped line count for proper scrolling
    let width = inner.width as usize;
    let total_lines: u16 = if width > 0 {
        lines
            .iter()
            .map(|line| {
                let line_width: usize = line.spans.iter().map(|s| s.content.len()).sum();
                line_width.max(1).div_ceil(width) as u16 // ceil division
            })
            .sum()
    } else {
        lines.len() as u16
    };
    let visible = inner.height;
    let max_scroll = total_lines.saturating_sub(visible);
    let scroll = if app.chat_scroll >= max_scroll {
        max_scroll
    } else {
        app.chat_scroll
    };

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        inner,
    );
}

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.chat_streaming {
        " Streaming... (Esc to stop) "
    } else {
        " Message (Enter to send, Tab: model, Shift+Tab: endpoint) "
    };

    let block = Block::default()
        .title(title)
        .title_style(if app.chat_streaming {
            Style::default().fg(theme::YELLOW)
        } else {
            theme::title()
        })
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if app.chat_streaming {
            theme::YELLOW
        } else {
            theme::BORDER
        }))
        .style(Style::default().bg(theme::BG));

    let input_text = if app.chat_streaming {
        String::new()
    } else {
        format!("{}▊", app.chat_input)
    };

    let paragraph = Paragraph::new(input_text).style(theme::text()).block(block);
    f.render_widget(paragraph, area);
}

/// Parse inline markdown into styled spans.
/// Supports: **bold**, *italic*, `code`, ### headings
fn parse_markdown_spans(line: &str) -> Vec<Span<'_>> {
    // Handle heading lines
    if let Some(rest) = line.strip_prefix("### ") {
        return vec![Span::styled(
            rest,
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];
    }
    if let Some(rest) = line.strip_prefix("## ") {
        return vec![Span::styled(
            rest,
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];
    }
    if let Some(rest) = line.strip_prefix("# ") {
        return vec![Span::styled(
            rest,
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];
    }
    // Handle bullet points
    let (bullet_prefix, rest) = if let Some(rest) = line.strip_prefix("- ") {
        ("• ", rest)
    } else if let Some(rest) = line.strip_prefix("* ") {
        ("• ", rest)
    } else {
        ("", line)
    };

    let mut spans = Vec::new();
    if !bullet_prefix.is_empty() {
        spans.push(Span::styled(bullet_prefix, theme::text()));
    }

    let chars: Vec<char> = rest.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut buf = String::new();

    while i < len {
        // **bold**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), theme::text()));
                buf.clear();
            }
            i += 2;
            let start = i;
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '*') {
                i += 1;
            }
            let bold_text: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                bold_text,
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ));
            if i + 1 < len {
                i += 2; // skip closing **
            }
            continue;
        }

        // `code`
        if chars[i] == '`' {
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), theme::text()));
                buf.clear();
            }
            i += 1;
            let start = i;
            while i < len && chars[i] != '`' {
                i += 1;
            }
            let code_text: String = chars[start..i].iter().collect();
            spans.push(Span::styled(code_text, Style::default().fg(theme::YELLOW)));
            if i < len {
                i += 1; // skip closing `
            }
            continue;
        }

        // *italic* (single asterisk, not double)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), theme::text()));
                buf.clear();
            }
            i += 1;
            let start = i;
            while i < len && chars[i] != '*' {
                i += 1;
            }
            let italic_text: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                italic_text,
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::ITALIC),
            ));
            if i < len {
                i += 1; // skip closing *
            }
            continue;
        }

        buf.push(chars[i]);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(buf, theme::text()));
    }

    if spans.is_empty() {
        spans.push(Span::styled("", theme::text()));
    }

    spans
}
