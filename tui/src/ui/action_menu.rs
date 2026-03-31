use ratatui::{
    layout::{Constraint, Flex, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use super::theme;
use crate::{app::App, types::ActionMenuItem};

/// Render the worker action menu overlay.
pub fn render_action_menu(f: &mut Frame, app: &App) {
    if !app.show_action_menu {
        return;
    }

    let items = ActionMenuItem::all();
    let height = items.len() as u16 + 4; // borders + title + hint
    let width = 36u16;

    let area = f.area();
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(height),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Worker Actions ")
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if i == app.action_menu_index {
                Style::default()
                    .fg(theme::BG)
                    .bg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT)
            };
            let label = format!(" {} ", item.label());
            ListItem::new(Line::from(Span::styled(label, style)))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(app.action_menu_index));

    // We need a hint line at the bottom; render the block first, then list inside
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    // Split inner: list + hint
    let [list_area, hint_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

    f.render_stateful_widget(List::new(list_items), list_area, &mut list_state);

    let hint = Paragraph::new(Line::from(vec![
        Span::styled("↑↓/jk", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(" navigate  ", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled("Enter", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(" select  ", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled("Esc", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(" close", Style::default().fg(theme::TEXT_MUTED)),
    ]));
    f.render_widget(hint, hint_area);
}

/// Render the add worker menu overlay.
pub fn render_add_menu(f: &mut Frame, app: &App) {
    use crate::types::{AddMenuState, LocalModelPreset};

    let Some(ref state) = app.add_menu_state else {
        return;
    };

    match state {
        AddMenuState::SelectCategory => {
            render_menu(
                f,
                " Add Worker ",
                &[
                    ("1", "External", "openai, anthropic, etc."),
                    ("2", "Local", "sglang, vllm"),
                    ("3", "Custom URL", "manual"),
                ],
            );
        }
        AddMenuState::SelectProvider => {
            render_menu(
                f,
                " External Provider ",
                &[
                    ("1", "OpenAI", "https://api.openai.com"),
                    ("2", "Anthropic", "https://api.anthropic.com"),
                    ("3", "xAI (Grok)", "https://api.x.ai"),
                    ("4", "Gemini", "https://generativelanguage.googleapis.com"),
                ],
            );
        }
        AddMenuState::EnterApiKey { provider, input } => {
            render_text_input(
                f,
                &format!(" Add {} ", provider.label()),
                "API Key:",
                input,
                true,
            );
        }
        AddMenuState::SelectRuntime => {
            render_menu(
                f,
                " Local Backend ",
                &[
                    ("1", "SGLang", "high-performance serving"),
                    ("2", "vLLM", "versatile serving"),
                ],
            );
        }
        AddMenuState::SelectConnection { runtime } => {
            render_menu(
                f,
                &format!(" {} — Connection ", runtime.label()),
                &[
                    ("1", "HTTP", "OpenAI-compatible REST"),
                    ("2", "gRPC", "high-performance binary"),
                ],
            );
        }
        AddMenuState::SelectModel { runtime, .. } => {
            let presets = LocalModelPreset::all();
            let mut items: Vec<(String, String, String)> = presets
                .iter()
                .enumerate()
                .map(|(i, p)| (format!("{}", i + 1), p.label(), format!("TP={}", p.tp())))
                .collect();
            items.push((
                format!("{}", presets.len() + 1),
                "Custom model...".to_string(),
                "enter model ID + TP".to_string(),
            ));

            let title = format!(" {} — Model ", runtime.label());
            let refs: Vec<(&str, &str, &str)> = items
                .iter()
                .map(|(n, l, d)| (n.as_str(), l.as_str(), d.as_str()))
                .collect();
            render_menu(f, &title, &refs);
        }
        AddMenuState::EnterCustomModel {
            runtime,
            field,
            model_id,
            tp,
            extra_args,
            ..
        } => {
            render_custom_model_form(
                f,
                &format!(" {} — Custom Model ", runtime.label()),
                *field,
                model_id,
                tp,
                extra_args,
            );
        }
        AddMenuState::EnterCustomUrl { input } => {
            render_text_input(f, " Custom Worker ", "URL:", input, false);
        }
    }
}

fn render_menu(f: &mut Frame, title: &str, items: &[(&str, &str, &str)]) {
    let height = items.len() as u16 * 2 + 5;
    let width = 50u16;

    let area = f.area();
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(height),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = Vec::new();
    for (num, label, desc) in items {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" [{num}] "),
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(*label, Style::default().fg(theme::TEXT)),
            Span::styled(format!("  {desc}"), Style::default().fg(theme::TEXT_MUTED)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " Esc to cancel",
        Style::default().fg(theme::TEXT_MUTED),
    )));

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn render_text_input(f: &mut Frame, title: &str, label: &str, input: &str, masked: bool) {
    let width = 55u16;
    let height = 8u16;

    let area = f.area();
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(height),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let display = if input.is_empty() {
        Span::styled("Type here...", Style::default().fg(theme::TEXT_MUTED))
    } else if masked {
        Span::styled("*".repeat(input.len()), Style::default().fg(theme::TEXT))
    } else {
        Span::styled(input, Style::default().fg(theme::TEXT))
    };

    let lines = vec![
        Line::from(Span::styled(
            label,
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            display,
            Span::styled("▊", Style::default().fg(theme::ACCENT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Enter to confirm  Esc to cancel",
            Style::default().fg(theme::TEXT_MUTED),
        )),
    ];

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn render_custom_model_form(
    f: &mut Frame,
    title: &str,
    active_field: u8,
    model_id: &str,
    tp: &str,
    extra_args: &str,
) {
    let width = 60u16;
    let height = 14u16;

    let area = f.area();
    let [_, vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(height),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [popup] = Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .areas(vert);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(theme::title())
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::PANEL_BG));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let field_style = |idx: u8| {
        if idx == active_field {
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT_MUTED)
        }
    };
    let cursor = |idx: u8| {
        if idx == active_field {
            Span::styled("▊", Style::default().fg(theme::ACCENT))
        } else {
            Span::raw("")
        }
    };

    let lines = vec![
        Line::from(Span::styled("Model ID:", field_style(0))),
        Line::from(vec![
            Span::styled(
                if model_id.is_empty() && active_field == 0 {
                    " e.g. meta-llama/Llama-3.1-8B-Instruct".to_string()
                } else {
                    format!(" {model_id}")
                },
                if model_id.is_empty() {
                    Style::default().fg(theme::TEXT_MUTED)
                } else {
                    Style::default().fg(theme::TEXT)
                },
            ),
            cursor(0),
        ]),
        Line::from(""),
        Line::from(Span::styled("Tensor Parallel (TP):", field_style(1))),
        Line::from(vec![
            Span::styled(format!(" {tp}"), Style::default().fg(theme::TEXT)),
            cursor(1),
        ]),
        Line::from(""),
        Line::from(Span::styled("Extra Args:", field_style(2))),
        Line::from(vec![
            Span::styled(
                if extra_args.is_empty() {
                    " e.g. --max-model-len 16384".to_string()
                } else {
                    format!(" {extra_args}")
                },
                if extra_args.is_empty() {
                    Style::default().fg(theme::TEXT_MUTED)
                } else {
                    Style::default().fg(theme::TEXT)
                },
            ),
            cursor(2),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Tab next  Shift+Tab prev  Enter confirm  Esc cancel",
            Style::default().fg(theme::TEXT_MUTED),
        )),
    ];

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}
