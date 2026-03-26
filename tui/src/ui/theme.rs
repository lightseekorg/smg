use ratatui::style::{Color, Modifier, Style};

// Terminal-native ANSI 16 colors — adapts to user's terminal theme.
pub const BG: Color = Color::Reset; // terminal default background
pub const PANEL_BG: Color = Color::Reset; // same as terminal
pub const STATS_BG: Color = Color::Reset; // same as terminal
pub const BORDER: Color = Color::DarkGray;
pub const TEXT: Color = Color::Reset; // terminal default foreground
pub const TEXT_MUTED: Color = Color::DarkGray;
pub const ACCENT: Color = Color::Blue;
pub const GREEN: Color = Color::Green;
pub const YELLOW: Color = Color::Yellow;
pub const RED: Color = Color::Red;
pub const PURPLE: Color = Color::Magenta;

/// Style for panel titles (accent + bold).
pub fn title() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Style for section labels (muted).
pub fn label() -> Style {
    Style::default().fg(TEXT_MUTED)
}

/// Style for primary text.
pub fn text() -> Style {
    Style::default().fg(TEXT)
}

/// Severity color for a 0.0–1.0 ratio.
pub fn severity(ratio: f64) -> Color {
    if ratio < 0.5 {
        GREEN
    } else if ratio < 0.8 {
        YELLOW
    } else {
        RED
    }
}

/// Standard panel block with border.
pub fn panel(title: &str) -> ratatui::widgets::Block<'_> {
    ratatui::widgets::Block::default()
        .title(title)
        .title_style(self::title())
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(BG))
}
