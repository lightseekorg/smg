use std::collections::VecDeque;

use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

/// Unicode block characters for 8-level sparkline.
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render a sparkline from a VecDeque of f64 values.
/// Scales values to min/max of the buffer and maps to 8 Unicode block levels.
pub fn render_sparkline(f: &mut Frame, data: &VecDeque<f64>, color: Color, area: Rect) {
    if data.is_empty() || area.width == 0 {
        return;
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let width = area.width as usize;
    let start = data.len().saturating_sub(width);
    let chars: String = data
        .iter()
        .skip(start)
        .map(|&v| {
            let idx = if range <= f64::EPSILON {
                3
            } else {
                ((v - min) / range * 7.0).round() as usize
            };
            BLOCKS[idx.min(7)]
        })
        .collect();

    let line = Line::from(Span::styled(chars, Style::default().fg(color)));
    f.render_widget(Paragraph::new(line), area);
}

/// Build a gauge bar string: filled portion + empty portion.
/// Returns (filled_str, empty_str, percentage).
pub fn gauge_bar(ratio: f64, width: usize) -> (String, String, u16) {
    let ratio = ratio.clamp(0.0, 1.0);
    let pct = (ratio * 100.0).round() as u16;
    let filled = ((ratio * width as f64).round() as usize).min(width);
    let empty = width.saturating_sub(filled);
    ("▓".repeat(filled), "░".repeat(empty), pct)
}
