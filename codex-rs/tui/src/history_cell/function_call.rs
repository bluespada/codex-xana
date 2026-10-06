//! Transcript row for a core function tool call that owns no richer cell.

use super::PrefixedWrappedHistoryCell;
use crate::style::accent_color;
use codex_protocol::DEFAULT_FUNCTION_NAMESPACE;
use ratatui::style::Stylize;
use ratatui::text::Line;

/// Output lines a collapsed row shows before the remainder is summarized.
const PREVIEW_LINES: usize = 4;

/// Render a completed function tool call as `Called <name>` plus a bounded output preview.
///
/// Tools that own a typed item (commands, patches, MCP calls, images, web search) render
/// themselves; this covers plain function tools such as `web_fetch`, whose only record is
/// the call and its output. Styling follows the dynamic and MCP call rows so every tool
/// call reads the same way in the transcript.
pub(crate) fn new_function_call_output(
    name: &str,
    namespace: Option<&str>,
    output: &str,
) -> PrefixedWrappedHistoryCell {
    let mut lines = vec![Line::from(vec![
        "•".green(),
        " ".into(),
        "Called ".bold(),
        display_tool_name(name, namespace).fg(accent_color()),
    ])];
    let preview = preview_lines(output);
    for (index, text) in preview.iter().enumerate() {
        lines.push(output_line(index, text.to_string()));
    }
    let hidden = hidden_line_count(output, preview.len());
    if hidden > 0 {
        lines.push(Line::from(vec![
            "    ".dim(),
            format!("… {hidden} more lines hidden").dim(),
        ]));
    }
    PrefixedWrappedHistoryCell::new(lines, Line::from(""), Line::from("  "))
}

/// The default namespace is not part of how the tool is named anywhere else.
fn display_tool_name(name: &str, namespace: Option<&str>) -> String {
    match namespace {
        Some(namespace) if !namespace.is_empty() && namespace != DEFAULT_FUNCTION_NAMESPACE => {
            format!("{namespace}.{name}")
        }
        _ => name.to_string(),
    }
}

/// Leading non-empty output lines, so a fetch shows its URL and status before page text.
fn preview_lines(output: &str) -> Vec<&str> {
    output
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .take(PREVIEW_LINES)
        .collect()
}

fn hidden_line_count(output: &str, shown: usize) -> usize {
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
        .saturating_sub(shown)
}

fn output_line(index: usize, text: String) -> Line<'static> {
    Line::from(vec![
        if index == 0 {
            "  └ ".dim()
        } else {
            "    ".dim()
        },
        text.dim(),
    ])
}
