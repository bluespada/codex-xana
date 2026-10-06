//! Fetches a URL over HTTP and returns its readable text.
//!
//! The tool exists so a turn can read a page without shelling out to a network
//! client, and it stays available when web search is disabled or when the
//! provider has no hosted search tool. Extraction is deliberately
//! dependency-free: prose-less elements are dropped, block boundaries become
//! newlines, tags and entities are stripped, and runs of spaces collapse.
//! Preformatted blocks lose their indentation, and non-text responses are
//! refused rather than guessed at.

use std::time::Duration;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
use codex_http_client::ClientRouteClass;
use codex_http_client::RouteAwareClientPool;
use codex_protocol::DEFAULT_FUNCTION_NAMESPACE;
use codex_protocol::items::FunctionCallOutputItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseInputItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_json::json;
use std::collections::BTreeMap;

pub(crate) const WEB_FETCH_TOOL_NAME: &str = "web_fetch";

/// Response bytes read before extraction stops.
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
/// Characters returned to the model unless the caller asks for another amount.
const DEFAULT_MAX_CHARS: usize = 50_000;
/// Upper bound on the caller's `max_chars`.
const MAX_MAX_CHARS: usize = 500_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const USER_AGENT: &str = concat!("codex/", env!("CARGO_PKG_VERSION"));

/// Elements whose contents never reach the reader as prose.
const SKIPPED_ELEMENTS: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
];
/// Elements that end a line of prose, so their boundaries become newlines.
const BLOCK_ELEMENTS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
];
/// Named entities the extractor understands. Numeric references are decoded
/// separately, so this table only needs the common named ones.
const NAMED_ENTITIES: &[(&str, &str)] = &[
    ("amp", "&"),
    ("apos", "'"),
    ("gt", ">"),
    ("lt", "<"),
    ("nbsp", " "),
    ("quot", "\""),
];

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default)]
    max_chars: Option<usize>,
}

struct WebFetchOutput {
    text: String,
}

impl ToolOutput for WebFetchOutput {
    fn log_output(&self) -> String {
        self.text.clone()
    }

    fn success_for_logging(&self) -> bool {
        true
    }

    fn contains_external_context(&self) -> bool {
        true
    }

    fn to_response_item(&self, call_id: &str, payload: &ToolPayload) -> ResponseInputItem {
        FunctionToolOutput::from_text(self.text.clone(), Some(true))
            .to_response_item(call_id, payload)
    }

    fn code_mode_result(&self, _payload: &ToolPayload) -> JsonValue {
        json!({ "content": self.text })
    }
}

pub struct WebFetchHandler;

impl ToolExecutor<ToolInvocation> for WebFetchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_FETCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        let properties = BTreeMap::from([
            (
                "url".to_string(),
                JsonSchema::string(Some("Absolute http or https URL to fetch.".to_string())),
            ),
            (
                "max_chars".to_string(),
                JsonSchema::number(Some(format!(
                    "Maximum characters of text to return. Defaults to {DEFAULT_MAX_CHARS}, capped at {MAX_MAX_CHARS}."
                ))),
            ),
        ]);
        ToolSpec::Function(ResponsesApiTool {
            name: WEB_FETCH_TOOL_NAME.to_string(),
            description: "Fetch a URL and return its readable text.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                properties,
                Some(vec!["url".to_string()]),
                Some(false.into()),
            ),
            output_schema: None,
        })
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(async move {
            let arguments = match &invocation.payload {
                ToolPayload::Function { arguments } => arguments,
                _ => {
                    return Err(FunctionCallError::RespondToModel(format!(
                        "{WEB_FETCH_TOOL_NAME} handler received unsupported payload"
                    )));
                }
            };
            let args: WebFetchArgs = serde_json::from_str(arguments).map_err(|err| {
                FunctionCallError::RespondToModel(format!(
                    "failed to parse {WEB_FETCH_TOOL_NAME} arguments: {err}"
                ))
            })?;
            let url = args.url.trim().to_string();
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(FunctionCallError::RespondToModel(
                    "url must be an absolute http or https URL".to_string(),
                ));
            }

            let max_chars = resolve_max_chars(args.max_chars);
            let outcome = fetch_readable_text(&invocation, url.as_str(), max_chars).await;

            // Plain function tools own no typed transcript cell, so publish the call and its
            // outcome as a function call output item, which the transcript renders as one row.
            let item_text = match &outcome {
                Ok(text) => text.clone(),
                Err(err) => err.to_string(),
            };
            let item = TurnItem::FunctionCallOutput(FunctionCallOutputItem {
                id: invocation.call_id.clone(),
                name: invocation.tool_name.name.clone(),
                // The default namespace is core's normalization sink, not part of the name the
                // model used, so it is left out of the item.
                namespace: invocation
                    .tool_name
                    .namespace
                    .clone()
                    .filter(|namespace| namespace != DEFAULT_FUNCTION_NAMESPACE),
                output: FunctionCallOutputBody::Text(item_text),
            });
            invocation
                .session
                .emit_turn_item_started(invocation.turn.as_ref(), &item)
                .await;
            invocation
                .session
                .emit_turn_item_completed(invocation.turn.as_ref(), item)
                .await;

            Ok(boxed_tool_output(WebFetchOutput { text: outcome? }))
        })
    }
}

impl CoreToolRuntime for WebFetchHandler {}

/// Fetch `url` and render it as `URL`/status/content-type header lines plus readable text.
async fn fetch_readable_text(
    invocation: &ToolInvocation,
    url: &str,
    max_chars: usize,
) -> Result<String, FunctionCallError> {
    let client = RouteAwareClientPool::with_connect_timeout(
        invocation.turn.config.http_client_factory(),
        ClientRouteClass::Other,
        CONNECT_TIMEOUT,
    );
    let mut response = client
        .get(url)
        .header("user-agent", USER_AGENT)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|err| FunctionCallError::RespondToModel(format!("GET {url} failed: {err}")))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_default();
    let mut body = Vec::new();
    while body.len() < MAX_RESPONSE_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => body.extend_from_slice(&chunk),
            Ok(None) => break,
            Err(err) => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "GET {url} failed while reading the response: {err}"
                )));
            }
        }
    }

    if !(200..300).contains(&status) {
        return Err(FunctionCallError::RespondToModel(format!(
            "GET {url} returned HTTP {status}"
        )));
    }
    let content_type_label = if content_type.is_empty() {
        "unknown".to_string()
    } else {
        content_type.clone()
    };
    if !is_textual(&content_type) {
        return Err(FunctionCallError::RespondToModel(format!(
            "GET {url} returned unsupported content type `{content_type_label}`; {WEB_FETCH_TOOL_NAME} reads text only"
        )));
    }

    let body = String::from_utf8_lossy(&body);
    let text = if is_html(&content_type) {
        html_to_text(&body)
    } else {
        body.into_owned()
    };
    let (text, truncated) = truncate_chars(&text, max_chars);
    let mut rendered =
        format!("URL: {url}\nHTTP status: {status}\nContent-Type: {content_type_label}\n\n{text}");
    if truncated {
        rendered.push_str(&format!("\n\n[truncated at {max_chars} characters]"));
    }
    Ok(rendered)
}

fn resolve_max_chars(requested: Option<usize>) -> usize {
    requested
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_CHARS)
        .min(MAX_MAX_CHARS)
}

fn is_html(content_type: &str) -> bool {
    let content_type = content_type.to_ascii_lowercase();
    content_type.contains("html") || content_type.contains("xhtml")
}

/// Whether the response may be read as text. Everything else is refused so the
/// model never receives a binary body rendered as lossy UTF-8.
fn is_textual(content_type: &str) -> bool {
    let content_type = content_type.to_ascii_lowercase();
    if content_type.is_empty() {
        return true;
    }
    if content_type.starts_with("text/") {
        return true;
    }
    content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
        || content_type.contains("x-yaml")
        || content_type.contains("yaml")
}

/// Returns the text with at most `max_chars` characters, plus whether anything
/// was dropped.
fn truncate_chars(text: &str, max_chars: usize) -> (String, bool) {
    if text.chars().count() <= max_chars {
        return (text.to_string(), false);
    }
    (text.chars().take(max_chars).collect(), true)
}

/// Extracts readable text from an HTML document without a parser.
fn html_to_text(html: &str) -> String {
    let stripped = strip_tags(html);
    let decoded = decode_entities(&stripped);
    collapse_whitespace(&decoded)
}

fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        if let Some(body) = after.strip_prefix("!--") {
            match body.find("-->") {
                Some(end) => {
                    rest = &body[end + 3..];
                    continue;
                }
                // An unterminated comment swallows the remainder of the document.
                None => return out,
            }
        }
        // A `<` that does not start a tag is text, which is what a browser
        // renders for a stray angle bracket; only a tag name, a closing slash,
        // or a declaration opens one.
        if !starts_tag(after) {
            out.push('<');
            rest = after;
            continue;
        }
        let Some(close) = after.find('>') else {
            // An unterminated tag is not a tag; keep it as text.
            out.push_str(rest[open..].trim_start_matches('<'));
            return out;
        };
        let tag = &after[..close];
        rest = &after[close + 1..];

        let name = element_name(tag);
        if !tag.starts_with('/') && matches_element(name, SKIPPED_ELEMENTS) {
            rest = skip_element(rest, name);
            continue;
        }
        if matches_element(name, BLOCK_ELEMENTS) {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    out.push_str(rest);
    out
}

/// Whether the text after a `<` starts a tag.
fn starts_tag(after: &str) -> bool {
    match after.chars().next() {
        Some(character) => character.is_ascii_alphabetic() || matches!(character, '/' | '!' | '?'),
        None => false,
    }
}

/// The element name from a start, end, or self-closing tag such as `div`, `/p`,
/// or `br/`.
fn element_name(tag: &str) -> &str {
    let tag = tag.trim_start_matches('/');
    let end = tag
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(tag.len());
    &tag[..end]
}

fn matches_element(name: &str, elements: &[&str]) -> bool {
    !name.is_empty()
        && elements
            .iter()
            .any(|element| name.eq_ignore_ascii_case(element))
}

/// Returns the remainder of the document after the closing tag of `name`.
/// An element that never closes swallows the remainder, matching how a browser
/// keeps the rest of the document inside it.
fn skip_element<'a>(rest: &'a str, name: &str) -> &'a str {
    let mut search_from = 0;
    while let Some(offset) = rest[search_from..].find('<') {
        let start = search_from + offset;
        let after = &rest[start + 1..];
        if let Some(body) = after.strip_prefix('/')
            && matches_element(element_name(body), &[name])
            && let Some(close) = body.find('>')
        {
            return &body[close + 1..];
        }
        search_from = start + 1;
    }
    ""
}

fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(ampersand) = rest.find('&') {
        out.push_str(&rest[..ampersand]);
        let after = &rest[ampersand + 1..];
        let Some(semicolon) = after.find(';').filter(|end| *end <= 10) else {
            out.push('&');
            rest = after;
            continue;
        };
        let entity = &after[..semicolon];
        match decode_entity(entity) {
            Some(decoded) => {
                out.push_str(&decoded);
                rest = &after[semicolon + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<String> {
    if let Some(named) = NAMED_ENTITIES
        .iter()
        .find(|(name, _)| entity.eq_ignore_ascii_case(name))
    {
        return Some(named.1.to_string());
    }
    let numeric = entity.strip_prefix('#')?;
    let code = match numeric.strip_prefix(['x', 'X']) {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None => numeric.parse::<u32>().ok()?,
    };
    char::from_u32(code).map(String::from)
}

/// Collapses HTML whitespace: runs of spaces and tabs become one space, blank
/// lines are dropped, and leading indentation disappears. Preformatted blocks
/// therefore lose their layout.
fn collapse_whitespace(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.split('\n') {
        let collapsed = collapse_spaces(line);
        let collapsed = collapsed.trim_end();
        if collapsed.is_empty() {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            continue;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(collapsed);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines.join("\n").trim().to_string()
}

fn collapse_spaces(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_run = false;
    for character in line.chars() {
        if character == ' ' || character == '\t' || character == '\r' {
            in_run = true;
            continue;
        }
        if in_run && !out.is_empty() {
            out.push(' ');
        }
        in_run = false;
        out.push(character);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_prose_less_elements_and_keeps_prose() {
        let html = "<html><head><style>body{color:red}</style></head><body>\
            <h1>Title</h1><p>First paragraph.</p>\
            <script>console.log('nope')</script><p>Second paragraph.</p></body></html>";
        assert_eq!(
            html_to_text(html),
            "Title\nFirst paragraph.\nSecond paragraph."
        );
    }

    #[test]
    fn decodes_named_and_numeric_entities() {
        assert_eq!(
            html_to_text("<p>a &amp; b &lt;c&gt; &#39;d&#39; &#x41;</p>"),
            "a & b <c> 'd' A"
        );
    }

    #[test]
    fn collapses_indentation_and_blank_lines() {
        let html = "<div>\n   <p>   indented\n</p>\n\n\n<p>next</p>\n</div>";
        assert_eq!(html_to_text(html), "indented\nnext");
    }

    #[test]
    fn keeps_an_unterminated_tag_as_text() {
        assert_eq!(
            html_to_text("<p>broken < not a tag</p>"),
            "broken < not a tag"
        );
    }

    #[test]
    fn an_unclosed_skipped_element_swallows_the_remainder() {
        assert_eq!(
            html_to_text("<p>kept</p><script>var a = '<p>hidden"),
            "kept"
        );
    }

    #[test]
    fn clamps_the_requested_character_budget() {
        assert_eq!(resolve_max_chars(None), DEFAULT_MAX_CHARS);
        assert_eq!(resolve_max_chars(Some(0)), DEFAULT_MAX_CHARS);
        assert_eq!(resolve_max_chars(Some(10)), 10);
        assert_eq!(resolve_max_chars(Some(usize::MAX)), MAX_MAX_CHARS);
    }

    #[test]
    fn truncates_long_text_and_says_so() {
        assert_eq!(truncate_chars("abcd", 4), ("abcd".to_string(), false));
        assert_eq!(truncate_chars("abcde", 4), ("abcd".to_string(), true));
        assert_eq!(truncate_chars("héllo", 2), ("hé".to_string(), true));
    }

    #[test]
    fn reads_textual_content_types_only() {
        assert!(is_textual("text/html; charset=utf-8"));
        assert!(is_textual("application/json"));
        assert!(is_textual(""));
        assert!(!is_textual("image/png"));
        assert!(!is_textual("application/pdf"));
        assert!(is_html("text/html; charset=UTF-8"));
        assert!(!is_html("text/plain"));
    }

    #[test]
    fn the_spec_is_a_function_tool_named_web_fetch() {
        let spec = WebFetchHandler.spec();
        let ToolSpec::Function(tool) = spec else {
            panic!("web_fetch must be a plain function tool so every wire protocol can carry it");
        };
        assert_eq!(tool.name, WEB_FETCH_TOOL_NAME);
        assert_eq!(
            WebFetchHandler.tool_name(),
            ToolName::plain(WEB_FETCH_TOOL_NAME)
        );
    }
}
