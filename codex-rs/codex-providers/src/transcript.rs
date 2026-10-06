//! Rendering the Responses-shaped transcript into a provider-neutral form.
//!
//! Every wire protocol disagrees about how a conversation is laid out, but
//! they agree on what a conversation contains. This module holds that common
//! shape so each protocol adapter only has to know its own presentation rules.

use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_tools::TOOL_SEARCH_TOOL_NAME;

use crate::tools::wire_tool_name;

/// Conversation role in a provider-neutral transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
}

/// One element of a message body.
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    /// Plain text, whether it came from the user or the model.
    Text(String),
    /// A tool invocation requested by the model.
    ToolCall {
        id: String,
        name: String,
        /// Raw JSON text, exactly as the model produced it.
        arguments: String,
    },
    /// The result of a tool invocation, to be sent back to the model.
    ToolResult {
        call_id: String,
        name: Option<String>,
        content: String,
    },
    /// What the model thought before answering, with the signature the provider
    /// issued for it.
    Thinking {
        text: String,
        /// Providers that sign their reasoning reject a replay of the block
        /// that carries no signature, so it travels with the text.
        signature: Option<String>,
    },
    /// A reasoning block the provider redacted. Nothing but its opaque payload
    /// can be replayed.
    RedactedThinking { data: String },
}

/// A single message in the conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    /// The concatenated text of this message, ignoring tool parts.
    pub(crate) fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if let Part::Text(text) = part {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    }
}

fn coalesce_where(messages: Vec<Message>, mergeable: fn(Role) -> bool) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for message in messages {
        match out.last_mut() {
            Some(last) if last.role == message.role && mergeable(message.role) => {
                last.parts.extend(message.parts);
            }
            _ => out.push(message),
        }
    }
    out
}

/// Merge adjacent messages of any role.
///
/// The Messages API requires strictly alternating roles, so every consecutive
/// run of one role has to become a single message.
pub(crate) fn coalesce(messages: Vec<Message>) -> Vec<Message> {
    coalesce_where(messages, |_| true)
}

/// Merge adjacent assistant messages, leaving every other role split.
///
/// Chat Completions needs one assistant message per turn but expects each tool
/// result to be its own `tool` role message, so user turns must stay separate.
pub(crate) fn coalesce_assistant(messages: Vec<Message>) -> Vec<Message> {
    coalesce_where(messages, |role| role == Role::Assistant)
}

/// Render the Responses-shaped transcript into provider-neutral messages.
///
/// Items that only exist in the Responses protocol are dropped: compaction
/// markers, hosted tool calls such as web search and image generation, and
/// freeform tool calls all have no representation in the other wire protocols.
/// Reasoning survives as [`Part::Thinking`] because a provider that signs its
/// reasoning needs the block back, verbatim, when the turn is replayed.
pub fn normalize(items: &[ResponseItem]) -> Vec<Message> {
    let mut messages: Vec<Message> = Vec::with_capacity(items.len());
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let role = match role.as_str() {
                    "system" => Role::System,
                    "developer" => Role::Developer,
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                let parts: Vec<Part> = content.iter().filter_map(text_part).collect();
                if !parts.is_empty() {
                    messages.push(Message::new(role, parts));
                }
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => messages.push(Message::new(
                Role::Assistant,
                vec![Part::ToolCall {
                    id: call_id.clone(),
                    // The call went out under its flat name, and the provider
                    // answers to that same name when it replays the call.
                    name: wire_tool_name(namespace.as_deref(), name),
                    arguments: arguments.clone(),
                }],
            )),
            ResponseItem::FunctionCallOutput {
                call_id,
                name,
                namespace,
                output,
                ..
            } => {
                // Outputs are either a plain string or structured content
                // items. Only the text form survives into the other protocols.
                if let Some(content) = output.text_content() {
                    messages.push(Message::new(
                        Role::User,
                        vec![Part::ToolResult {
                            call_id: call_id.clone().unwrap_or_default(),
                            name: name
                                .as_deref()
                                .map(|name| wire_tool_name(namespace.as_deref(), name)),
                            content: content.to_string(),
                        }],
                    ));
                }
            }
            // Tool discovery has no item type here, so the schemas it found are
            // replayed as the text of the result the model asked for.
            ResponseItem::ToolSearchOutput { call_id, tools, .. } => {
                let content = serde_json::to_string(&tools).unwrap_or_else(|error| {
                    format!("failed to render {TOOL_SEARCH_TOOL_NAME} output: {error}")
                });
                messages.push(Message::new(
                    Role::User,
                    vec![Part::ToolResult {
                        call_id: call_id.clone().unwrap_or_default(),
                        name: Some(TOOL_SEARCH_TOOL_NAME.to_string()),
                        content,
                    }],
                ));
            }
            ResponseItem::Reasoning {
                summary,
                content,
                encrypted_content,
                ..
            } => {
                let text = reasoning_text(content.as_deref(), summary);
                if text.is_empty() {
                    // A redacted block carries no text, only the payload the
                    // provider wants back.
                    if let Some(data) = encrypted_content.clone() {
                        messages.push(Message::new(
                            Role::Assistant,
                            vec![Part::RedactedThinking { data }],
                        ));
                    }
                } else {
                    messages.push(Message::new(
                        Role::Assistant,
                        vec![Part::Thinking {
                            text,
                            signature: encrypted_content.clone(),
                        }],
                    ));
                }
            }
            _ => {}
        }
    }
    messages
}

/// The text of a reasoning item, preferring its raw content over a summary.
fn reasoning_text(
    content: Option<&[ReasoningItemContent]>,
    summary: &[ReasoningItemReasoningSummary],
) -> String {
    let joined = |parts: Vec<&str>| parts.join("\n");
    let text = content
        .map(|items| {
            joined(
                items
                    .iter()
                    .map(|item| match item {
                        ReasoningItemContent::ReasoningText { text }
                        | ReasoningItemContent::Text { text } => text.as_str(),
                    })
                    .collect(),
            )
        })
        .unwrap_or_default();
    if !text.is_empty() {
        return text;
    }
    joined(
        summary
            .iter()
            .map(|item| match item {
                ReasoningItemReasoningSummary::SummaryText { text } => text.as_str(),
            })
            .collect(),
    )
}

fn text_part(item: &ContentItem) -> Option<Part> {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            Some(Part::Text(text.clone()))
        }
        // Images and audio are Responses-shaped inputs. The other protocols
        // encode them differently and that encoding is not built yet.
        ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn function_call(name: &str, arguments: &str, call_id: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: name.to_string(),
            namespace: None,
            arguments: arguments.to_string(),
            encrypted_function_args: None,
            call_id: call_id.to_string(),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn roles_and_text_survive_normalization() {
        let items = [message("system", "be brief"), message("user", "hello")];
        let messages = normalize(&items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].text(), "be brief");
        assert_eq!(messages[1].role, Role::User);
    }

    #[test]
    fn a_namespaced_call_is_flattened_the_way_it_was_sent() {
        let items = [ResponseItem::FunctionCall {
            id: None,
            name: "search".to_string(),
            namespace: Some("mcp__sample__".to_string()),
            arguments: "{}".to_string(),
            encrypted_function_args: None,
            call_id: "call-1".to_string(),
            internal_chat_message_metadata_passthrough: None,
        }];
        assert_eq!(
            normalize(&items),
            vec![Message::new(
                Role::Assistant,
                vec![Part::ToolCall {
                    id: "call-1".to_string(),
                    name: "mcp__sample__search".to_string(),
                    arguments: "{}".to_string(),
                }]
            )]
        );
    }

    #[test]
    fn discovered_tool_schemas_come_back_as_a_tool_result() {
        let items = [ResponseItem::ToolSearchOutput {
            id: None,
            call_id: Some("call-search".to_string()),
            status: "completed".to_string(),
            execution: "client".to_string(),
            tools: vec![serde_json::json!({"type": "function", "name": "search"})],
            internal_chat_message_metadata_passthrough: None,
        }];

        assert_eq!(
            normalize(&items),
            vec![Message::new(
                Role::User,
                vec![Part::ToolResult {
                    call_id: "call-search".to_string(),
                    name: Some(TOOL_SEARCH_TOOL_NAME.to_string()),
                    content: r#"[{"type":"function","name":"search"}]"#.to_string(),
                }]
            )]
        );
    }

    #[test]
    fn reasoning_becomes_a_thinking_part_with_its_signature() {
        let items = [
            ResponseItem::Reasoning {
                id: None,
                summary: Vec::new(),
                content: Some(vec![ReasoningItemContent::ReasoningText {
                    text: "weighing it".to_string(),
                }]),
                encrypted_content: Some("sig-1".to_string()),
                internal_chat_message_metadata_passthrough: None,
            },
            message("user", "hello"),
        ];
        let messages = normalize(&items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::Assistant);
        assert_eq!(
            messages[0].parts,
            vec![Part::Thinking {
                text: "weighing it".to_string(),
                signature: Some("sig-1".to_string()),
            }]
        );
        // Reasoning is not conversation text.
        assert_eq!(messages[0].text(), "");
    }

    #[test]
    fn a_redacted_reasoning_item_keeps_only_its_payload() {
        let items = [ResponseItem::Reasoning {
            id: None,
            summary: Vec::new(),
            content: None,
            encrypted_content: Some("opaque".to_string()),
            internal_chat_message_metadata_passthrough: None,
        }];
        assert_eq!(
            normalize(&items),
            vec![Message::new(
                Role::Assistant,
                vec![Part::RedactedThinking {
                    data: "opaque".to_string()
                }]
            )]
        );
    }

    #[test]
    fn adjacent_assistant_text_and_tool_calls_coalesce() {
        let items = [
            message("assistant", "working on it"),
            function_call("shell", "{\"command\":\"ls\"}", "call-1"),
        ];
        let messages = coalesce_assistant(normalize(&items));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::Assistant);
        assert_eq!(messages[0].parts.len(), 2);
    }

    #[test]
    fn user_and_assistant_roles_do_not_coalesce() {
        let items = [message("user", "one"), message("assistant", "two")];
        assert_eq!(coalesce(normalize(&items)).len(), 2);
    }
}
