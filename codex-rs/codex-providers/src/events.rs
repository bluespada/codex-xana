//! The response items the stream decoders synthesize.
//!
//! Providers report a turn as incremental events. Codex records a turn as
//! items, so the decoders build the items here, in one place, rather than
//! repeating the shape in every protocol.

use codex_protocol::ToolName;
use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;

/// The response item a decoder has announced and not closed yet.
///
/// The agent loop attributes every delta to the item the stream announced last,
/// so a delta that belongs to another item has to close the open one before it
/// announces its own. Providers interleave thinking with text, and a gateway may
/// reorder or delay a block's stop, either of which would otherwise leave a
/// delta with no item to belong to.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) enum OpenItem {
    #[default]
    None,
    Message,
    Reasoning,
}

/// An assistant message with no text yet.
///
/// Sent when a text block starts so the caller can attach the deltas that
/// follow. The completed message replaces it once the turn ends.
pub(crate) fn assistant_message_started() -> ResponseItem {
    assistant_message(String::new(), None)
}

/// The assistant message for a finished turn.
pub(crate) fn assistant_message_finished(text: String, phase: MessagePhase) -> ResponseItem {
    assistant_message(text, Some(phase))
}

fn assistant_message(text: String, phase: Option<MessagePhase>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: if text.is_empty() {
            Vec::new()
        } else {
            vec![ContentItem::OutputText { text }]
        },
        phase,
        internal_chat_message_metadata_passthrough: None,
    }
}

/// A reasoning item with no content yet.
///
/// Sent when a reasoning block starts so the caller can attach the deltas that
/// follow. Codex records reasoning deltas against the active item, so it must
/// be announced before any of them arrive.
pub(crate) fn reasoning_started() -> ResponseItem {
    reasoning(String::new(), None)
}

/// The reasoning item for a finished block, carrying what the model thought
/// and, when the provider issued one, the signature it needs back on replay.
pub(crate) fn reasoning_finished(text: String, encrypted_content: Option<String>) -> ResponseItem {
    reasoning(text, encrypted_content)
}

fn reasoning(text: String, encrypted_content: Option<String>) -> ResponseItem {
    ResponseItem::Reasoning {
        id: None,
        // A provider that streams its thinking reports raw text and no summary,
        // so the text goes in the field the client shows and the Responses API
        // fills for its own models. Raw content is a second view of the same
        // text, and filling both would show it twice.
        summary: if text.is_empty() {
            Vec::new()
        } else {
            vec![ReasoningItemReasoningSummary::SummaryText { text }]
        },
        content: None,
        encrypted_content,
        internal_chat_message_metadata_passthrough: None,
    }
}

/// A function call announcement, sent before its arguments stream in.
pub(crate) fn function_call_started(call_id: &str, tool: &ToolName) -> ResponseItem {
    function_call(call_id, tool, String::new())
}

/// The function call for a finished turn, carrying the model's arguments.
pub(crate) fn function_call_finished(
    call_id: &str,
    tool: &ToolName,
    arguments: &str,
) -> ResponseItem {
    function_call(call_id, tool, arguments.to_string())
}

/// The call as this crate received it, resolved back to the identity the agent
/// loop exposed: a flattened name alone cannot say which namespace it came from.
fn function_call(call_id: &str, tool: &ToolName, arguments: String) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: tool.name.clone(),
        namespace: tool.namespace.clone(),
        // A tool with no arguments still needs a JSON object here; an empty
        // string would fail to parse when the call runs.
        arguments: if arguments.trim().is_empty() {
            "{}".to_string()
        } else {
            arguments
        },
        encrypted_function_args: None,
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

/// Asserts the item lifecycle the agent loop requires of a decoder: every delta
/// arrives while the item it belongs to is the one the stream last announced,
/// because that is the item the loop attributes it to. A delta with anything
/// else open panics a debug build and is dropped in a release one.
#[cfg(test)]
pub(crate) fn assert_deltas_have_an_open_item(events: &[codex_api::ResponseEvent]) {
    use codex_api::ResponseEvent;

    /// The item the loop would be holding.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Open {
        None,
        Message,
        Reasoning,
        Other,
    }

    fn kind(item: &ResponseItem) -> Open {
        match item {
            ResponseItem::Message { .. } => Open::Message,
            ResponseItem::Reasoning { .. } => Open::Reasoning,
            _ => Open::Other,
        }
    }

    let mut open = Open::None;
    for event in events {
        match event {
            ResponseEvent::OutputItemAdded(item) => open = kind(item),
            ResponseEvent::OutputItemDone(_) => open = Open::None,
            ResponseEvent::OutputTextDelta(text) => {
                assert_eq!(open, Open::Message, "text {text:?} with {open:?} open");
            }
            ResponseEvent::ReasoningSummaryDelta { delta, .. }
            | ResponseEvent::ReasoningContentDelta { delta, .. } => {
                assert_eq!(
                    open,
                    Open::Reasoning,
                    "reasoning {delta:?} with {open:?} open"
                );
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finished_reasoning_item_carries_its_text_and_signature() {
        let item = reasoning_finished("weighing it".to_string(), Some("sig-1".to_string()));
        assert!(matches!(
            item,
            ResponseItem::Reasoning {
                ref summary,
                content: None,
                ref encrypted_content,
                ..
            } if summary == &vec![ReasoningItemReasoningSummary::SummaryText {
                    text: "weighing it".to_string(),
                }]
                && encrypted_content.as_deref() == Some("sig-1")
        ));
    }

    #[test]
    fn a_started_reasoning_item_has_no_content() {
        assert!(matches!(
            reasoning_started(),
            ResponseItem::Reasoning {
                summary,
                content: None,
                encrypted_content: None,
                ..
            } if summary.is_empty()
        ));
    }

    #[test]
    fn an_empty_call_still_carries_json_arguments() {
        let item = function_call_started("call-1", &ToolName::plain("shell"));
        assert!(matches!(
            item,
            ResponseItem::FunctionCall { ref arguments, .. } if arguments == "{}"
        ));
    }

    #[test]
    fn a_namespaced_call_keeps_its_namespace() {
        let item = function_call_finished(
            "call-1",
            &ToolName::namespaced("mcp__sample__", "search"),
            r#"{"query":"rust"}"#,
        );
        assert!(matches!(
            item,
            ResponseItem::FunctionCall { ref name, ref namespace, .. }
                if name == "search" && namespace.as_deref() == Some("mcp__sample__")
        ));
    }

    #[test]
    fn a_started_message_has_no_content_until_text_arrives() {
        let item = assistant_message_started();
        assert!(matches!(
            item,
            ResponseItem::Message { ref content, .. } if content.is_empty()
        ));
    }

    #[test]
    fn a_finished_message_carries_its_text_and_phase() {
        let item = assistant_message_finished("done".to_string(), MessagePhase::FinalAnswer);
        assert!(matches!(
            item,
            ResponseItem::Message { phase: Some(MessagePhase::FinalAnswer), ref content, .. }
                if content == &vec![ContentItem::OutputText { text: "done".to_string() }]
        ));
    }
}
