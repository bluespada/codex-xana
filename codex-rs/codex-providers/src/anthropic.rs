//! Request bodies and stream decoding for the Anthropic Messages API.
//!
//! The path is `/messages` relative to the provider base URL, which carries
//! the version the same way `https://api.openai.com/v1` does.
//!
//! The wire vocabulary is `claudius`'s: a request is a [`MessageCreateParams`]
//! and the stream is decoded with [`parse_message_stream_event`] into typed
//! [`MessageStreamEvent`]s. Codex owns the transport that carries them, so
//! auth, provider headers, TLS and proxy policy stay where they already were,
//! and this module only renders the protocol and maps its stream back onto
//! codex's item vocabulary.

use claudius::ContentBlock;
use claudius::ContentBlockDelta;
use claudius::ContentBlockDeltaEvent;
use claudius::ContentBlockStartEvent;
use claudius::ContentBlockStopEvent;
use claudius::MessageCreateParams;
use claudius::MessageDeltaEvent;
use claudius::MessageParam;
use claudius::MessageParamContent;
use claudius::MessageRole;
use claudius::MessageStartEvent;
use claudius::MessageStreamEvent;
use claudius::Model;
use claudius::RedactedThinkingBlock;
use claudius::SseEvent;
use claudius::StopReason;
use claudius::TextBlock;
use claudius::ThinkingBlock;
use claudius::ThinkingConfig;
use claudius::ToolResultBlock;
use claudius::ToolResultBlockContent;
use claudius::ToolUseBlock;
use claudius::parse_message_stream_event;

use codex_api::ApiError;
use codex_api::ResponseEvent;
use codex_api::SseEventDecoder;
use codex_protocol::models::MessagePhase;
use codex_protocol::protocol::TokenUsage;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use tracing::debug;

use crate::ChatRequest;
use crate::Message;
use crate::Part;
use crate::ProviderError;
use crate::Role;
use crate::events;
use crate::events::OpenItem;
use crate::tools;
use crate::tools::ToolAliases;
use crate::transcript::coalesce;
use crate::transcript::normalize;

/// The Messages API requires a token budget, so this is sent when the caller
/// does not configure one.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8192;

/// The name a frame carries when it has no `event:` field of its own.
const SSE_DEFAULT_EVENT: &str = "message";

/// Build a `/messages` request body, together with the flat tool names it
/// exposes and the tools they stand for.
pub fn build(request: &ChatRequest<'_>) -> Result<(Value, ToolAliases), ProviderError> {
    let body = request_body(request)?;
    let mut body = serde_json::to_value(&body).map_err(|source| ProviderError::Serialize {
        context: "anthropic messages",
        source,
    })?;
    let tool_aliases = if request.tools.is_empty() {
        ToolAliases::default()
    } else {
        let (tools, aliases) = tools::wire_tools(request.tools)?;
        // The API takes custom tools untagged. `claudius` models the tool list
        // as a union whose custom arm adds a `type` field on the way out, so
        // the definitions are translated back into the flat shape the provider
        // expects. That translation is what this crate is for.
        body["tools"] =
            serde_json::to_value(tools::anthropic_function_tools(&tools)).map_err(|source| {
                ProviderError::Serialize {
                    context: "anthropic tools",
                    source,
                }
            })?;
        aliases
    };
    if let Some(effort) = request.effort {
        // The effort codex resolved goes out as written, including values the
        // SDK's `Effort` enum has no variant for. Which of them a provider
        // accepts is the provider's business, and its refusal belongs on the
        // error path rather than in a value quietly rewritten here.
        body["output_config"] = json!({ "effort": effort.as_str() });
    }
    Ok((body, tool_aliases))
}

fn request_body(request: &ChatRequest<'_>) -> Result<MessageCreateParams, ProviderError> {
    let normalized = normalize(request.items);
    // Instructions are a top-level field here, so system and developer turns
    // move out of the message list before it is coalesced.
    let system = system_prompt(request.instructions, &normalized);
    let conversation: Vec<Message> = normalized
        .into_iter()
        .filter(|message| !matches!(message.role, Role::System | Role::Developer))
        .collect();
    let messages = coalesce(conversation)
        .iter()
        .map(render_message)
        .collect::<Result<Vec<MessageParam>, ProviderError>>()?;

    let mut params = MessageCreateParams::new_streaming(
        request
            .max_output_tokens
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
        messages,
        // A provider endpoint names its own models, so the identifier is
        // passed through rather than matched against the known set.
        Model::Custom(request.model.to_string()),
    );
    if !request.stream {
        params = params.with_stream(false);
    }
    if !system.is_empty() {
        params = params.with_system_string(system);
    }
    // Thinking is asked for on every turn, in the adaptive mode the API has
    // moved to: the model decides its own depth, and the deprecated `enabled`
    // mode with a token budget is not used. A provider that cannot think in
    // this mode answers with an error, which travels codex's normal error path
    // instead of being hidden behind a silent downgrade.
    Ok(params.with_thinking(ThinkingConfig::Adaptive))
}

fn system_prompt(instructions: Option<&str>, messages: &[Message]) -> String {
    let mut sections: Vec<String> = Vec::new();
    if let Some(instructions) = instructions
        && !instructions.trim().is_empty()
    {
        sections.push(instructions.to_string());
    }
    for message in messages {
        if matches!(message.role, Role::System | Role::Developer) {
            let text = message.text();
            if !text.trim().is_empty() {
                sections.push(text);
            }
        }
    }
    sections.join("\n\n")
}

fn render_message(message: &Message) -> Result<MessageParam, ProviderError> {
    let mut blocks = Vec::with_capacity(message.parts.len());
    for part in &message.parts {
        match part {
            Part::Text(text) => blocks.push(ContentBlock::Text(TextBlock::new(text.clone()))),
            Part::ToolCall {
                id,
                name,
                arguments,
            } => {
                // Tool arguments arrive as JSON text. An empty string means the
                // tool takes no arguments rather than a malformed invocation.
                let input: Value = if arguments.trim().is_empty() {
                    Value::Object(serde_json::Map::new())
                } else {
                    serde_json::from_str(arguments).map_err(|source| {
                        ProviderError::InvalidToolArguments {
                            name: name.clone(),
                            source,
                        }
                    })?
                };
                blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(
                    id.clone(),
                    name.clone(),
                    input,
                )));
            }
            // Tool results are content blocks inside a user turn; the Messages
            // API has no tool role.
            Part::ToolResult {
                call_id, content, ..
            } => {
                let mut block = ToolResultBlock::new(call_id.clone());
                block.content = Some(ToolResultBlockContent::String(content.clone()));
                blocks.push(ContentBlock::ToolResult(block));
            }
            // Reasoning is replayed as the provider wrote it. A signed block
            // that comes back without its signature is rejected, and the
            // signature travels in the same field the Responses protocol uses
            // for encrypted reasoning.
            Part::Thinking { text, signature } => {
                blocks.push(ContentBlock::Thinking(ThinkingBlock {
                    thinking: text.clone(),
                    signature: signature.clone().unwrap_or_default(),
                }));
            }
            Part::RedactedThinking { data } => {
                blocks.push(ContentBlock::RedactedThinking(RedactedThinkingBlock::new(
                    data.clone(),
                )));
            }
        }
    }

    let role = match message.role {
        Role::Assistant => MessageRole::Assistant,
        _ => MessageRole::User,
    };
    Ok(MessageParam::new(MessageParamContent::Array(blocks), role))
}

// ---------------------------------------------------------------------------
// Stream decoding
// ---------------------------------------------------------------------------

/// Decodes the Anthropic Messages streaming vocabulary.
///
/// The stream is a sequence of named events: the message opens, each content
/// block opens and streams its deltas, and the message closes with a stop
/// reason and the output token count.
#[derive(Default)]
pub struct AnthropicDecoder {
    /// The flat names the request exposed, used to name the calls that come
    /// back with a namespace missing from the wire.
    tool_aliases: Arc<ToolAliases>,
    response_id: Option<String>,
    text: String,
    tool_uses: Vec<PartialToolUse>,
    stop_reason: Option<String>,
    input_tokens: i64,
    cached_input_tokens: i64,
    cache_write_input_tokens: i64,
    output_tokens: i64,
    thinking_tokens: i64,
    /// The index of the reasoning block that is streaming, when one is.
    reasoning_index: Option<usize>,
    reasoning_text: String,
    reasoning_signature: Option<String>,
    /// The item the client has been told is open, if any.
    open_item: OpenItem,
    completed: bool,
}

#[derive(Default)]
struct PartialToolUse {
    call_id: String,
    name: String,
    /// Arguments assembled from `input_json_delta` fragments.
    arguments: String,
    /// Whole arguments, sent up front by servers that do not stream them.
    input: Option<String>,
    announced: bool,
}

impl PartialToolUse {
    fn arguments(&self) -> &str {
        if self.arguments.is_empty() {
            self.input.as_deref().unwrap_or_default()
        } else {
            &self.arguments
        }
    }
}

impl SseEventDecoder for AnthropicDecoder {
    fn decode(&mut self, event: &str, data: &str) -> Result<Vec<ResponseEvent>, ApiError> {
        // Keep-alive comments carry no payload.
        if data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let sse = SseEvent {
            event: event_name(event, data),
            data: data.to_string(),
        };
        let event = match parse_message_stream_event(&sse) {
            Ok(event) => event,
            Err(error) => {
                // A provider error is an event this vocabulary has no variant
                // for, so its message is recovered from the payload.
                if let Some(message) = provider_error_message(data) {
                    return Err(ApiError::Stream(message));
                }
                debug!(
                    error = %error,
                    payload_bytes = data.len(),
                    "ignoring Anthropic stream event"
                );
                return Ok(Vec::new());
            }
        };
        match event {
            MessageStreamEvent::MessageStart(start) => Ok(self.message_start(start)),
            MessageStreamEvent::ContentBlockStart(start) => Ok(self.content_block_start(start)),
            MessageStreamEvent::ContentBlockDelta(delta) => Ok(self.content_block_delta(delta)),
            MessageStreamEvent::ContentBlockStop(stop) => Ok(self.content_block_stop(stop)),
            MessageStreamEvent::MessageDelta(delta) => {
                self.message_delta(delta);
                Ok(Vec::new())
            }
            MessageStreamEvent::MessageStop(_) => Ok(self.complete()),
            MessageStreamEvent::Ping => Ok(Vec::new()),
        }
    }

    fn finish(&mut self) -> Result<Vec<ResponseEvent>, ApiError> {
        Ok(self.complete())
    }
}

impl AnthropicDecoder {
    /// A decoder that names the calls it decodes the way the request exposed
    /// them.
    pub(crate) fn with_tool_aliases(tool_aliases: Arc<ToolAliases>) -> Self {
        Self {
            tool_aliases,
            ..Self::default()
        }
    }

    fn message_start(&mut self, start: MessageStartEvent) -> Vec<ResponseEvent> {
        let mut events = Vec::new();
        if !start.message.id.is_empty() {
            self.response_id = Some(start.message.id.clone());
            events.push(ResponseEvent::Created {
                response_id: Some(start.message.id),
            });
        }
        let usage = start.message.usage;
        self.input_tokens = i64::from(usage.input_tokens);
        self.cached_input_tokens = i64::from(usage.cache_read_input_tokens.unwrap_or(0));
        self.cache_write_input_tokens = i64::from(usage.cache_creation_input_tokens.unwrap_or(0));
        if let Some(details) = usage.output_tokens_details {
            self.thinking_tokens = i64::from(details.thinking_tokens);
        }
        events
    }

    fn content_block_start(&mut self, start: ContentBlockStartEvent) -> Vec<ResponseEvent> {
        let block = match start.content_block {
            ContentBlock::ToolUse(block) => block,
            ContentBlock::Thinking(_) => {
                // The thinking text and its signature arrive as deltas.
                self.open_reasoning(start.index);
                return Vec::new();
            }
            ContentBlock::RedactedThinking(block) => {
                // A redacted block has no deltas, only the payload the provider
                // needs back, so it opens and closes at once.
                let mut emitted = self.close_open_item();
                emitted.push(ResponseEvent::OutputItemAdded(events::reasoning_started()));
                emitted.push(ResponseEvent::OutputItemDone(events::reasoning_finished(
                    String::new(),
                    Some(block.data),
                )));
                return emitted;
            }
            // Text starts on its first delta, and server-side tools are not
            // callable here.
            _ => return Vec::new(),
        };

        let tool_use = partial_tool_use(&mut self.tool_uses, start.index);
        if !block.id.is_empty() {
            tool_use.call_id = block.id;
        }
        if !block.name.is_empty() {
            tool_use.name = block.name;
        }
        if !block.input.is_null() {
            tool_use.input = Some(block.input.to_string());
        }
        if tool_use.announced || tool_use.call_id.is_empty() || tool_use.name.is_empty() {
            return Vec::new();
        }
        tool_use.announced = true;
        let tool = self.tool_aliases.resolve_or_plain(&tool_use.name);
        vec![ResponseEvent::OutputItemAdded(
            events::function_call_started(&tool_use.call_id, &tool),
        )]
    }

    fn content_block_delta(&mut self, event: ContentBlockDeltaEvent) -> Vec<ResponseEvent> {
        match event.delta {
            ContentBlockDelta::TextDelta(delta) => {
                let mut events = self.open_message_item();
                self.text.push_str(&delta.text);
                events.push(ResponseEvent::OutputTextDelta(delta.text));
                events
            }
            ContentBlockDelta::InputJsonDelta(delta) => {
                let tool_use = partial_tool_use(&mut self.tool_uses, event.index);
                tool_use.arguments.push_str(&delta.partial_json);
                if !tool_use.announced {
                    return Vec::new();
                }
                vec![ResponseEvent::ToolCallInputDelta {
                    item_id: tool_use.call_id.clone(),
                    call_id: Some(tool_use.call_id.clone()),
                    delta: delta.partial_json,
                }]
            }
            ContentBlockDelta::ThinkingDelta(delta) => {
                let mut emitted = self.open_reasoning_item(event.index);
                self.reasoning_text.push_str(&delta.thinking);
                emitted.push(ResponseEvent::ReasoningSummaryDelta {
                    delta: delta.thinking,
                    summary_index: 0,
                });
                emitted
            }
            // The signature closes the thinking block the provider signed; it
            // is replayed with the text rather than shown.
            ContentBlockDelta::SignatureDelta(delta) => {
                // A signature that arrives after its block closed belongs to
                // the item that already went out.
                if self.reasoning_index.is_some() {
                    self.reasoning_signature = Some(delta.signature);
                }
                Vec::new()
            }
            // Citations have no representation in the response items Codex
            // records.
            other => {
                debug!(delta = ?other, "ignoring Anthropic content delta");
                Vec::new()
            }
        }
    }

    /// Closes the reasoning block the given index closes, if it is the one that
    /// is open.
    fn content_block_stop(&mut self, stop: ContentBlockStopEvent) -> Vec<ResponseEvent> {
        if self.reasoning_index != Some(stop.index) {
            return Vec::new();
        }
        self.flush_reasoning()
    }

    /// Records that a reasoning block is open at `index`.
    fn open_reasoning(&mut self, index: usize) {
        self.reasoning_index = Some(index);
        self.reasoning_text.clear();
        self.reasoning_signature = None;
    }

    fn close_reasoning(&mut self) {
        self.reasoning_index = None;
        self.reasoning_text.clear();
        self.reasoning_signature = None;
    }

    /// Announces the assistant message item a text delta streams into, closing
    /// the item that was open before it.
    fn open_message_item(&mut self) -> Vec<ResponseEvent> {
        if self.open_item == OpenItem::Message {
            return Vec::new();
        }
        let mut events = self.close_open_item();
        self.open_item = OpenItem::Message;
        events.push(ResponseEvent::OutputItemAdded(
            events::assistant_message_started(),
        ));
        events
    }

    /// Announces the reasoning item the block at `index` streams into, closing
    /// the item that was open before it.
    fn open_reasoning_item(&mut self, index: usize) -> Vec<ResponseEvent> {
        if self.open_item == OpenItem::Reasoning && self.reasoning_index == Some(index) {
            return Vec::new();
        }
        let mut events = self.close_open_item();
        if self.reasoning_index != Some(index) {
            // A block that announced nothing but its deltas.
            self.open_reasoning(index);
        }
        self.open_item = OpenItem::Reasoning;
        events.push(ResponseEvent::OutputItemAdded(events::reasoning_started()));
        events
    }

    /// Closes the item the client has open, if any, so a delta from another
    /// item cannot arrive while it is the one that item belongs to.
    fn close_open_item(&mut self) -> Vec<ResponseEvent> {
        match self.open_item {
            OpenItem::None => Vec::new(),
            OpenItem::Reasoning => self.flush_reasoning(),
            OpenItem::Message => {
                // Text that continues after an interleaved block is a new
                // message; only the one that ends the turn is the answer.
                self.open_item = OpenItem::None;
                if self.text.is_empty() {
                    return Vec::new();
                }
                vec![ResponseEvent::OutputItemDone(
                    events::assistant_message_finished(
                        std::mem::take(&mut self.text),
                        MessagePhase::Commentary,
                    ),
                )]
            }
        }
    }

    /// Closes an open reasoning block, whether the stream said so or ended.
    fn flush_reasoning(&mut self) -> Vec<ResponseEvent> {
        if self.reasoning_index.is_none() {
            return Vec::new();
        }
        // A block that streamed no deltas announced no item, so it owes none.
        let announced = self.open_item == OpenItem::Reasoning;
        let text = std::mem::take(&mut self.reasoning_text);
        let signature = self.reasoning_signature.take();
        self.close_reasoning();
        if !announced {
            return Vec::new();
        }
        self.open_item = OpenItem::None;
        vec![ResponseEvent::OutputItemDone(events::reasoning_finished(
            text, signature,
        ))]
    }

    fn message_delta(&mut self, event: MessageDeltaEvent) {
        if let Some(reason) = event.delta.stop_reason {
            self.stop_reason = Some(stop_reason(reason).to_string());
        }
        self.output_tokens = i64::from(event.usage.output_tokens);
        if let Some(details) = event.usage.output_tokens_details {
            self.thinking_tokens = i64::from(details.thinking_tokens);
        }
    }

    /// The terminal events for this turn, produced once.
    fn complete(&mut self) -> Vec<ResponseEvent> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;

        // A stream that ends with an item open still owes it: a reasoning item
        // belongs before the answer it explains, and the message that ends the
        // turn is the answer.
        let mut events = Vec::new();
        let tool_uses: Vec<PartialToolUse> = std::mem::take(&mut self.tool_uses)
            .into_iter()
            .filter(|tool_use| !tool_use.name.is_empty())
            .collect();
        if self.open_item == OpenItem::Reasoning {
            events.extend(self.close_open_item());
        }
        if self.open_item == OpenItem::Message && !self.text.is_empty() {
            let phase = if tool_uses.is_empty() {
                MessagePhase::FinalAnswer
            } else {
                MessagePhase::Commentary
            };
            self.open_item = OpenItem::None;
            events.push(ResponseEvent::OutputItemDone(
                events::assistant_message_finished(std::mem::take(&mut self.text), phase),
            ));
        }
        for tool_use in &tool_uses {
            let tool = self.tool_aliases.resolve_or_plain(&tool_use.name);
            events.push(ResponseEvent::OutputItemDone(
                events::function_call_finished(&tool_use.call_id, &tool, tool_use.arguments()),
            ));
        }
        events.push(ResponseEvent::Completed {
            response_id: self.response_id.clone().unwrap_or_default(),
            token_usage: self.token_usage(),
            usage_metadata: None,
            end_turn: Some(tool_uses.is_empty() && self.stop_reason.as_deref() != Some("tool_use")),
        });
        events
    }

    fn token_usage(&self) -> Option<TokenUsage> {
        let billed = self.input_tokens + self.cached_input_tokens + self.output_tokens;
        (billed > 0).then(|| TokenUsage {
            input_tokens: self.input_tokens,
            cached_input_tokens: self.cached_input_tokens,
            cache_write_input_tokens: self.cache_write_input_tokens,
            output_tokens: self.output_tokens,
            // Thinking tokens are billed as output tokens and are reported
            // separately when the server breaks them out.
            reasoning_output_tokens: self.thinking_tokens,
            total_tokens: self.input_tokens + self.output_tokens,
            codex_rollout_budget_units: None,
        })
    }
}

fn partial_tool_use(tool_uses: &mut Vec<PartialToolUse>, index: usize) -> &mut PartialToolUse {
    while tool_uses.len() <= index {
        tool_uses.push(PartialToolUse::default());
    }
    &mut tool_uses[index]
}

/// The event name to decode by.
///
/// Anthropic sends both an `event:` field and the same name inside the payload
/// as `type`. The payload is the fallback because a frame relayed without its
/// `event:` field arrives under the SSE default name instead.
fn event_name(event: &str, data: &str) -> String {
    if !event.is_empty() && event != SSE_DEFAULT_EVENT {
        return event.to_string();
    }
    serde_json::from_str::<Value>(data)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default()
}

/// The wire spelling of a stop reason, which is what `end_turn` compares.
fn stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::PauseTurn => "pause_turn",
        StopReason::Refusal => "refusal",
        StopReason::ModelContextWindowExceeded => "model_context_window_exceeded",
    }
}

/// The message of a provider-sent error, when the payload carries one.
fn provider_error_message(data: &str) -> Option<String> {
    let value: Value = serde_json::from_str(data).ok()?;
    let error = value.get("error")?;
    Some(
        error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| error.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ReasoningItemContent;
    use codex_protocol::models::ReasoningItemReasoningSummary;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::openai_models::ReasoningEffort;
    use codex_tools::ToolSpec;

    fn user_message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn request<'a>(items: &'a [ResponseItem]) -> ChatRequest<'a> {
        ChatRequest {
            model: "claude-test",
            instructions: Some("be terse"),
            items,
            tools: &[],
            stream: true,
            max_output_tokens: None,
            effort: None,
        }
    }

    #[test]
    fn instructions_become_the_system_field() {
        let items = [user_message("hello")];
        let (body, _) = build(&request(&items)).expect("build body");
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
        assert_eq!(body["stream"], true);
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["type"], "text");
        assert_eq!(messages[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn consecutive_user_turns_merge_into_one_message() {
        let items = [user_message("one"), user_message("two")];
        let (body, _) = build(&request(&items)).expect("build body");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn tool_calls_become_use_blocks_and_results_become_user_blocks() {
        let items = [
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: "{\"command\":\"ls\"}".to_string(),
                encrypted_function_args: None,
                call_id: "call-1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: Some("call-1".to_string()),
                name: Some("shell".to_string()),
                namespace: None,
                output: FunctionCallOutputPayload::from_text("total 0".to_string()),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let (body, _) = build(&request(&items)).expect("build body");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "tool_use");
        assert_eq!(messages[0]["content"][0]["input"]["command"], "ls");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "call-1");
    }

    #[test]
    fn malformed_tool_arguments_are_reported() {
        let items = [ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            namespace: None,
            arguments: "{not json".to_string(),
            encrypted_function_args: None,
            call_id: "call-1".to_string(),
            internal_chat_message_metadata_passthrough: None,
        }];
        let error = build(&request(&items)).expect_err("malformed arguments must fail");
        assert!(matches!(error, ProviderError::InvalidToolArguments { .. }));
    }

    fn decoded(chunks: &[&str]) -> Vec<ResponseEvent> {
        let mut decoder = AnthropicDecoder::default();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.decode("", chunk).expect("decode chunk"));
        }
        events
    }

    #[test]
    fn text_streams_into_one_final_message() {
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_1","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":10,"cache_read_input_tokens":4,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        assert!(matches!(
            events[0],
            ResponseEvent::Created { ref response_id } if response_id.as_deref() == Some("msg_1")
        ));
        assert!(matches!(events[1], ResponseEvent::OutputItemAdded(_)));
        assert!(matches!(events[2], ResponseEvent::OutputTextDelta(ref text) if text == "Hel"));
        assert!(matches!(events[3], ResponseEvent::OutputTextDelta(ref text) if text == "lo"));
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Message { phase: Some(MessagePhase::FinalAnswer), ref content, .. })
                if content == &vec![ContentItem::OutputText { text: "Hello".to_string() }]
        ));
        assert!(matches!(
            events[5],
            ResponseEvent::Completed {
                end_turn: Some(true),
                token_usage: Some(ref usage),
                ..
            } if usage.input_tokens == 10 && usage.cached_input_tokens == 4 && usage.output_tokens == 2
        ));
    }

    #[test]
    fn streamed_tool_arguments_become_one_call() {
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_2","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":1,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"shell","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        assert!(matches!(
            events[3],
            ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { ref name, ref call_id, .. })
                if name == "shell" && call_id == "toolu_1"
        ));
        assert!(matches!(
            events[4],
            ResponseEvent::ToolCallInputDelta { ref call_id, ref delta, .. }
                if delta == "{\"command\":" && call_id.as_deref() == Some("toolu_1")
        ));
        assert!(matches!(
            events[6],
            ResponseEvent::OutputItemDone(ResponseItem::Message {
                phase: Some(MessagePhase::Commentary),
                ..
            })
        ));
        assert!(matches!(
            events[7],
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { ref arguments, .. })
                if arguments == "{\"command\":\"ls\"}"
        ));
        assert!(matches!(
            events[8],
            ResponseEvent::Completed {
                end_turn: Some(false),
                ..
            }
        ));
    }

    /// A namespaced tool crosses over under one flat name, and the `tool_use`
    /// block that comes back is named the way the plan exposed it.
    #[test]
    fn a_namespaced_call_comes_back_named() {
        let tools = [ToolSpec::Namespace(codex_tools::ResponsesApiNamespace {
            name: "mcp__sample__".to_string(),
            description: "Sample server tools".to_string(),
            tools: vec![codex_tools::ResponsesApiNamespaceTool::Function(
                codex_tools::ResponsesApiTool {
                    name: "search".to_string(),
                    description: "Searches the sample server.".to_string(),
                    strict: false,
                    defer_loading: None,
                    parameters: codex_tools::JsonSchema::default(),
                    output_schema: None,
                },
            )],
        })];
        let (_, aliases) = tools::wire_tools(&tools).expect("flatten tools");
        let mut decoder = AnthropicDecoder::with_tool_aliases(Arc::new(aliases));

        let mut events = Vec::new();
        for chunk in [
            r#"{"type":"message_start","message":{"id":"msg_9","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":1,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"mcp__sample__search","input":{"query":"rust"}}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            events.extend(decoder.decode("", chunk).expect("decode chunk"));
        }

        assert!(events.iter().any(|event| matches!(
            event,
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, namespace, .. })
                if name == "search" && namespace.as_deref() == Some("mcp__sample__")
        )));
    }

    #[test]
    fn thinking_tokens_are_broken_out_of_the_output_count() {
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_3","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30,"output_tokens_details":{"thinking_tokens":20}}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        assert!(matches!(
            events[1],
            ResponseEvent::Completed {
                token_usage: Some(ref usage),
                ..
            } if usage.output_tokens == 30
                && usage.reasoning_output_tokens == 20
                && usage.total_tokens == 40
        ));
    }

    #[test]
    fn a_provider_error_ends_the_stream() {
        let mut decoder = AnthropicDecoder::default();
        let error = decoder
            .decode(
                "error",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            )
            .expect_err("an error event must fail");
        assert_eq!(error.to_string(), "stream error: Overloaded");
    }

    #[test]
    fn unknown_and_keepalive_events_are_skipped() {
        let events = decoded(&[
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta"}}"#,
            "not json",
            r#"{"type":"message_stop"}"#,
        ]);
        assert!(matches!(
            events.as_slice(),
            [ResponseEvent::Completed {
                end_turn: Some(true),
                ..
            }]
        ));
    }

    #[test]
    fn thinking_becomes_a_reasoning_item_with_its_signature() {
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_4","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":3,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weigh"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ing options"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        // The item is announced before its deltas, which is what codex
        // requires to attribute them.
        assert!(matches!(
            events[1],
            ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. })
        ));
        assert!(matches!(
            events[2],
            ResponseEvent::ReasoningSummaryDelta { ref delta, summary_index: 0 } if delta == "weigh"
        ));
        assert!(matches!(
            events[3],
            ResponseEvent::ReasoningSummaryDelta { ref delta, .. } if delta == "ing options"
        ));
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                ref summary,
                ref encrypted_content,
                ..
            }) if summary == &vec![ReasoningItemReasoningSummary::SummaryText {
                    text: "weighing options".to_string(),
                }]
                && encrypted_content.as_deref() == Some("sig-1")
        ));
        assert!(matches!(
            events[5],
            ResponseEvent::Completed {
                end_turn: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn text_after_a_late_thinking_block_keeps_an_item_open() {
        // A gateway can send a thinking block after the text it follows, whose
        // stop closes the reasoning item while the answer is still streaming.
        // The text that continues owes the loop an item of its own.
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_6","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":3,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"why"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        crate::events::assert_deltas_have_an_open_item(&events);
        assert!(events.iter().any(|event| matches!(
            event,
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning { summary, .. })
                if summary == &vec![ReasoningItemReasoningSummary::SummaryText { text: "why".to_string() }]
        )));
        let messages: Vec<&ResponseEvent> = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    ResponseEvent::OutputItemDone(ResponseItem::Message { .. })
                )
            })
            .collect();
        assert!(matches!(
            messages.as_slice(),
            [
                ResponseEvent::OutputItemDone(ResponseItem::Message {
                    phase: Some(MessagePhase::Commentary),
                    content,
                    ..
                }),
                ResponseEvent::OutputItemDone(ResponseItem::Message {
                    phase: Some(MessagePhase::FinalAnswer),
                    content: answer,
                    ..
                }),
            ] if content == &vec![ContentItem::OutputText { text: "Hel".to_string() }]
                && answer == &vec![ContentItem::OutputText { text: "lo".to_string() }]
        ));
    }

    #[test]
    fn redacted_thinking_keeps_only_its_payload() {
        let events = decoded(&[
            r#"{"type":"message_start","message":{"id":"msg_5","type":"message","content":[],"model":"claude-test","role":"assistant","usage":{"input_tokens":3,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}"#,
            r#"{"type":"message_stop"}"#,
        ]);

        assert!(matches!(
            events[2],
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                content: None,
                ref encrypted_content,
                ..
            }) if encrypted_content.as_deref() == Some("opaque")
        ));
    }

    #[test]
    fn effort_goes_out_as_codex_wrote_it() {
        let items = [user_message("hello")];
        for effort in [
            ReasoningEffort::Low,
            ReasoningEffort::XHigh,
            // A value this protocol's own enum does not model. Codex still
            // asked for it, so it is still what goes out.
            ReasoningEffort::Adaptive,
            ReasoningEffort::Custom("auto".to_string()),
        ] {
            let mut request = request(&items);
            request.effort = Some(&effort);
            let (body, _) = build(&request).expect("build body");
            assert_eq!(body["output_config"]["effort"], effort.as_str());
            // Thinking stays adaptive whatever the effort says.
            assert_eq!(body["thinking"]["type"], "adaptive");
        }
    }

    #[test]
    fn reasoning_replays_as_a_signed_thinking_block() {
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
            user_message("hello"),
        ];
        let (body, _) = build(&request(&items)).expect("build body");
        assert_eq!(body["thinking"]["type"], "adaptive");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "thinking");
        assert_eq!(messages[0]["content"][0]["thinking"], "weighing it");
        assert_eq!(messages[0]["content"][0]["signature"], "sig-1");
    }
}
