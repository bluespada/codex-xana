//! Request bodies and stream decoding for the OpenAI Chat Completions API.
//!
//! The wire vocabulary is `async-openai`'s: a request is a
//! [`CreateChatCompletionRequest`] and a chunk is a
//! [`CreateChatCompletionStreamResponse`]. Codex owns the transport that
//! carries them, so auth, provider headers, TLS and proxy policy stay where
//! they already were, and this module only renders the protocol and maps its
//! stream back onto codex's item vocabulary.

use async_openai::types::chat::ChatCompletionMessageToolCall;
use async_openai::types::chat::ChatCompletionMessageToolCalls;
use async_openai::types::chat::ChatCompletionRequestAssistantMessageArgs;
use async_openai::types::chat::ChatCompletionRequestAssistantMessageContent;
use async_openai::types::chat::ChatCompletionRequestMessage;
use async_openai::types::chat::ChatCompletionRequestSystemMessageArgs;
use async_openai::types::chat::ChatCompletionRequestSystemMessageContent;
use async_openai::types::chat::ChatCompletionRequestToolMessageArgs;
use async_openai::types::chat::ChatCompletionRequestToolMessageContent;
use async_openai::types::chat::ChatCompletionRequestUserMessageArgs;
use async_openai::types::chat::ChatCompletionRequestUserMessageContent;
use async_openai::types::chat::CompletionUsage;
use async_openai::types::chat::CreateChatCompletionRequest;
use async_openai::types::chat::CreateChatCompletionRequestArgs;
use async_openai::types::chat::CreateChatCompletionStreamResponse;
use async_openai::types::chat::FunctionCall;
use async_openai::types::chat::Role as OpenAiRole;

use codex_api::ApiError;
use codex_api::ResponseEvent;
use codex_api::SseEventDecoder;
use codex_protocol::models::MessagePhase;
use codex_protocol::protocol::TokenUsage;
use serde_json::Map;
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
use crate::transcript::coalesce_assistant;
use crate::transcript::normalize;

/// Build a `/chat/completions` request body, together with the flat tool names
/// it exposes and the tools they stand for.
pub fn build(request: &ChatRequest<'_>) -> Result<(Value, ToolAliases), ProviderError> {
    let (body, tool_aliases) = request_body(request)?;
    let mut body = serde_json::to_value(&body).map_err(|source| ProviderError::Serialize {
        context: "chat completions",
        source,
    })?;
    if let Some(effort) = request.effort {
        // The SDK models the effort levels it knows and would refuse to
        // serialize any other, while codex can resolve a value outside that
        // set. The resolved effort is therefore written where the provider
        // reads it, as it stands, and a provider that rejects it says so.
        body["reasoning_effort"] = Value::String(effort.as_str().to_string());
    }
    Ok((body, tool_aliases))
}

fn request_body(
    request: &ChatRequest<'_>,
) -> Result<(CreateChatCompletionRequest, ToolAliases), ProviderError> {
    const CONTEXT: &str = "chat completions";

    let mut messages = Vec::new();
    if let Some(instructions) = request.instructions {
        let system = ChatCompletionRequestSystemMessageArgs::default()
            .content(ChatCompletionRequestSystemMessageContent::Text(
                instructions.to_string(),
            ))
            .build()
            .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
        messages.push(ChatCompletionRequestMessage::System(system));
    }
    for message in &coalesce_assistant(normalize(request.items)) {
        // This protocol carries no reasoning, and a turn that holds nothing
        // else would be an empty message the server will not accept.
        if message.role == Role::Assistant
            && message.text().is_empty()
            && !message
                .parts
                .iter()
                .any(|part| matches!(part, Part::ToolCall { .. }))
        {
            continue;
        }
        messages.push(render_message(message)?);
    }

    let mut builder = CreateChatCompletionRequestArgs::default();
    builder
        .model(request.model)
        .messages(messages)
        .stream(request.stream);
    let tool_aliases = if request.tools.is_empty() {
        ToolAliases::default()
    } else {
        let (tools, aliases) = tools::wire_tools(request.tools)?;
        builder.tools(tools::openai_function_tools(&tools));
        aliases
    };
    if let Some(max_output_tokens) = request.max_output_tokens {
        builder.max_tokens(max_output_tokens);
    }
    let body = builder
        .build()
        .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
    Ok((body, tool_aliases))
}

fn render_message(message: &Message) -> Result<ChatCompletionRequestMessage, ProviderError> {
    const CONTEXT: &str = "chat completions message";

    // A tool result is its own message with the `tool` role, addressed back to
    // the call it answers.
    if let Some(Part::ToolResult {
        call_id, content, ..
    }) = message.parts.first()
    {
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .content(ChatCompletionRequestToolMessageContent::Text(
                content.clone(),
            ))
            .tool_call_id(call_id.clone())
            .build()
            .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
        return Ok(ChatCompletionRequestMessage::Tool(tool));
    }

    let text = message.text();
    let calls: Vec<ChatCompletionMessageToolCalls> = message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::ToolCall {
                id,
                name,
                arguments,
            } => Some(ChatCompletionMessageToolCalls::Function(
                ChatCompletionMessageToolCall {
                    id: id.clone(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                },
            )),
            _ => None,
        })
        .collect();

    match message.role {
        // Chat Completions has no developer role; system is the portable
        // spelling and is what every compatible server understands.
        Role::System | Role::Developer => {
            let system = ChatCompletionRequestSystemMessageArgs::default()
                .content(ChatCompletionRequestSystemMessageContent::Text(text))
                .build()
                .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
            Ok(ChatCompletionRequestMessage::System(system))
        }
        Role::User => {
            let user = ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Text(text))
                .build()
                .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
            Ok(ChatCompletionRequestMessage::User(user))
        }
        Role::Assistant => {
            let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
            if !text.is_empty() {
                builder.content(ChatCompletionRequestAssistantMessageContent::Text(text));
            }
            if !calls.is_empty() {
                builder.tool_calls(calls);
            }
            let assistant = builder
                .build()
                .map_err(|source| ProviderError::sdk_build(CONTEXT, source))?;
            Ok(ChatCompletionRequestMessage::Assistant(assistant))
        }
    }
}

// ---------------------------------------------------------------------------
// Stream decoding
// ---------------------------------------------------------------------------

/// The sentinel some servers send as the final event of a stream.
const DONE_SENTINEL: &str = "[DONE]";

/// Decodes the Chat Completions streaming vocabulary.
///
/// A turn arrives as one delta per chunk and ends either with a chunk that
/// carries a `finish_reason` or with the `[DONE]` sentinel, so the assistant
/// message and any tool calls are only complete at the end of the stream.
#[derive(Default)]
pub struct ChatCompletionsDecoder {
    /// The flat names the request exposed, used to name the calls that come
    /// back with a namespace missing from the wire.
    tool_aliases: Arc<ToolAliases>,
    response_id: Option<String>,
    created: bool,
    text: String,
    /// Reasoning text, which the protocol reports but does not replay.
    reasoning: String,
    /// The item the client has been told is open, if any.
    open_item: OpenItem,
    tool_calls: Vec<PartialToolCall>,
    finish_reason: Option<String>,
    usage: Option<TokenUsage>,
    completed: bool,
}

#[derive(Default)]
struct PartialToolCall {
    call_id: String,
    name: String,
    arguments: String,
    announced: bool,
}

/// One decoded chunk, in the SDK's shape plus the pieces the mapper reads in
/// their raw wire spelling.
struct Chunk {
    response: CreateChatCompletionStreamResponse,
    /// The first choice's `finish_reason` as the server spelled it. The SDK
    /// models it as a closed enum, and servers emit values outside it.
    finish_reason: Option<String>,
    /// Reasoning text removed from the payload, which the SDK does not model.
    reasoning: String,
}

impl SseEventDecoder for ChatCompletionsDecoder {
    fn decode(&mut self, _event: &str, data: &str) -> Result<Vec<ResponseEvent>, ApiError> {
        if data.trim() == DONE_SENTINEL {
            return Ok(self.complete());
        }
        if let Some(message) = error_message(data) {
            return Err(ApiError::Stream(message));
        }
        let Some(chunk) = chunk(data) else {
            return Ok(Vec::new());
        };

        let mut events = Vec::new();
        // A server that omits the chunk identifier still gets a turn; it just
        // has no response id to report.
        if !chunk.response.id.is_empty() && !self.created {
            let id = chunk.response.id.clone();
            self.created = true;
            self.response_id = Some(id.clone());
            events.push(ResponseEvent::Created {
                response_id: Some(id),
            });
        }
        if let Some(usage) = chunk.response.usage {
            self.usage = Some(token_usage(usage));
        }
        // Reasoning is reported as a reasoning item so it reaches the client,
        // but it is not replayed to the model: this protocol has no way to send
        // it back.
        if !chunk.reasoning.is_empty() {
            let mut opening = self.open_reasoning_item();
            self.reasoning.push_str(&chunk.reasoning);
            opening.push(ResponseEvent::ReasoningSummaryDelta {
                delta: chunk.reasoning,
                summary_index: 0,
            });
            events.extend(opening);
        }

        // Codex requests one completion, so only the first choice is a turn.
        let Some(choice) = chunk.response.choices.into_iter().next() else {
            return Ok(events);
        };
        if let Some(reason) = chunk.finish_reason {
            self.finish_reason = Some(reason);
        }
        if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
            events.extend(self.open_message_item());
            self.text.push_str(&text);
            events.push(ResponseEvent::OutputTextDelta(text));
        }
        for call in choice.delta.tool_calls.unwrap_or_default() {
            let index = call.index as usize;
            let partial = partial_tool_call(&mut self.tool_calls, index);
            if let Some(id) = call.id {
                partial.call_id = id;
            }
            let mut argument_delta = None;
            if let Some(function) = call.function {
                if let Some(name) = function.name
                    && partial.name.is_empty()
                {
                    partial.name = name;
                }
                if let Some(arguments) = function.arguments
                    && !arguments.is_empty()
                {
                    partial.arguments.push_str(&arguments);
                    argument_delta = Some(arguments);
                }
            }
            // The call is only addressable once its id and name have arrived,
            // which can be a chunk later than its first argument fragment.
            if !partial.announced && !partial.call_id.is_empty() && !partial.name.is_empty() {
                partial.announced = true;
                let tool = self.tool_aliases.resolve_or_plain(&partial.name);
                events.push(ResponseEvent::OutputItemAdded(
                    events::function_call_started(&partial.call_id, &tool),
                ));
            }
            if partial.announced
                && let Some(delta) = argument_delta
            {
                events.push(ResponseEvent::ToolCallInputDelta {
                    item_id: partial.call_id.clone(),
                    call_id: Some(partial.call_id.clone()),
                    delta,
                });
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<ResponseEvent>, ApiError> {
        Ok(self.complete())
    }
}

impl ChatCompletionsDecoder {
    /// A decoder that names the calls it decodes the way the request exposed
    /// them.
    pub(crate) fn with_tool_aliases(tool_aliases: Arc<ToolAliases>) -> Self {
        Self {
            tool_aliases,
            ..Self::default()
        }
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

    /// Announces the reasoning item a reasoning delta streams into, closing the
    /// item that was open before it.
    fn open_reasoning_item(&mut self) -> Vec<ResponseEvent> {
        if self.open_item == OpenItem::Reasoning {
            return Vec::new();
        }
        let mut events = self.close_open_item();
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

    /// Closes the open reasoning item, whether the stream said so or ended.
    fn flush_reasoning(&mut self) -> Vec<ResponseEvent> {
        if self.open_item != OpenItem::Reasoning {
            return Vec::new();
        }
        self.open_item = OpenItem::None;
        vec![ResponseEvent::OutputItemDone(events::reasoning_finished(
            std::mem::take(&mut self.reasoning),
            None,
        ))]
    }

    /// The terminal events for this turn, produced once.
    fn complete(&mut self) -> Vec<ResponseEvent> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;

        let mut events = Vec::new();
        let calls: Vec<PartialToolCall> = std::mem::take(&mut self.tool_calls)
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .collect();
        // A stream that ends with an item open still owes it: a reasoning item
        // belongs before the answer it explains, and the message that ends the
        // turn is the answer.
        if self.open_item == OpenItem::Reasoning {
            events.extend(self.close_open_item());
        }
        if self.open_item == OpenItem::Message && !self.text.is_empty() {
            let phase = if calls.is_empty() {
                MessagePhase::FinalAnswer
            } else {
                MessagePhase::Commentary
            };
            self.open_item = OpenItem::None;
            events.push(ResponseEvent::OutputItemDone(
                events::assistant_message_finished(std::mem::take(&mut self.text), phase),
            ));
        }
        for call in &calls {
            let tool = self.tool_aliases.resolve_or_plain(&call.name);
            events.push(ResponseEvent::OutputItemDone(
                events::function_call_finished(&call.call_id, &tool, &call.arguments),
            ));
        }
        events.push(ResponseEvent::Completed {
            response_id: self.response_id.clone().unwrap_or_default(),
            token_usage: self.usage.take(),
            usage_metadata: None,
            end_turn: Some(calls.is_empty() && self.finish_reason.as_deref() != Some("tool_calls")),
        });
        events
    }
}

fn partial_tool_call(calls: &mut Vec<PartialToolCall>, index: usize) -> &mut PartialToolCall {
    while calls.len() <= index {
        calls.push(PartialToolCall::default());
    }
    &mut calls[index]
}

/// Decodes one chunk through the SDK's stream-response type.
fn chunk(data: &str) -> Option<Chunk> {
    let mut value: Value = serde_json::from_str(data).ok()?;
    let finish_reason = take_finish_reason(&mut value);
    let reasoning = take_reasoning(&mut value);
    normalize_chunk(&mut value);
    match serde_json::from_value::<CreateChatCompletionStreamResponse>(value) {
        Ok(response) => Some(Chunk {
            response,
            finish_reason,
            reasoning,
        }),
        Err(error) => {
            debug!(
                error = %error,
                payload_bytes = data.len(),
                "Failed to parse Chat Completions chunk"
            );
            None
        }
    }
}

/// Takes the first choice's `finish_reason` out of the payload as a string.
fn take_finish_reason(value: &mut Value) -> Option<String> {
    let reason = value
        .get_mut("choices")?
        .as_array_mut()?
        .first_mut()?
        .get_mut("finish_reason")?;
    match std::mem::replace(reason, Value::Null) {
        Value::String(reason) => Some(reason),
        _ => None,
    }
}

/// Takes reasoning text, reported under either of two names in the wild, out
/// of the payload so it can be reported as a reasoning item.
fn take_reasoning(value: &mut Value) -> String {
    let mut text = String::new();
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return text;
    };
    for delta in choices
        .iter_mut()
        .filter_map(|choice| choice.get_mut("delta"))
        .filter_map(Value::as_object_mut)
    {
        for field in ["reasoning_content", "reasoning"] {
            if let Some(Value::String(fragment)) = delta.remove(field) {
                text.push_str(&fragment);
            }
        }
    }
    text
}

/// Fills the fields the SDK types require and coerces the deviations
/// compatible servers are known to send, so a chunk that is merely untidy is
/// decoded instead of dropped.
fn normalize_chunk(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    insert_string(object, "id", "");
    insert_string(object, "object", "chat.completion.chunk");
    insert_string(object, "model", "");
    insert_number(object, "created");
    match object.get_mut("usage") {
        Some(usage) if usage.is_object() => normalize_usage(usage),
        // Usage is either an object or absent, which the SDK models as `None`.
        Some(usage) if !usage.is_null() => *usage = Value::Null,
        _ => {}
    }

    let Some(choices) = object.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices.iter_mut().filter_map(Value::as_object_mut) {
        insert_number(choice, "index");
        if !choice.get("delta").is_some_and(Value::is_object) {
            choice.insert("delta".to_string(), json!({}));
        }
        normalize_delta(choice);
    }
}

fn normalize_delta(choice: &mut Map<String, Value>) {
    let Some(delta) = choice.get_mut("delta").and_then(Value::as_object_mut) else {
        return;
    };
    // Some servers send the role the SDK does not know, which would otherwise
    // reject the whole chunk. The role is not used by the mapper.
    if let Some(role) = delta.get("role")
        && serde_json::from_value::<OpenAiRole>(role.clone()).is_err()
    {
        delta.remove("role");
    }
    if let Some(content) = delta.get_mut("content") {
        match content {
            // OpenAI sends a string, a few compatible servers an array of parts.
            Value::Array(parts) => {
                let mut text = String::new();
                for part in parts.iter() {
                    if let Some(fragment) = part.get("text").and_then(Value::as_str) {
                        text.push_str(fragment);
                    }
                }
                *content = Value::String(text);
            }
            Value::String(_) | Value::Null => {}
            _ => *content = Value::Null,
        }
    }
    let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) else {
        return;
    };
    for call in calls.iter_mut().filter_map(Value::as_object_mut) {
        insert_number(call, "index");
    }
}

/// Reports usage the way codex accounts for it.
fn token_usage(usage: CompletionUsage) -> TokenUsage {
    let total_tokens = if usage.total_tokens > 0 {
        i64::from(usage.total_tokens)
    } else {
        i64::from(usage.prompt_tokens) + i64::from(usage.completion_tokens)
    };
    TokenUsage {
        input_tokens: i64::from(usage.prompt_tokens),
        cached_input_tokens: i64::from(
            usage
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0),
        ),
        cache_write_input_tokens: 0,
        output_tokens: i64::from(usage.completion_tokens),
        reasoning_output_tokens: i64::from(
            usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or(0),
        ),
        total_tokens,
        codex_rollout_budget_units: None,
    }
}

/// Fills a token count the SDK type requires, which a server omitting it
/// (usage is reported incrementally) would otherwise fail on.
fn normalize_usage(usage: &mut Value) {
    let Some(object) = usage.as_object_mut() else {
        return;
    };
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        insert_number(object, field);
    }
}

/// Inserts a numeric field the SDK type requires when it is missing or null.
fn insert_number(object: &mut Map<String, Value>, field: &str) {
    if !object.get(field).is_some_and(Value::is_number) {
        object.insert(field.to_string(), json!(0));
    }
}

/// Inserts a string field the SDK type requires when it is missing or null.
fn insert_string(object: &mut Map<String, Value>, field: &str, default: &str) {
    if !object.get(field).is_some_and(Value::is_string) {
        object.insert(field.to_string(), Value::String(default.to_string()));
    }
}

/// The message of a provider-sent error, when the chunk carries one.
fn error_message(data: &str) -> Option<String> {
    let value: Value = serde_json::from_str(data).ok()?;
    let error = value.get("error")?.as_object()?;
    Some(
        error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| Value::Object(error.clone()).to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(items: &'a [ResponseItem], tools: &'a [ToolSpec]) -> ChatRequest<'a> {
        ChatRequest {
            model: "local-model",
            instructions: Some("be terse"),
            items,
            tools,
            stream: true,
            max_output_tokens: Some(256),
            effort: None,
        }
    }

    #[test]
    fn effort_goes_out_as_codex_wrote_it() {
        let items: [ResponseItem; 0] = [];
        for effort in [
            ReasoningEffort::Low,
            // A mode this API documents elsewhere but this field does not:
            // it is still what codex asked for, so it is what goes out.
            ReasoningEffort::Adaptive,
        ] {
            let mut request = request(&items, &[]);
            request.effort = Some(&effort);
            let (body, _) = build(&request).expect("build body");
            assert_eq!(body["reasoning_effort"], effort.as_str());
        }
    }

    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ReasoningItemReasoningSummary;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::openai_models::ReasoningEffort;
    use codex_tools::ToolSpec;

    #[test]
    fn assistant_text_and_tool_call_share_one_message() {
        let items = [
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![codex_protocol::models::ContentItem::OutputText {
                    text: "checking".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: "{\"command\":\"ls\"}".to_string(),
                encrypted_function_args: None,
                call_id: "call-1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let (body, _) = build(&request(&items, &[])).expect("build body");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(body["model"], "local-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "checking");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "shell");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call-1");
    }

    #[test]
    fn tool_results_become_the_tool_role() {
        let items = [ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some("call-1".to_string()),
            name: Some("shell".to_string()),
            namespace: None,
            output: codex_protocol::models::FunctionCallOutputPayload::from_text(
                "total 0".to_string(),
            ),
            internal_chat_message_metadata_passthrough: None,
        }];
        let (body, _) = build(&request(&items, &[])).expect("build body");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call-1");
        assert_eq!(messages[1]["content"], "total 0");
    }

    fn decoded(chunks: &[&str]) -> Vec<ResponseEvent> {
        let mut decoder = ChatCompletionsDecoder::default();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.decode("", chunk).expect("decode chunk"));
        }
        events
    }

    #[test]
    fn text_streams_into_one_final_message() {
        let events = decoded(&[
            r#"{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}"#,
            "[DONE]",
        ]);

        assert!(matches!(
            events[0],
            ResponseEvent::Created { ref response_id } if response_id.as_deref() == Some("chatcmpl-1")
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
                ref response_id,
                end_turn: Some(true),
                token_usage: Some(ref usage),
                ..
            } if response_id == "chatcmpl-1" && usage.input_tokens == 10 && usage.cached_input_tokens == 4
        ));
    }

    #[test]
    fn streamed_tool_arguments_become_one_call() {
        let events = decoded(&[
            r#"{"id":"chatcmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-9","function":{"name":"shell","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-2","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ]);

        assert!(matches!(
            events[1],
            ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { ref name, ref call_id, .. })
                if name == "shell" && call_id == "call-9"
        ));
        assert!(matches!(
            events[2],
            ResponseEvent::ToolCallInputDelta { ref delta, .. } if delta == "{\"command\":"
        ));
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { ref arguments, .. })
                if arguments == "{\"command\":\"ls\"}"
        ));
        assert!(matches!(
            events[5],
            ResponseEvent::Completed {
                end_turn: Some(false),
                ..
            }
        ));
    }

    /// Servers add fields, omit the ones the SDK requires, and spell
    /// `finish_reason` their own way. None of that may cost the turn its text.
    #[test]
    fn untidy_chunks_still_decode() {
        let events = decoded(&[
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"text","text":"Hi"},{"type":"text","text":" there"}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"call-7","function":{"name":"shell"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{}"}}]},"finish_reason":"stop_sequence"}],"usage":{"prompt_tokens":3},"extra":{"vendor":"x"}}"#,
            "[DONE]",
        ]);

        assert!(matches!(events[0], ResponseEvent::OutputItemAdded(_)));
        assert!(
            matches!(events[1], ResponseEvent::OutputTextDelta(ref text) if text == "Hi there")
        );
        assert!(matches!(
            events[2],
            ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { ref call_id, .. })
                if call_id == "call-7"
        ));
        assert!(matches!(
            events[3],
            ResponseEvent::ToolCallInputDelta { ref delta, .. } if delta == "{}"
        ));
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Message { ref content, .. })
                if content == &vec![ContentItem::OutputText { text: "Hi there".to_string() }]
        ));
        assert!(matches!(
            events[6],
            ResponseEvent::Completed {
                token_usage: Some(ref usage),
                end_turn: Some(false),
                ..
            } if usage.input_tokens == 3 && usage.total_tokens == 3
        ));
    }

    #[test]
    fn reasoning_content_becomes_a_reasoning_item() {
        let events = decoded(&[
            r#"{"id":"chatcmpl-3","choices":[{"delta":{"reasoning_content":"weigh"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-3","choices":[{"delta":{"reasoning_content":"ing"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-3","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);

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
            ResponseEvent::ReasoningSummaryDelta { ref delta, .. } if delta == "ing"
        ));
        // The reasoning item closes before the message opens, so the text that
        // follows it is announced as the item it belongs to.
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                ref summary,
                encrypted_content: None,
                ..
            }) if summary == &vec![ReasoningItemReasoningSummary::SummaryText {
                    text: "weighing".to_string(),
                }]
        ));
        assert!(matches!(
            events[5],
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            events[7],
            ResponseEvent::OutputItemDone(ResponseItem::Message { ref content, .. })
                if content == &vec![ContentItem::OutputText { text: "Hi".to_string() }]
        ));
        crate::events::assert_deltas_have_an_open_item(&events);
    }

    #[test]
    fn reasoning_between_two_text_chunks_splits_the_messages() {
        // A model that thinks again mid-answer owes the loop an item for the
        // text that follows, or its deltas have nothing to attach to.
        let events = decoded(&[
            r#"{"id":"chatcmpl-4","choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-4","choices":[{"delta":{"reasoning_content":"why"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-4","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);

        crate::events::assert_deltas_have_an_open_item(&events);
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
    fn an_error_chunk_ends_the_stream() {
        let mut decoder = ChatCompletionsDecoder::default();
        let error = decoder
            .decode("", r#"{"error":{"message":"model overloaded"}}"#)
            .expect_err("error chunk must fail");
        assert_eq!(error.to_string(), "stream error: model overloaded");
    }

    #[test]
    fn an_empty_stream_still_completes_the_turn() {
        let mut decoder = ChatCompletionsDecoder::default();
        let events = decoder.finish().expect("finish stream");
        assert!(matches!(
            events.as_slice(),
            [ResponseEvent::Completed {
                end_turn: Some(true),
                ..
            }]
        ));
    }

    /// A namespaced tool crosses over under one flat name, and the call that
    /// comes back is named the way the plan exposed it rather than as the
    /// provider spelled it.
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
        let mut decoder = ChatCompletionsDecoder::with_tool_aliases(Arc::new(aliases));

        let mut events = Vec::new();
        for chunk in [
            r#"{"id":"chatcmpl-3","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-7","function":{"name":"mcp__sample__search","arguments":"{\"query\":\"rust\"}"}}]},"finish_reason":null}]}"#,
            "[DONE]",
        ] {
            events.extend(decoder.decode("", chunk).expect("decode chunk"));
        }

        assert!(events.iter().any(|event| matches!(
            event,
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, namespace, .. })
                if name == "search" && namespace.as_deref() == Some("mcp__sample__")
        )));
    }
}
