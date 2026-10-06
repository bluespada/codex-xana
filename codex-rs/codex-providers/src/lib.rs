//! Wire-protocol backends for the model providers Codex can talk to.
//!
//! The agent loop speaks one vocabulary: it consumes response events from
//! `codex-api` and produces `ResponseItem`s. Everything below that line is
//! owned here. Each [`WireApi`] maps to a [`WireProtocol`] that knows its
//! endpoint, its tool schema shape, and how to render a transcript into a
//! request body, so callers never match on the wire protocol themselves.
//!
//! The wire vocabulary of each protocol comes from the vendor SDK rather than
//! from structs written here: `async-openai` for Chat Completions and, once it
//! lands, for Responses, and `claudius` for the Messages API. The SDKs supply
//! the typed request and stream models; codex keeps the transport, so auth,
//! provider headers, query parameters, the no-proxy and custom-CA policy,
//! telemetry and the retry loop stay where they already were. `claudius` cannot
//! be the client for the same reason: it builds its headers from a single API
//! key and offers no way to add a provider's own, no query parameters, and no
//! client injection.
//!
//! What a protocol rendered here does not inherit from the Responses path:
//! guardian review metadata, the `x-openai-*` request headers, zstd request
//! compression, extension response interceptors, and the rollout inference
//! trace, which is disabled here as it is for a Responses attempt that carries
//! no trace. Those are Responses-only concerns rather than omissions of a
//! protocol. Auth recovery and backoff are shared from `codex-api`: the
//! endpoint session retries, and `map_api_error` refreshes credentials on 401.
//!
//! Reasoning crosses turns in both directions. Messages thinking blocks are
//! carried as reasoning items, the signature the provider issued travels with
//! them, and they are replayed as signed thinking blocks; Chat Completions has
//! no field in which to send reasoning back, so its text is reported for
//! display only. The effort codex resolved for the turn is written to each
//! protocol's own field verbatim: nothing here rewrites the value, translates it
//! between vocabularies, or hides a provider's rejection of it.

mod anthropic;
mod completions;
mod events;
mod tools;
mod transcript;

pub use anthropic::AnthropicDecoder;
pub use anthropic::DEFAULT_MAX_OUTPUT_TOKENS;
pub use completions::ChatCompletionsDecoder;
pub use tools::ToolAliases;
pub use tools::wire_tool_name;
pub use transcript::Message;
pub use transcript::Part;
pub use transcript::Role;
pub use transcript::normalize;

use codex_api::SseEventDecoder;
use codex_model_provider_info::WireApi;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::ToolSpec;
use serde_json::Value;
use std::sync::Arc;

/// Errors raised while rendering a provider request.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The configured wire protocol has no request builder in this build.
    #[error("wire_api = \"{0}\" has no request builder")]
    UnsupportedWireApi(WireApi),
    /// A tool call carried arguments that were not valid JSON.
    #[error("tool call `{name}` carried arguments that are not valid JSON: {source}")]
    InvalidToolArguments {
        name: String,
        source: serde_json::Error,
    },
    /// A value that should have been serializable was not.
    #[error("could not serialize the {context} request: {source}")]
    Serialize {
        context: &'static str,
        source: serde_json::Error,
    },
    /// The protocol's SDK rejected the request it was asked to build.
    #[error("could not build the {context} request: {message}")]
    SdkBuild {
        context: &'static str,
        message: String,
    },
}

impl ProviderError {
    /// Wraps an SDK build failure, keeping its message for the caller.
    pub(crate) fn sdk_build(context: &'static str, source: impl std::fmt::Display) -> Self {
        Self::SdkBuild {
            context,
            message: source.to_string(),
        }
    }
}

/// Provider-neutral inputs from which every wire protocol is rendered.
#[derive(Debug)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    /// System-level instructions. Some protocols send these as a top-level
    /// field rather than as a message.
    pub instructions: Option<&'a str>,
    /// The Responses-shaped transcript to render.
    pub items: &'a [ResponseItem],
    /// Tools offered to the model, in the canonical Responses shape.
    pub tools: &'a [ToolSpec],
    pub stream: bool,
    /// Upper bound on generated tokens. The Messages API requires one, so a
    /// protocol with that requirement falls back to its own default.
    pub max_output_tokens: Option<u32>,
    /// The reasoning effort codex resolved for this turn, written to the
    /// protocol's own field as it stands.
    pub effort: Option<&'a ReasoningEffort>,
}

/// A rendered request body and the path it is posted to.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestBody {
    /// Path relative to the provider base URL, for example `/messages`. The
    /// base URL carries the API version, as `https://api.openai.com/v1` does.
    pub path: &'static str,
    pub body: Value,
}

/// The outcome of rendering a request.
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltRequest {
    /// `codex-api` already serializes the Responses body, including its
    /// prompt-cache keys, text controls, and raw tool JSON. Keep using it.
    ResponsesApi,
    /// A body this crate rendered, and the tools its flat names stand for.
    Json {
        request: RequestBody,
        tool_aliases: Arc<ToolAliases>,
    },
}

/// Endpoint and feature set of one wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireProtocol {
    pub wire_api: WireApi,
    /// Path relative to the provider base URL.
    pub path: &'static str,
    /// Whether this build has a WebSocket transport for the protocol.
    pub supports_websockets: bool,
    /// Whether the protocol carries reasoning items across turns.
    pub supports_reasoning: bool,
    /// Whether the protocol accepts a prompt-cache key.
    pub supports_prompt_cache: bool,
}

impl WireProtocol {
    /// The descriptor for a configured wire protocol.
    pub fn for_api(wire_api: WireApi) -> Self {
        match wire_api {
            WireApi::OpenAiResponses => Self {
                wire_api,
                path: "/responses",
                supports_websockets: true,
                supports_reasoning: true,
                supports_prompt_cache: true,
            },
            WireApi::OpenAiCompletions => Self {
                wire_api,
                path: "/chat/completions",
                supports_websockets: false,
                supports_reasoning: false,
                supports_prompt_cache: false,
            },
            WireApi::AnthropicMessages => Self {
                wire_api,
                // Relative to a base URL that already carries the version, as
                // `https://api.anthropic.com/v1` does.
                path: "/messages",
                supports_websockets: false,
                supports_reasoning: false,
                supports_prompt_cache: false,
            },
        }
    }

    /// Render the request body for this protocol.
    pub fn build_request(&self, request: &ChatRequest<'_>) -> Result<BuiltRequest, ProviderError> {
        match self.wire_api {
            WireApi::OpenAiResponses => Ok(BuiltRequest::ResponsesApi),
            WireApi::OpenAiCompletions => {
                let (body, tool_aliases) = completions::build(request)?;
                Ok(BuiltRequest::Json {
                    request: RequestBody {
                        path: self.path,
                        body,
                    },
                    tool_aliases: Arc::new(tool_aliases),
                })
            }
            WireApi::AnthropicMessages => {
                let (body, tool_aliases) = anthropic::build(request)?;
                Ok(BuiltRequest::Json {
                    request: RequestBody {
                        path: self.path,
                        body,
                    },
                    tool_aliases: Arc::new(tool_aliases),
                })
            }
        }
    }

    /// The decoder for the events this protocol streams back, or `None` when
    /// the protocol's own transport decodes them, as the Responses API does.
    ///
    /// The aliases come from the request that was just rendered: a flat name on
    /// the wire only resolves back to its tool with the mapping that sent it.
    pub fn event_decoder(
        &self,
        tool_aliases: Arc<ToolAliases>,
    ) -> Option<Box<dyn SseEventDecoder>> {
        match self.wire_api {
            WireApi::OpenAiResponses => None,
            WireApi::OpenAiCompletions => Some(Box::new(
                ChatCompletionsDecoder::with_tool_aliases(tool_aliases),
            )),
            WireApi::AnthropicMessages => {
                Some(Box::new(AnthropicDecoder::with_tool_aliases(tool_aliases)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_wire_api_has_a_descriptor() {
        for api in [
            WireApi::OpenAiResponses,
            WireApi::OpenAiCompletions,
            WireApi::AnthropicMessages,
        ] {
            assert_eq!(WireProtocol::for_api(api).wire_api, api);
        }
    }

    #[test]
    fn responses_keeps_its_own_serializer() {
        let protocol = WireProtocol::for_api(WireApi::OpenAiResponses);
        let request = ChatRequest {
            model: "gpt-test",
            instructions: None,
            items: &[],
            tools: &[],
            stream: true,
            max_output_tokens: None,
            effort: None,
        };
        assert!(matches!(
            protocol.build_request(&request),
            Ok(BuiltRequest::ResponsesApi)
        ));
    }

    #[test]
    fn a_rendered_request_carries_its_tool_aliases() {
        let protocol = WireProtocol::for_api(WireApi::OpenAiCompletions);
        let tools = [ToolSpec::Function(codex_tools::ResponsesApiTool {
            name: "shell".to_string(),
            description: "runs a command".to_string(),
            strict: false,
            defer_loading: None,
            parameters: codex_tools::JsonSchema::default(),
            output_schema: None,
        })];
        let request = ChatRequest {
            model: "local-model",
            instructions: None,
            items: &[],
            tools: &tools,
            stream: true,
            max_output_tokens: None,
            effort: None,
        };

        let Ok(BuiltRequest::Json {
            request: body,
            tool_aliases,
        }) = protocol.build_request(&request)
        else {
            panic!("completions renders a JSON body");
        };
        assert_eq!(body.path, "/chat/completions");
        assert_eq!(body.body["tools"][0]["function"]["name"], "shell");
        assert_eq!(tool_aliases.resolve("shell"), None);
    }
}
