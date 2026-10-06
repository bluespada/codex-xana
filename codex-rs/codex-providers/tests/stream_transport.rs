#![allow(clippy::expect_used)]
//! Transport tests that drive the real request and stream plumbing.
//!
//! Each test hands a canned provider stream to the same client and decoder the
//! agent loop uses, so the endpoint path, the request headers, the SSE framing,
//! and the decoded event order are all exercised together.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::Compression;
use codex_api::EventStreamClient;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::RetryConfig;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_model_provider_info::WireApi;
use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use codex_providers::ToolAliases;
use codex_providers::WireProtocol;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;

#[derive(Clone)]
struct FixtureTransport {
    body: String,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl FixtureTransport {
    fn new(body: String) -> Self {
        Self {
            body,
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().expect("request log").clone()
    }
}

impl HttpTransport for FixtureTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build(
            "execute should not run for a streaming request".to_string(),
        ))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.requests.lock().expect("request log").push(req);
        let body = self.body.clone();
        let bytes = futures::stream::iter(vec![Ok::<Bytes, TransportError>(Bytes::from(body))]);
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(bytes),
        })
    }
}

#[derive(Clone)]
struct TestAuth;

impl AuthProvider for TestAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert("x-test-auth", "applied".parse().expect("header value"));
    }
}

fn provider(base_url: &str) -> Provider {
    Provider {
        name: "fixture".to_string(),
        base_url: base_url.to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(5),
    }
}

/// A stream body in the shape both protocols use: one `data:` line per event.
fn sse_body(events: &[&str]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(event);
        body.push_str("\n\n");
    }
    body
}

/// Run a protocol's decoder over a canned stream, exactly as the agent loop does.
async fn stream_events(
    transport: FixtureTransport,
    wire_api: WireApi,
    base_url: &str,
) -> Vec<ResponseEvent> {
    let protocol = WireProtocol::for_api(wire_api);
    let decoder = protocol
        .event_decoder(Arc::new(ToolAliases::default()))
        .expect("protocol event decoder");
    let client = EventStreamClient::new(transport, provider(base_url), Arc::new(TestAuth));
    let mut stream = client
        .stream(
            protocol.path,
            serde_json::json!({"model": "fixture-model"}),
            HeaderMap::new(),
            Compression::None,
            decoder,
        )
        .await
        .expect("open stream");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("decoded event"));
    }
    events
}

#[tokio::test]
async fn chat_completions_streams_text_and_a_tool_call_end_to_end() {
    let body = sse_body(&[
        r#"{"id":"chatcmpl-7","choices":[{"delta":{"role":"assistant","content":"Reply"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-7","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-3","function":{"name":"shell","arguments":"{\"command\":\"ls\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let transport = FixtureTransport::new(body);
    let observed = transport.clone();
    let events = stream_events(
        transport,
        WireApi::OpenAiCompletions,
        "https://example.com/v1",
    )
    .await;

    assert!(matches!(
        events[0],
        ResponseEvent::Created { ref response_id } if response_id.as_deref() == Some("chatcmpl-7")
    ));
    assert!(matches!(events[1], ResponseEvent::OutputItemAdded(_)));
    assert!(matches!(events[2], ResponseEvent::OutputTextDelta(ref text) if text == "Reply"));
    assert!(matches!(
        events[3],
        ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { ref name, .. }) if name == "shell"
    ));
    assert!(matches!(
        events[5],
        ResponseEvent::OutputItemDone(ResponseItem::Message {
            phase: Some(MessagePhase::Commentary),
            ref content,
            ..
        }) if content == &vec![ContentItem::OutputText { text: "Reply".to_string() }]
    ));
    assert!(matches!(
        events[6],
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { ref arguments, .. })
            if arguments == "{\"command\":\"ls\"}"
    ));
    assert!(matches!(
        events[7],
        ResponseEvent::Completed {
            end_turn: Some(false),
            ..
        }
    ));

    let requests = observed.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://example.com/v1/chat/completions");
    assert_eq!(requests[0].method, http::Method::POST);
    assert_eq!(
        requests[0].headers["accept"]
            .to_str()
            .expect("accept header"),
        "text/event-stream"
    );
    assert_eq!(
        requests[0].headers["x-test-auth"]
            .to_str()
            .expect("auth header"),
        "applied"
    );
}

#[tokio::test]
async fn anthropic_messages_streams_text_and_a_tool_use_end_to_end() {
    let body = sse_body(&[
        r#"{"type":"message_start","message":{"id":"msg_9","type":"message","content":[],"model":"fixture-model","role":"assistant","usage":{"input_tokens":7,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reply"}}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_9","name":"shell","input":{}}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"ls\"}"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let transport = FixtureTransport::new(body);
    let observed = transport.clone();
    let events = stream_events(
        transport,
        WireApi::AnthropicMessages,
        "https://example.com/v1",
    )
    .await;

    assert!(matches!(
        events[0],
        ResponseEvent::Created { ref response_id } if response_id.as_deref() == Some("msg_9")
    ));
    assert!(matches!(events[1], ResponseEvent::OutputItemAdded(_)));
    assert!(matches!(events[2], ResponseEvent::OutputTextDelta(ref text) if text == "Reply"));
    assert!(matches!(
        events[3],
        ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { ref call_id, .. })
            if call_id == "toolu_9"
    ));
    assert!(matches!(
        events[5],
        ResponseEvent::OutputItemDone(ResponseItem::Message {
            phase: Some(MessagePhase::Commentary),
            ..
        })
    ));
    assert!(matches!(
        events[6],
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { ref arguments, .. })
            if arguments == "{\"command\":\"ls\"}"
    ));
    assert!(matches!(
        events[7],
        ResponseEvent::Completed {
            end_turn: Some(false),
            token_usage: Some(ref usage),
            ..
        } if usage.input_tokens == 7 && usage.output_tokens == 9
    ));

    let requests = observed.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://example.com/v1/messages");
    assert_eq!(
        requests[0].headers["accept"]
            .to_str()
            .expect("accept header"),
        "text/event-stream"
    );
}

#[tokio::test]
async fn anthropic_messages_streams_thinking_before_the_answer() {
    let body = sse_body(&[
        r#"{"type":"message_start","message":{"id":"msg_10","type":"message","content":[],"model":"fixture-model","role":"assistant","usage":{"input_tokens":5,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-9"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let events = stream_events(
        FixtureTransport::new(body),
        WireApi::AnthropicMessages,
        "https://example.com/v1",
    )
    .await;

    assert!(matches!(
        events[1],
        ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. })
    ));
    assert!(matches!(
        events[2],
        ResponseEvent::ReasoningSummaryDelta { ref delta, .. } if delta == "hmm"
    ));
    assert!(matches!(
        events[3],
        ResponseEvent::OutputItemDone(ResponseItem::Reasoning { ref encrypted_content, .. })
            if encrypted_content.as_deref() == Some("sig-9")
    ));
}
