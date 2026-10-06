use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

const REQUEST_ID_HEADER: &str = "x-request-id";

/// Translates one provider's stream vocabulary into the response events the
/// agent loop consumes.
///
/// Implementations live next to the protocol they decode. The SSE framing,
/// idle timeout, telemetry, and channel plumbing stay in this crate, so a
/// decoder only has to know what its provider sends.
pub trait SseEventDecoder: Send {
    /// Decode a single server-sent event.
    ///
    /// `event` is the SSE `event:` field, empty when the stream does not name
    /// its events. Returning no events skips the event; returning an error
    /// ends the stream.
    fn decode(&mut self, event: &str, data: &str) -> Result<Vec<ResponseEvent>, ApiError>;

    /// Decode anything owed when the stream ends without a terminal event,
    /// such as a stream that closes after the last content block.
    fn finish(&mut self) -> Result<Vec<ResponseEvent>, ApiError> {
        Ok(Vec::new())
    }
}

impl SseEventDecoder for Box<dyn SseEventDecoder> {
    fn decode(&mut self, event: &str, data: &str) -> Result<Vec<ResponseEvent>, ApiError> {
        (**self).decode(event, data)
    }

    fn finish(&mut self) -> Result<Vec<ResponseEvent>, ApiError> {
        (**self).finish()
    }
}

/// Drive an SSE byte stream through `decoder` and forward the decoded events.
pub(crate) fn spawn_decoded_stream<D: SseEventDecoder + 'static>(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    decoder: D,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(run_decoded_stream(
        stream_response.bytes,
        tx_event,
        idle_timeout,
        telemetry,
        decoder,
    ));

    ResponseStream {
        rx_event,
        upstream_request_id,
        interrupt: None,
    }
}

async fn run_decoded_stream<D: SseEventDecoder>(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    mut decoder: D,
) {
    let mut stream = stream.eventsource();
    loop {
        let start = Instant::now();
        let response = tokio::select! {
            biased;
            _ = tx_event.closed() => return,
            response = timeout(idle_timeout, stream.next()) => response,
        };
        if let Some(telemetry) = telemetry.as_ref() {
            telemetry.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("SSE Error: {e:#}");
                let error = match e {
                    eventsource_stream::EventStreamError::Transport(
                        error @ codex_client::TransportError::Policy(_),
                    ) => ApiError::Transport(error),
                    error => ApiError::Stream(error.to_string()),
                };
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Ok(None) => {
                let finished = decoder.finish();
                let _ = forward_events(&tx_event, finished).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        trace!("SSE event: {}", &sse.data);
        let events = decoder.decode(&sse.event, &sse.data);
        if !forward_events(&tx_event, events).await {
            return;
        }
    }
}

/// Send decoded events, reporting whether the consumer is still listening.
async fn forward_events(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    events: Result<Vec<ResponseEvent>, ApiError>,
) -> bool {
    match events {
        Ok(events) => {
            for event in events {
                if tx_event.send(Ok(event)).await.is_err() {
                    return false;
                }
            }
            true
        }
        Err(error) => {
            let _ = tx_event.send(Err(error)).await;
            false
        }
    }
}
