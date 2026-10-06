use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::sse::SseEventDecoder;
use crate::sse::spawn_decoded_stream;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde_json::Value;
use std::sync::Arc;
use tracing::instrument;

/// Streams a JSON request whose responses arrive as server-sent events.
///
/// The Responses API has its own client because it also has a WebSocket
/// transport. This one covers protocols that post a JSON body and read a
/// stream back: they differ only in their endpoint path and in the
/// [`SseEventDecoder`] that translates their events.
pub struct EventStreamClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> EventStreamClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    /// Post `body` to `path` and decode the response stream with `decoder`.
    ///
    /// `path` is resolved against the provider base URL for this request, so
    /// providers that route endpoints differently still work.
    #[instrument(
        name = "event_stream.stream",
        level = "info",
        skip_all,
        fields(http.method = "POST", api.path = path)
    )]
    pub async fn stream<D>(
        &self,
        path: &str,
        body: Value,
        extra_headers: HeaderMap,
        compression: Compression,
        decoder: D,
    ) -> Result<ResponseStream, ApiError>
    where
        D: SseEventDecoder + 'static,
    {
        let body = EncodedJsonBody::encode(&body)
            .map_err(|e| ApiError::Stream(format!("failed to encode {path} request: {e}")))?;
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let stream_response = self
            .session
            .stream_encoded_json_with(Method::POST, path, extra_headers, Some(body), |req| {
                req.headers.insert(
                    http::header::ACCEPT,
                    HeaderValue::from_static("text/event-stream"),
                );
                req.compression = request_compression;
            })
            .await?;

        Ok(spawn_decoded_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            decoder,
        ))
    }
}
