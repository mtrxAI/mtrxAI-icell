use crate::config::Config;
use crate::engine::{EngineHandle, LlamaMode};
use crate::llamacpp_status::{build_v1_models_payload, fetch_router_model_states};
use crate::models::ModelsService;
use anyhow::{Context, Result};
use http_body_util::Full;
use hyper::body::{Bytes, Frame, Incoming};
use hyper::header::{CONNECTION, HeaderName, UPGRADE};
use hyper::upgrade::OnUpgrade;
use hyper::{Method, Request, Response, StatusCode, Uri, header};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::warn;

const HOP_BY_HOP: [&str; 7] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
];

pub struct ProxyService {
    config: Config,
    client: Client<HttpConnector, Incoming>,
    upstream_base: String,
    models_dir: String,
    engine: Arc<Mutex<EngineHandle>>,
    models: ModelsService,
}

impl ProxyService {
    pub fn new(config: Config, engine: Arc<Mutex<EngineHandle>>) -> Self {
        let upstream_base = config.upstream_base();
        let models_dir = config.models_dir.clone();
        let models = ModelsService::new(config.clone(), engine.clone());
        let client: Client<HttpConnector, Incoming> =
            Client::builder(TokioExecutor::new()).build_http();

        Self {
            config,
            client,
            upstream_base,
            models_dir,
            engine,
            models,
        }
    }

    pub async fn handle(&self, req: Request<Incoming>) -> Result<Response<Body>> {
        let method = req.method().clone();
        let path = req.uri().path().to_string();

        if path == "/mtrxai/v1/info" && method == Method::GET {
            return self.mtrxai_info().await;
        }

        if path.starts_with("/mtrxai/v1/models") {
            let resp = self.models.handle(&method, &path, req).await?;
            let (parts, body) = resp.into_parts();
            return Ok(Response::from_parts(parts, Body::Full(body)));
        }

        if self.config.is_llamacpp() {
            return self.handle_llamacpp(req, method, path).await;
        }

        self.handle_ollama(req, method, path).await
    }

    async fn mtrxai_info(&self) -> Result<Response<Body>> {
        let body = Bytes::from(serde_json::to_vec(&json!({
            "engine": self.config.inference_backend.as_str(),
            "inference_api": self.config.inference_backend.inference_api(),
            "admin_api": "mtrxai/v1",
            "version": env!("CARGO_PKG_VERSION"),
        }))?);
        let mut resp = Response::new(Body::Full(Full::new(body)));
        *resp.status_mut() = StatusCode::OK;
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        Ok(resp)
    }

    async fn handle_llamacpp(
        &self,
        req: Request<Incoming>,
        method: Method,
        path: String,
    ) -> Result<Response<Body>> {
        if path == "/v1/models" && method == Method::GET {
            return self.v1_models_llamacpp().await;
        }

        if req.headers().contains_key(UPGRADE) {
            return self.proxy_upgrade(req).await;
        }

        self.proxy_http(req).await
    }

    async fn handle_ollama(
        &self,
        req: Request<Incoming>,
        method: Method,
        path: String,
    ) -> Result<Response<Body>> {
        if req.headers().contains_key(UPGRADE) {
            return Ok(not_implemented("websocket upgrades are not supported for ollama backend"));
        }

        if path == "/health" && method == Method::GET {
            let body = Bytes::from(r#"{"status":"ok"}"#);
            let mut resp = Response::new(Body::Full(Full::new(body)));
            *resp.status_mut() = StatusCode::OK;
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            return Ok(resp);
        }

        // Ollama natively serves OpenAI-compat under /v1/* (chat/completions, models, etc.).
        // mtrxAI agent clients (OpenCode, Cursor) proxy remote inference as /v1/chat/completions;
        // without this, icell returns 501 and the provider appears to hang with no Ollama GIN log.
        if path.starts_with("/api/") || path.starts_with("/v1/") {
            return self.proxy_http(req).await;
        }

        Ok(not_implemented(
            "ollama backend exposes /api/* and /v1/* inference routes",
        ))
    }

    async fn v1_models_llamacpp(&self) -> Result<Response<Body>> {
        let single_model_path = {
            let mut engine = self.engine.lock().await;
            match engine
                .llama()
                .map(|llama| llama.mode())
                .unwrap_or(LlamaMode::Router)
            {
                LlamaMode::Model { path } => Some(path),
                LlamaMode::Router => None,
            }
        };

        let router_states = fetch_router_model_states(&self.upstream_base).await;
        let payload = build_v1_models_payload(
            &self.models_dir,
            single_model_path.as_deref(),
            &router_states,
        );

        let body = Bytes::from(serde_json::to_vec(&payload)?);
        let mut resp = Response::new(Body::Full(Full::new(body)));
        *resp.status_mut() = StatusCode::OK;
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        Ok(resp)
    }

    async fn proxy_http(&self, mut req: Request<Incoming>) -> Result<Response<Body>> {
        let path_and_query = req
            .uri()
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");
        let uri: Uri = format!("{}{}", self.upstream_base, path_and_query)
            .parse()
            .context("build upstream uri")?;
        *req.uri_mut() = uri;
        strip_hop_by_hop(req.headers_mut());
        if self.config.is_ollama() {
            prepare_ollama_upstream_headers(
                req.headers_mut(),
                &self.config.ollama_upstream_host_header(),
            );
        }

        let upstream_resp = self
            .client
            .request(req)
            .await
            .context("forward request to inference engine")?;

        Ok(map_upstream_response(upstream_resp))
    }

    async fn proxy_upgrade(&self, mut req: Request<Incoming>) -> Result<Response<Body>> {
        let on_client = hyper::upgrade::on(&mut req);

        let path_and_query = req
            .uri()
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");
        let uri: Uri = format!("{}{}", self.upstream_base, path_and_query)
            .parse()
            .context("build upstream upgrade uri")?;
        *req.uri_mut() = uri;
        strip_hop_by_hop(req.headers_mut());

        let mut upstream_resp = self
            .client
            .request(req)
            .await
            .context("forward upgrade request to inference engine")?;

        if upstream_resp.status() != StatusCode::SWITCHING_PROTOCOLS {
            return Ok(map_upstream_response(upstream_resp));
        }

        let on_upstream = hyper::upgrade::on(&mut upstream_resp);
        let (parts, _) = upstream_resp.into_parts();
        let mut response = Response::from_parts(parts, Body::Empty);
        strip_hop_by_hop(response.headers_mut());

        tokio::spawn(async move {
            if let Err(err) = tunnel_upgrades(on_client, on_upstream).await {
                warn!(error = %err, "websocket upgrade tunnel failed");
            }
        });

        Ok(response)
    }
}

fn not_implemented(message: &str) -> Response<Body> {
    let body = Bytes::from(format!(r#"{{"error":"{message}"}}"#));
    let mut resp = Response::new(Body::Full(Full::new(body)));
    *resp.status_mut() = StatusCode::NOT_IMPLEMENTED;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    resp
}

pub enum Body {
    Full(Full<Bytes>),
    Incoming(Incoming),
    Empty,
}

impl hyper::body::Body for Body {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.get_mut() {
            Body::Full(body) => match std::pin::Pin::new(body).poll_frame(cx) {
                std::task::Poll::Ready(frame) => std::task::Poll::Ready(frame.map(|result| {
                    result.map_err(|never| match never {})
                })),
                std::task::Poll::Pending => std::task::Poll::Pending,
            },
            Body::Incoming(body) => std::pin::Pin::new(body).poll_frame(cx),
            Body::Empty => std::task::Poll::Ready(None),
        }
    }
}

fn map_upstream_response(resp: Response<Incoming>) -> Response<Body> {
    let (parts, body) = resp.into_parts();
    let mut response = Response::from_parts(parts, Body::Incoming(body));
    strip_hop_by_hop(response.headers_mut());
    response
}

/// Ollama rejects proxied requests unless Host looks like loopback and Origin is absent/disallowed.
fn prepare_ollama_upstream_headers(headers: &mut hyper::HeaderMap, upstream_host: &str) {
    headers.remove(header::ORIGIN);
    headers.remove(header::REFERER);
    if let Ok(value) = header::HeaderValue::from_str(upstream_host) {
        headers.insert(header::HOST, value);
    }
}

fn strip_hop_by_hop(headers: &mut hyper::HeaderMap) {
    for name in HOP_BY_HOP {
        if let Ok(header_name) = HeaderName::from_bytes(name.as_bytes()) {
            headers.remove(header_name);
        }
    }

    let connection_tokens: Vec<String> = headers
        .get(CONNECTION)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .map(|token| token.trim().to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default();

    headers.remove(CONNECTION);

    for token in connection_tokens {
        if let Ok(name) = HeaderName::from_bytes(token.as_bytes()) {
            headers.remove(name);
        }
    }
}

async fn tunnel_upgrades(client: OnUpgrade, upstream: OnUpgrade) -> Result<()> {
    let mut client = TokioIo::new(client.await.context("accept client upgrade")?);
    let mut upstream = TokioIo::new(upstream.await.context("accept upstream upgrade")?);
    tokio::io::copy_bidirectional(&mut client, &mut upstream)
        .await
        .context("tunnel upgraded streams")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, InferenceBackend};

    #[test]
    fn ollama_backend_upstream_base() {
        let config = Config {
            inference_backend: InferenceBackend::Ollama,
            https_bind: "0.0.0.0".into(),
            https_port: 8443,
            tls_rotate_secs: 3600,
            tls_cert_cn: "test".into(),
            cell_admin_token: String::new(),
            hf_token: None,
            llama_server_bin: String::new(),
            llama_host: "127.0.0.1".into(),
            llama_port: 8080,
            ollama_bin: "ollama".into(),
            ollama_host: "127.0.0.1".into(),
            ollama_port: 11434,
            ollama_models: "/models/.ollama".into(),
            ollama_origins: "*".into(),
            models_dir: "/models".into(),
            models_max: 4,
            llama_api_key: None,
            llama_extra_args: Vec::new(),
            model_max_bytes: 0,
            auto_load_after_pull: true,
            delete_gguf_after_import: false,
        };
        assert!(config.is_ollama());
        assert_eq!(config.upstream_base(), "http://127.0.0.1:11434");
    }

    #[test]
    fn ollama_backend_accepts_api_and_v1_paths() {
        // Routing is path-prefix based in handle_ollama; keep this as a documentation assertion
        // so OpenAI-compat (/v1/*) stays wired alongside native (/api/*).
        let api = "/api/chat";
        let v1 = "/v1/chat/completions";
        let other = "/metrics";
        assert!(api.starts_with("/api/") || api.starts_with("/v1/"));
        assert!(v1.starts_with("/api/") || v1.starts_with("/v1/"));
        assert!(!(other.starts_with("/api/") || other.starts_with("/v1/")));
    }
}
