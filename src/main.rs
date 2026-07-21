mod config;
mod engine;
mod llamacpp_status;
mod models;
mod privilege;
mod proxy;
mod tls;

use anyhow::{Context, Result};
use config::Config;
use engine::{spawn_child_reaper, EngineHandle};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Empty, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use proxy::ProxyService;
use std::net::SocketAddr;
use std::sync::Arc;
use tls::RotatingTls;
use tokio::sync::Mutex;
use tokio_rustls::TlsAcceptor;
use tracing::{error, info, warn};

type BoxResponse = Response<BoxBody<Bytes, hyper::Error>>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tls::init_crypto_provider()?;

    let config = Config::from_env()?;
    privilege::prepare_models_volume(&config.models_dir)?;
    info!(
        https_port = config.https_port,
        tls_rotate_secs = config.tls_rotate_secs,
        models_dir = %config.models_dir,
        inference_backend = config.inference_backend.as_str(),
        pull_auth = !config.cell_admin_token.is_empty(),
        hf_token_configured = config.hf_token.is_some(),
        hf_download_backend = "hf-hub-v1",
        "starting mtrxai-icell"
    );

    // Auth: CELL_ADMIN_TOKEN uses constant-time bearer verify (mtrxai-auth).
    // Empty token leaves /mtrxai/v1/models/* open (legacy/dev). Fail closed with
    // MTRXAI_ICELL_REQUIRE_ADMIN=1. INFERENCE_DEV=1 softens the warning only.
    if config.cell_admin_token.is_empty() {
        if env_truthy("MTRXAI_ICELL_REQUIRE_ADMIN") {
            anyhow::bail!("CELL_ADMIN_TOKEN is required when MTRXAI_ICELL_REQUIRE_ADMIN=1");
        }
        if env_truthy("INFERENCE_DEV") {
            warn!("CELL_ADMIN_TOKEN is empty (INFERENCE_DEV): admin model routes are open");
        } else {
            warn!(
                "CELL_ADMIN_TOKEN is empty: /mtrxai/v1/models/* routes are unauthenticated \
                 (set a token, or MTRXAI_ICELL_REQUIRE_ADMIN=1 to refuse boot)"
            );
        }
    }
    if config.hf_token.is_none() {
        warn!("HF_TOKEN is not set: gated Hugging Face models will fail with 403");
    }

    spawn_child_reaper();

    let mut engine = EngineHandle::from_config(config.clone());
    engine.start().await?;

    let engine = Arc::new(Mutex::new(engine));

    let tls = Arc::new(RotatingTls::new(
        &config.tls_cert_cn,
        config.tls_rotate_secs,
    )?);

    let proxy = Arc::new(ProxyService::new(config.clone(), Arc::clone(&engine)));
    let addr: SocketAddr = format!("{}:{}", config.https_bind, config.https_port)
        .parse()
        .context("parse HTTPS listen address")?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind HTTPS on {addr}"))?;
    info!(%addr, "listening for TLS connections");

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, peer) = accept.context("accept TLS connection")?;
                let acceptor = TlsAcceptor::from(tls.server_config());
                let proxy = Arc::clone(&proxy);
                tokio::spawn(async move {
                    if let Err(err) = serve_connection(acceptor, stream, proxy).await {
                        tracing::warn!(%peer, error = %err, "connection closed");
                    }
                });
            }
            _ = &mut shutdown => {
                info!("shutdown signal received");
                break;
            }
        }
    }

    engine.lock().await.stop();
    Ok(())
}

async fn serve_connection(
    acceptor: TlsAcceptor,
    stream: tokio::net::TcpStream,
    proxy: Arc<ProxyService>,
) -> Result<()> {
    let tls_stream = acceptor.accept(stream).await.context("TLS handshake")?;
    let io = TokioIo::new(tls_stream);

    let service = service_fn(move |req: Request<Incoming>| {
        let proxy = Arc::clone(&proxy);
        async move { handle_request(proxy, req).await }
    });

    if let Err(err) =
        hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
            .http1_only()
            .serve_connection(io, service)
            .await
    {
        anyhow::bail!("serve HTTP connection: {err}");
    }

    Ok(())
}

async fn handle_request(
    proxy: Arc<ProxyService>,
    req: Request<Incoming>,
) -> Result<BoxResponse, hyper::Error> {
    match proxy.handle(req).await {
        Ok(resp) => Ok(box_response(resp)),
        Err(err) => {
            error!(error = %err, "request failed");
            Ok(internal_error())
        }
    }
}

fn box_response(resp: Response<proxy::Body>) -> BoxResponse {
    let (parts, body) = resp.into_parts();
    let mapped = match body {
        proxy::Body::Full(full) => full.map_err(|never| match never {}).boxed(),
        proxy::Body::Incoming(incoming) => incoming.boxed(),
        proxy::Body::Empty => Empty::<Bytes>::new()
            .map_err(|never| match never {})
            .boxed(),
    };
    Response::from_parts(parts, mapped)
}

fn internal_error() -> BoxResponse {
    Response::builder()
        .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
        .body(
            Full::from("internal server error")
                .map_err(|never| match never {})
                .boxed(),
        )
        .expect("valid error response")
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("register ctrl-c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("register SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
