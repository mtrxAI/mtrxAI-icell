use crate::config::Config;
use crate::engine::{EngineHandle, LlamaMode};
use crate::llamacpp_status::{
    fetch_router_models, parse_router_model_states, resolve_router_model_id,
    router_load_error_is_already_running, router_status_is_loaded,
};
use anyhow::{bail, Context, Result};
use std::path::Path;
use tracing::warn;

pub async fn load_model(config: &Config, engine: &mut EngineHandle, model_ref: &str) -> Result<()> {
    if config.is_llamacpp() {
        load_llamacpp(config, engine, model_ref).await
    } else {
        super::ollama_lifecycle::load_model(config, model_ref).await
    }
}

pub async fn unload_model(config: &Config, engine: &mut EngineHandle) -> Result<()> {
    if config.is_llamacpp() {
        unload_llamacpp(engine).await
    } else {
        super::ollama_lifecycle::unload_all(config).await
    }
}

pub async fn delete_model(
    config: &Config,
    engine: &mut EngineHandle,
    model_ref: &str,
) -> Result<()> {
    if config.is_llamacpp() {
        delete_llamacpp(config, engine, model_ref).await
    } else {
        super::ollama_lifecycle::delete_model(config, model_ref).await
    }
}

pub async fn register_after_pull(config: &Config, gguf_path: &Path) -> Result<String> {
    if config.is_llamacpp() {
        Ok(super::model_id_from_path(gguf_path, &config.models_dir))
    } else {
        super::ollama_lifecycle::create_from_gguf(config, gguf_path).await
    }
}

pub async fn auto_load_after_pull(
    config: &Config,
    engine: &mut EngineHandle,
    model_id: &str,
) -> Result<()> {
    load_model(config, engine, model_id).await
}

async fn load_llamacpp(config: &Config, engine: &mut EngineHandle, model_ref: &str) -> Result<()> {
    let filename = normalize_gguf_filename(model_ref)?;
    super::validate_filename(&filename)?;
    let model_path = super::safe_join(&config.models_dir, &filename)?;
    if !model_path.exists() {
        bail!("model file not found: {}", model_path.display());
    }

    let use_router = engine
        .llama()
        .is_some_and(|llama| llama.mode() == LlamaMode::Router);
    if use_router {
        match llamacpp_router_load(config, &filename).await {
            Ok(()) => return Ok(()),
            Err(err) => {
                warn!(
                    error = %err,
                    model = %filename,
                    "router load failed; falling back to dedicated llama-server model mode"
                );
            }
        }
    }

    let llama = engine.llama().context("llama engine not active")?;
    llama
        .restart_mode(LlamaMode::Model {
            path: model_path.display().to_string(),
        })
        .await
        .context("restart llama-server with model")?;
    Ok(())
}

async fn unload_llamacpp(engine: &mut EngineHandle) -> Result<()> {
    let llama = engine.llama().context("llama engine not active")?;
    llama
        .restart_mode(LlamaMode::Router)
        .await
        .context("restart llama-server in router mode")?;
    Ok(())
}

async fn delete_llamacpp(
    config: &Config,
    engine: &mut EngineHandle,
    model_ref: &str,
) -> Result<()> {
    use std::path::PathBuf;
    use tokio::fs;

    let filename = normalize_gguf_filename(model_ref)?;
    super::validate_filename(&filename)?;
    let path = super::safe_join(&config.models_dir, &filename)?;

    if let Some(llama) = engine.llama() {
        if let LlamaMode::Model { path: loaded } = llama.mode() {
            if PathBuf::from(&loaded) == path {
                llama.restart_mode(LlamaMode::Router).await.ok();
            }
        }
    }

    fs::remove_file(&path)
        .await
        .with_context(|| format!("delete {}", path.display()))?;
    Ok(())
}

pub async fn llamacpp_router_load(config: &Config, model_ref: &str) -> Result<()> {
    let upstream = config.llama_upstream_base();
    let payload = fetch_router_models(&upstream, true)
        .await
        .context("reload llama-server router model catalog")?;

    let router_id = resolve_router_model_id(&payload, model_ref).with_context(|| {
        format!("model '{model_ref}' not found in llama-server router catalog after reload")
    })?;

    let states = parse_router_model_states(&payload);
    if states
        .get(&router_id)
        .is_some_and(|status| router_status_is_loaded(status))
    {
        return Ok(());
    }

    let url = format!("{upstream}/models/load");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("build llama-server load client")?;
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "model": router_id }))
        .send()
        .await
        .context("call llama-server /models/load")?;

    if resp.status().is_success() {
        return Ok(());
    }

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if router_load_error_is_already_running(status, &text) {
        return Ok(());
    }

    bail!("llama-server /models/load failed: {status} {text}");
}

fn normalize_gguf_filename(model_ref: &str) -> Result<String> {
    let trimmed = model_ref.trim();
    if trimmed.ends_with(".gguf") {
        super::validate_filename(trimmed)?;
        return Ok(trimmed.to_string());
    }
    bail!("llama.cpp load/delete requires a .gguf filename");
}
