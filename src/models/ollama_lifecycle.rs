use crate::config::Config;
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use std::path::Path;
use tracing::info;

pub fn gguf_to_ollama_tag(filename: &str) -> String {
    let stem = filename
        .strip_suffix(".gguf")
        .unwrap_or(filename)
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let name = if stem.is_empty() {
        "model".to_string()
    } else {
        stem
    };
    format!("{name}:latest")
}

pub async fn create_from_gguf(config: &Config, gguf_path: &Path) -> Result<String> {
    let filename = gguf_path
        .file_name()
        .and_then(|n| n.to_str())
        .context("gguf path has no filename")?;
    let tag = gguf_to_ollama_tag(filename);
    let abs = gguf_path
        .canonicalize()
        .unwrap_or_else(|_| gguf_path.to_path_buf());
    let modelfile = format!("FROM {}\n", abs.display());

    info!(tag = %tag, path = %abs.display(), "importing gguf into ollama");

    let client = reqwest::Client::new();
    let url = format!("{}/api/create", config.ollama_upstream_base());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "model": tag,
            "modelfile": modelfile,
        }))
        .send()
        .await
        .context("ollama create request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("ollama create failed: {status} {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut last_error = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read ollama create stream")?;
        if chunk.is_empty() {
            continue;
        }
        for line in chunk.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) {
                if value.get("status").and_then(|s| s.as_str()) == Some("success") {
                    if config.delete_gguf_after_import {
                        let _ = tokio::fs::remove_file(gguf_path).await;
                    }
                    return Ok(tag);
                }
                if let Some(err) = value.get("error").and_then(|e| e.as_str()) {
                    last_error = Some(err.to_string());
                }
            }
        }
    }

    if let Some(err) = last_error {
        bail!("ollama create failed: {err}");
    }

    if config.delete_gguf_after_import {
        let _ = tokio::fs::remove_file(gguf_path).await;
    }
    Ok(tag)
}

pub async fn load_model(config: &Config, model_ref: &str) -> Result<()> {
    let tag = resolve_ollama_tag(config, model_ref).await?;
    warmup_model(config, &tag, "30m").await
}

pub async fn unload_all(config: &Config) -> Result<()> {
    let running = fetch_running_models(config).await?;
    for tag in running {
        unload_model(config, &tag).await.ok();
    }
    Ok(())
}

pub async fn unload_model(config: &Config, tag: &str) -> Result<()> {
    warmup_model(config, tag, "0").await
}

pub async fn delete_model(config: &Config, model_ref: &str) -> Result<()> {
    let tag = resolve_ollama_tag(config, model_ref).await?;
    unload_model(config, &tag).await.ok();

    let client = reqwest::Client::new();
    let url = format!("{}/api/delete", config.ollama_upstream_base());
    let resp = client
        .delete(&url)
        .json(&serde_json::json!({ "model": tag }))
        .send()
        .await
        .context("ollama delete request")?;

    if resp.status().is_success() {
        maybe_delete_gguf(config, model_ref).await;
        return Ok(());
    }

    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "model": tag }))
        .send()
        .await
        .context("ollama delete fallback post")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("ollama delete failed: {status} {text}");
    }

    maybe_delete_gguf(config, model_ref).await;
    Ok(())
}

async fn resolve_ollama_tag(config: &Config, model_ref: &str) -> Result<String> {
    let trimmed = model_ref.trim();
    if trimmed.ends_with(".gguf") {
        let path = super::safe_join(&config.models_dir, trimmed)?;
        if path.exists() {
            return create_from_gguf(config, &path).await;
        }
        return Ok(gguf_to_ollama_tag(trimmed));
    }
    if trimmed.contains(':') {
        return Ok(trimmed.to_string());
    }
    Ok(format!("{trimmed}:latest"))
}

async fn warmup_model(config: &Config, tag: &str, keep_alive: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("build ollama warmup client")?;
    let url = format!("{}/api/generate", config.ollama_upstream_base());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "model": tag,
            "prompt": "",
            "stream": false,
            "keep_alive": keep_alive,
        }))
        .send()
        .await
        .context("ollama warmup request")?;

    if resp.status().is_success() {
        return Ok(());
    }

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.as_u16() == 400 && text.contains("embed") {
        let url = format!("{}/api/embed", config.ollama_upstream_base());
        let resp = client
            .post(&url)
            .json(&serde_json::json!({
                "model": tag,
                "input": ".",
                "keep_alive": keep_alive,
            }))
            .send()
            .await
            .context("ollama embed warmup request")?;
        if resp.status().is_success() {
            return Ok(());
        }
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("ollama embed warmup failed: {status} {text}");
    }

    bail!("ollama warmup failed: {status} {text}");
}

async fn fetch_running_models(config: &Config) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/ps", config.ollama_upstream_base());
    let json: serde_json::Value = client
        .get(&url)
        .send()
        .await
        .context("ollama ps request")?
        .json()
        .await
        .context("parse ollama ps response")?;

    let mut tags = Vec::new();
    if let Some(models) = json.get("models").and_then(|m| m.as_array()) {
        for model in models {
            if let Some(name) = model
                .get("name")
                .or_else(|| model.get("model"))
                .and_then(|v| v.as_str())
            {
                tags.push(name.to_string());
            }
        }
    }
    Ok(tags)
}

async fn maybe_delete_gguf(config: &Config, model_ref: &str) {
    if !model_ref.ends_with(".gguf") {
        return;
    }
    if let Ok(path) = super::safe_join(&config.models_dir, model_ref.trim()) {
        let _ = tokio::fs::remove_file(path).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_ollama_tag_from_gguf_filename() {
        assert_eq!(
            gguf_to_ollama_tag("Llama-3.2-3B-Q4_K_M.gguf"),
            "llama-3.2-3b-q4_k_m:latest"
        );
    }
}
