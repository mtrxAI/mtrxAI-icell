use crate::config::Config;
use anyhow::{Context, Result, bail};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tracing::info;

pub struct OllamaEngine {
    child: Option<Child>,
    config: Config,
}

impl OllamaEngine {
    pub fn new(config: Config) -> Self {
        Self {
            child: None,
            config,
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        if let Err(err) = std::fs::create_dir_all(&self.config.ollama_models) {
            tracing::warn!(
                error = %err,
                dir = %self.config.ollama_models,
                "could not create ollama models dir"
            );
        }

        let mut cmd = Command::new(&self.config.ollama_bin);
        cmd.arg("serve");
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        cmd.env("HOME", "/tmp");
        cmd.env("OLLAMA_HOST", format!("{}:{}", self.config.ollama_host, self.config.ollama_port));
        cmd.env("OLLAMA_MODELS", &self.config.ollama_models);
        cmd.env("OLLAMA_ORIGINS", &self.config.ollama_origins);

        info!(
            bin = %self.config.ollama_bin,
            host = %self.config.ollama_host,
            port = self.config.ollama_port,
            models = %self.config.ollama_models,
            "starting ollama serve"
        );

        let child = cmd.spawn().context("spawn ollama serve")?;
        self.child = Some(child);

        self.wait_for_health().await?;
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            info!("stopping ollama serve");
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    pub fn upstream_base(&self) -> String {
        self.config.ollama_upstream_base()
    }

    async fn wait_for_health(&mut self) -> Result<()> {
        let url = self.config.health_url();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .context("build ollama health check client")?;

        for attempt in 1..=120 {
            if self
                .child
                .as_mut()
                .and_then(|child| child.try_wait().ok())
                .flatten()
                .is_some()
            {
                bail!("ollama serve exited during startup");
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!(url = %url, "ollama serve is healthy");
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::debug!(status = %resp.status(), attempt, "waiting for ollama health");
                }
                Err(err) => {
                    tracing::debug!(error = %err, attempt, "waiting for ollama health");
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        bail!("ollama serve health check timed out");
    }
}

impl Drop for OllamaEngine {
    fn drop(&mut self) {
        self.stop();
    }
}
