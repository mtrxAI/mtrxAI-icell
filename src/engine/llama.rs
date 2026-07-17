use crate::config::Config;
use anyhow::{Context, Result, bail};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
#[cfg(unix)]
use tracing::warn;
use tracing::info;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlamaMode {
    Router,
    Model { path: String },
}

pub struct LlamaEngine {
    child: Option<Child>,
    config: Config,
    mode: LlamaMode,
}

impl LlamaEngine {
    pub fn new(config: Config) -> Self {
        Self {
            child: None,
            config,
            mode: LlamaMode::Router,
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        self.start_mode(self.mode.clone()).await
    }

    pub async fn restart_mode(&mut self, mode: LlamaMode) -> Result<()> {
        self.stop();
        self.mode = mode.clone();
        self.start_mode(mode).await
    }

    pub fn mode(&self) -> LlamaMode {
        self.mode.clone()
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            info!("stopping llama-server");
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    pub fn upstream_base(&self) -> String {
        self.config.llama_upstream_base()
    }

    async fn start_mode(&mut self, mode: LlamaMode) -> Result<()> {
        if let Err(err) = std::fs::create_dir_all(&self.config.models_dir) {
            tracing::warn!(
                error = %err,
                dir = %self.config.models_dir,
                "could not create models dir; using existing mount"
            );
        }

        let mut cmd = Command::new(&self.config.llama_server_bin);
        cmd.arg("--host")
            .arg(&self.config.llama_host)
            .arg("--port")
            .arg(self.config.llama_port.to_string());

        match &mode {
            LlamaMode::Router => {
                cmd.arg("--models-dir")
                    .arg(&self.config.models_dir)
                    .arg("--models-autoload")
                    .arg("--models-max")
                    .arg(self.config.models_max.to_string());
            }
            LlamaMode::Model { path } => {
                cmd.arg("--model").arg(path);
            }
        }

        if let Some(api_key) = &self.config.llama_api_key {
            cmd.arg("--api-key").arg(api_key);
        }

        for arg in &self.config.llama_extra_args {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        cmd.env("HOME", "/tmp");
        cmd.env("HF_HOME", "/tmp/huggingface");
        cmd.env("XDG_CACHE_HOME", "/tmp/.cache");

        info!(
            bin = %self.config.llama_server_bin,
            host = %self.config.llama_host,
            port = self.config.llama_port,
            models_dir = %self.config.models_dir,
            mode = ?mode,
            "starting llama-server"
        );

        let child = cmd.spawn().context("spawn llama-server")?;
        self.child = Some(child);

        self.wait_for_health().await?;
        Ok(())
    }

    async fn wait_for_health(&mut self) -> Result<()> {
        let url = self.config.llama_health_url();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .context("build health check client")?;

        for attempt in 1..=120 {
            if self
                .child
                .as_mut()
                .and_then(|child| child.try_wait().ok())
                .flatten()
                .is_some()
            {
                bail!("llama-server exited during startup");
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!(url = %url, "llama-server is healthy");
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::debug!(status = %resp.status(), attempt, "waiting for llama-server health");
                }
                Err(err) => {
                    tracing::debug!(error = %err, attempt, "waiting for llama-server health");
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        bail!("llama-server health check timed out");
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(unix)]
pub fn spawn_child_reaper() {
    tokio::spawn(async {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigchld = match signal(SignalKind::child()) {
            Ok(sig) => sig,
            Err(err) => {
                warn!(error = %err, "failed to register SIGCHLD handler");
                return;
            }
        };
        loop {
            sigchld.recv().await;
            loop {
                let mut status = 0;
                let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
                if pid <= 0 {
                    break;
                }
                warn!(pid, status, "reaped child process");
            }
        }
    });
}

#[cfg(not(unix))]
pub fn spawn_child_reaper() {}
