mod llama;
mod ollama;

pub use llama::{LlamaEngine, LlamaMode, spawn_child_reaper};
pub use ollama::OllamaEngine;

use crate::config::{Config, InferenceBackend};
use anyhow::Result;

pub enum EngineHandle {
    Llama(LlamaEngine),
    Ollama(OllamaEngine),
}

impl EngineHandle {
    pub fn from_config(config: Config) -> Self {
        match config.inference_backend {
            InferenceBackend::LlamaCpp => Self::Llama(LlamaEngine::new(config)),
            InferenceBackend::Ollama => Self::Ollama(OllamaEngine::new(config)),
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        match self {
            Self::Llama(engine) => engine.start().await,
            Self::Ollama(engine) => engine.start().await,
        }
    }

    pub fn stop(&mut self) {
        match self {
            Self::Llama(engine) => engine.stop(),
            Self::Ollama(engine) => engine.stop(),
        }
    }

    pub fn upstream_base(&self) -> String {
        match self {
            Self::Llama(engine) => engine.upstream_base(),
            Self::Ollama(engine) => engine.upstream_base(),
        }
    }

    pub fn llama(&mut self) -> Option<&mut LlamaEngine> {
        match self {
            Self::Llama(engine) => Some(engine),
            Self::Ollama(_) => None,
        }
    }
}
