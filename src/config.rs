use anyhow::{Context, Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceBackend {
    LlamaCpp,
    Ollama,
}

impl InferenceBackend {
    pub fn from_env_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "llamacpp" | "llama.cpp" | "llama_cpp" => Ok(Self::LlamaCpp),
            "ollama" => Ok(Self::Ollama),
            other => bail!("unsupported INFERENCE_BACKEND: {other}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llamacpp",
            Self::Ollama => "ollama",
        }
    }

    pub fn inference_api(self) -> &'static str {
        match self {
            Self::LlamaCpp => "openai",
            Self::Ollama => "ollama",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub inference_backend: InferenceBackend,
    pub https_bind: String,
    pub https_port: u16,
    pub tls_rotate_secs: u64,
    pub tls_cert_cn: String,
    pub cell_admin_token: String,
    pub hf_token: Option<String>,
    pub llama_server_bin: String,
    pub llama_host: String,
    pub llama_port: u16,
    pub ollama_bin: String,
    pub ollama_host: String,
    pub ollama_port: u16,
    pub ollama_models: String,
    pub ollama_origins: String,
    pub models_dir: String,
    pub models_max: u32,
    pub llama_api_key: Option<String>,
    pub llama_extra_args: Vec<String>,
    pub model_max_bytes: u64,
    pub auto_load_after_pull: bool,
    pub delete_gguf_after_import: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let cell_admin_token = std::env::var("CELL_ADMIN_TOKEN").unwrap_or_default();

        let llama_extra_args = std::env::var("LLAMA_EXTRA_ARGS")
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect();

        let inference_backend = std::env::var("INFERENCE_BACKEND")
            .map(|v| InferenceBackend::from_env_str(&v))
            .unwrap_or(Ok(InferenceBackend::LlamaCpp))?;

        Ok(Self {
            inference_backend,
            https_bind: std::env::var("HTTPS_BIND").unwrap_or_else(|_| "0.0.0.0".into()),
            https_port: parse_env("HTTPS_PORT", 8443)?,
            tls_rotate_secs: parse_env("TLS_ROTATE_SECS", 3600)?,
            tls_cert_cn: std::env::var("TLS_CERT_CN")
                .unwrap_or_else(|_| "llama-inference.local".into()),
            cell_admin_token,
            hf_token: non_empty_env("HF_TOKEN"),
            llama_server_bin: std::env::var("LLAMA_SERVER_BIN")
                .unwrap_or_else(|_| "/app/llama-server".into()),
            llama_host: std::env::var("LLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            llama_port: parse_env("LLAMA_PORT", 8080)?,
            ollama_bin: std::env::var("OLLAMA_BIN").unwrap_or_else(|_| "ollama".into()),
            ollama_host: std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            ollama_port: parse_env("OLLAMA_PORT", 11434)?,
            ollama_models: std::env::var("OLLAMA_MODELS")
                .unwrap_or_else(|_| "/models/.ollama".into()),
            ollama_origins: std::env::var("OLLAMA_ORIGINS").unwrap_or_else(|_| "*".into()),
            models_dir: std::env::var("MODELS_DIR").unwrap_or_else(|_| "/models".into()),
            models_max: parse_env("MODELS_MAX", 4)?,
            llama_api_key: non_empty_env("LLAMA_API_KEY"),
            llama_extra_args,
            model_max_bytes: parse_env("MODEL_MAX_BYTES", 80_u64 * 1024 * 1024 * 1024)?,
            auto_load_after_pull: parse_bool_env("AUTO_LOAD_AFTER_PULL", true),
            delete_gguf_after_import: parse_bool_env("DELETE_GGUF_AFTER_IMPORT", false),
        })
    }

    pub fn is_ollama(&self) -> bool {
        self.inference_backend == InferenceBackend::Ollama
    }

    pub fn is_llamacpp(&self) -> bool {
        self.inference_backend == InferenceBackend::LlamaCpp
    }

    pub fn upstream_base(&self) -> String {
        match self.inference_backend {
            InferenceBackend::LlamaCpp => self.llama_upstream_base(),
            InferenceBackend::Ollama => self.ollama_upstream_base(),
        }
    }

    pub fn health_url(&self) -> String {
        match self.inference_backend {
            InferenceBackend::LlamaCpp => format!("{}/health", self.llama_upstream_base()),
            InferenceBackend::Ollama => format!("{}/api/tags", self.ollama_upstream_base()),
        }
    }

    pub fn llama_upstream_base(&self) -> String {
        format!("http://{}:{}", self.llama_host, self.llama_port)
    }

    pub fn ollama_upstream_base(&self) -> String {
        format!("http://{}:{}", self.ollama_host, self.ollama_port)
    }

    /// Host header Ollama expects on loopback-proxied requests (avoids 403 from host verification).
    pub fn ollama_upstream_host_header(&self) -> String {
        format!("{}:{}", self.ollama_host, self.ollama_port)
    }

    pub fn llama_health_url(&self) -> String {
        format!("{}/health", self.llama_upstream_base())
    }
}

fn parse_env<T>(key: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(key) {
        Ok(raw) if !raw.is_empty() => raw
            .parse()
            .with_context(|| format!("invalid value for {key}")),
        _ => Ok(default),
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn parse_bool_env(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(raw) if raw.eq_ignore_ascii_case("true") || raw == "1" => true,
        Ok(raw) if raw.eq_ignore_ascii_case("false") || raw == "0" => false,
        Ok(_) => default,
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_inference_backend_values() {
        assert_eq!(
            InferenceBackend::from_env_str("llamacpp").unwrap(),
            InferenceBackend::LlamaCpp
        );
        assert_eq!(
            InferenceBackend::from_env_str("ollama").unwrap(),
            InferenceBackend::Ollama
        );
        assert!(InferenceBackend::from_env_str("unknown").is_err());
    }
}
