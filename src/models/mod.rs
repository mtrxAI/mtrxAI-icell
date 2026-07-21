mod lifecycle;
mod ollama_lifecycle;

#[cfg(test)]
mod tests;

use crate::config::Config;
use crate::engine::EngineHandle;
use anyhow::{Context, Result, bail};
use dashmap::DashMap;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::{Method, Request, Response, StatusCode, header};
use mtrxai_auth::verify_bearer_token;
use mtrxai_icell_api::{
    JobStatus, MODELS_DELETE_PATH, MODELS_LOAD_PATH, MODELS_PULL_PATH, MODELS_UNLOAD_PATH,
    PullJobRecord, PullRequest,
};
use reqwest::header as reqwest_header;
use serde::Deserialize;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::sync::Mutex;
use hf_hub::{split_id, HFClient};
use tracing::{info, warn};
use uuid::Uuid;

const HF_USER_AGENT: &str = "huggingface_hub/0.26.0; rust-reqwest; mtrxai-icell";

#[derive(Clone)]
pub struct ModelsService {
    config: Config,
    jobs: Arc<DashMap<String, PullJobRecord>>,
    engine: Arc<Mutex<EngineHandle>>,
}

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(rename = "rfilename")]
    rfilename: String,
}

struct ResolvedHfFile {
    rfilename: String,
}

impl ModelsService {
    pub fn new(config: Config, engine: Arc<Mutex<EngineHandle>>) -> Self {
        Self {
            config,
            jobs: Arc::new(DashMap::new()),
            engine,
        }
    }

    pub async fn handle(
        &self,
        method: &Method,
        path: &str,
        req: Request<Incoming>,
    ) -> Result<Response<Full<Bytes>>> {
        if path == MODELS_PULL_PATH && *method == Method::POST {
            return self.post_pull(req).await;
        }

        if let Some(job_id) = path.strip_prefix(&format!("{MODELS_PULL_PATH}/")) {
            if *method == Method::GET && !job_id.is_empty() {
                return self.get_pull_job(job_id, req).await;
            }
        }

        if path == MODELS_LOAD_PATH && *method == Method::POST {
            return self.post_load(req).await;
        }

        if path == MODELS_UNLOAD_PATH && *method == Method::POST {
            return self.post_unload(req).await;
        }

        if path == MODELS_DELETE_PATH && *method == Method::POST {
            return self.post_delete(req).await;
        }

        json_response(
            StatusCode::NOT_FOUND,
            serde_json::json!({ "error": "not found" }),
        )
    }

    async fn post_pull(&self, req: Request<Incoming>) -> Result<Response<Full<Bytes>>> {
        if !authorized(&req, &self.config.cell_admin_token) {
            return json_response(
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            );
        }

        let body = req
            .into_body()
            .collect()
            .await
            .context("read pull request body")?
            .to_bytes();
        let pull: PullRequest =
            serde_json::from_slice(&body).context("parse pull request json")?;

        validate_repo(&pull.repo)?;

        let job_id = Uuid::new_v4().to_string();
        let record = PullJobRecord {
            id: job_id.clone(),
            status: JobStatus::Queued,
            repo: pull.repo.clone(),
            quant: pull.quant.clone(),
            path: None,
            model_id: None,
            error: None,
            started_at: unix_now(),
            finished_at: None,
        };
        self.jobs.insert(job_id.clone(), record.clone());

        let service = self.clone();
        tokio::spawn(async move {
            service.run_pull_job(job_id, pull).await;
        });

        json_response(
            StatusCode::ACCEPTED,
            serde_json::json!({
                "status": "queued",
                "job_id": record.id,
            }),
        )
    }

    async fn get_pull_job(&self, job_id: &str, req: Request<Incoming>) -> Result<Response<Full<Bytes>>> {
        if !authorized(&req, &self.config.cell_admin_token) {
            return json_response(
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            );
        }

        let Some(record) = self.jobs.get(job_id) else {
            return json_response(
                StatusCode::NOT_FOUND,
                serde_json::json!({ "error": "job not found" }),
            );
        };
        json_response(StatusCode::OK, serde_json::to_value(record.value())?)
    }

    async fn post_load(&self, req: Request<Incoming>) -> Result<Response<Full<Bytes>>> {
        if !authorized(&req, &self.config.cell_admin_token) {
            return json_response(
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            );
        }

        #[derive(Deserialize)]
        struct LoadBody {
            model: String,
        }

        let body = req
            .into_body()
            .collect()
            .await
            .context("read load request body")?
            .to_bytes();
        let parsed: LoadBody = serde_json::from_slice(&body).context("parse load request json")?;

        let mut engine = self.engine.lock().await;
        lifecycle::load_model(&self.config, &mut engine, parsed.model.trim())
            .await
            .context("load model")?;

        json_response(StatusCode::OK, serde_json::json!({ "status": "ok" }))
    }

    async fn post_unload(&self, req: Request<Incoming>) -> Result<Response<Full<Bytes>>> {
        if !authorized(&req, &self.config.cell_admin_token) {
            return json_response(
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            );
        }

        let mut engine = self.engine.lock().await;
        lifecycle::unload_model(&self.config, &mut engine)
            .await
            .context("unload model")?;

        json_response(StatusCode::OK, serde_json::json!({ "status": "ok" }))
    }

    async fn post_delete(&self, req: Request<Incoming>) -> Result<Response<Full<Bytes>>> {
        if !authorized(&req, &self.config.cell_admin_token) {
            return json_response(
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            );
        }

        #[derive(Deserialize)]
        struct DeleteBody {
            filename: String,
        }

        let body = req
            .into_body()
            .collect()
            .await
            .context("read delete request body")?
            .to_bytes();
        let parsed: DeleteBody =
            serde_json::from_slice(&body).context("parse delete request json")?;

        let mut engine = self.engine.lock().await;
        lifecycle::delete_model(&self.config, &mut engine, parsed.filename.trim())
            .await
            .context("delete model")?;

        json_response(StatusCode::OK, serde_json::json!({ "status": "ok" }))
    }

    async fn run_pull_job(&self, job_id: String, pull: PullRequest) {
        self.update_job(&job_id, |job| {
            job.status = JobStatus::Running;
        });

        match self.execute_pull(&job_id, &pull).await {
            Ok((path, model_id)) => {
                info!(job_id = %job_id, path = %path.display(), model_id = %model_id, "model pull completed");

                if self.config.auto_load_after_pull {
                    let mut engine = self.engine.lock().await;
                    if let Err(err) =
                        lifecycle::auto_load_after_pull(&self.config, &mut engine, &model_id).await
                    {
                        warn!(error = %err, model_id = %model_id, "auto-load after pull failed");
                    }
                } else if self.config.is_llamacpp() {
                    if let Err(err) = lifecycle::llamacpp_router_load(&self.config, &model_id).await {
                        warn!(error = %err, model_id = %model_id, "auto-load after pull failed");
                    }
                }

                self.update_job(&job_id, |job| {
                    job.status = JobStatus::Ok;
                    job.path = Some(path.display().to_string());
                    job.model_id = Some(model_id);
                    job.finished_at = Some(unix_now());
                });
            }
            Err(err) => {
                let detail = format!("{err:#}");
                warn!(job_id = %job_id, error = %detail, "model pull failed");
                self.update_job(&job_id, |job| {
                    job.status = JobStatus::Failed;
                    job.error = Some(detail);
                    job.finished_at = Some(unix_now());
                });
            }
        }
    }

    async fn execute_pull(
        &self,
        job_id: &str,
        pull: &PullRequest,
    ) -> Result<(PathBuf, String)> {
        let files = resolve_hf_gguf_files(&self.config, &pull.repo, pull.quant.as_deref()).await?;
        if files.is_empty() {
            bail!("no .gguf files found for repo {}", pull.repo);
        }

        let single = files.len() == 1;
        let mut written_paths = Vec::new();
        for hf_file in files {
            let target_name = if single {
                pull.filename
                    .clone()
                    .unwrap_or_else(|| hf_file.rfilename.clone())
            } else {
                hf_file.rfilename.clone()
            };
            let path = self
                .download_gguf(&pull.repo, &hf_file.rfilename, &target_name)
                .await?;
            written_paths.push(path);
        }

        let primary = written_paths
            .first()
            .cloned()
            .context("missing downloaded path")?;

        if self.config.is_ollama() {
            self.update_job(job_id, |job| {
                job.status = JobStatus::Importing;
            });
        }

        let model_id = lifecycle::register_after_pull(&self.config, &primary).await?;
        Ok((primary, model_id))
    }

    async fn download_gguf(
        &self,
        repo: &str,
        remote_name: &str,
        target_name: &str,
    ) -> Result<PathBuf> {
        validate_filename(target_name)?;

        let dest = safe_join(&self.config.models_dir, target_name)?;
        if dest.exists() {
            bail!("target already exists: {}", dest.display());
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).await.context("create model parent dir")?;
        }

        let tmp_dir = safe_join(&self.config.models_dir, ".tmp")?;
        fs::create_dir_all(&tmp_dir).await.ok();

        let staged_path = download_hf_hub_file(
            repo,
            remote_name,
            &tmp_dir,
            self.config.hf_token.as_deref(),
            self.config.model_max_bytes,
        )
        .await
        .with_context(|| format!("download hf model file {repo}/{remote_name}"))?;

        fs::rename(&staged_path, &dest)
            .await
            .with_context(|| format!("rename {} -> {}", staged_path.display(), dest.display()))?;

        Ok(dest)
    }

    fn update_job(&self, job_id: &str, update: impl FnOnce(&mut PullJobRecord)) {
        if let Some(mut entry) = self.jobs.get_mut(job_id) {
            update(&mut entry);
        }
    }
}

pub(crate) fn model_id_from_path(path: &Path, models_dir: &str) -> String {
    path.strip_prefix(models_dir)
        .ok()
        .and_then(|rel| rel.to_str())
        .map(|s| s.trim_start_matches('/').replace('\\', "/"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string()
        })
}

async fn resolve_hf_gguf_files(
    config: &Config,
    repo: &str,
    quant: Option<&str>,
) -> Result<Vec<ResolvedHfFile>> {
    validate_repo(repo)?;

    let url = format!("https://huggingface.co/api/models/{repo}");
    let client = build_hf_http_client()?;
    let resp = hf_get(&client, &url, config.hf_token.as_deref())
        .await
        .context("query hugging face model metadata")?;

    let info: HfModelInfo = resp
        .error_for_status()
        .context("hugging face metadata request failed")?
        .json()
        .await
        .context("parse hugging face metadata")?;

    let siblings = info.siblings;
    let ggufs: Vec<String> = siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| name.ends_with(".gguf"))
        .collect();

    if ggufs.is_empty() {
        return Ok(Vec::new());
    }

    let mut selected = select_gguf_files_for_quant(&ggufs, quant);

    if selected.is_empty() {
        bail!("no gguf matching requested quant");
    }

    let mmproj: Vec<String> = siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| {
            name.ends_with(".gguf") && (name.starts_with("mmproj") || name.contains("mmproj"))
        })
        .collect();

    let mut resolved = Vec::new();
    for name in selected.drain(..) {
        resolved.push(ResolvedHfFile { rfilename: name });
    }
    for name in mmproj {
        if !resolved.iter().any(|f| f.rfilename == name) {
            resolved.push(ResolvedHfFile { rfilename: name });
        }
    }

    Ok(resolved)
}

fn authorized(req: &Request<Incoming>, expected: &str) -> bool {
    // Empty token keeps pull/load routes open (dev). Prefer CELL_ADMIN_TOKEN in prod;
    // set MTRXAI_ICELL_REQUIRE_ADMIN=1 to refuse boot without a token (see main.rs).
    if expected.is_empty() {
        return true;
    }
    let header_value = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    verify_bearer_token(header_value, expected)
}

fn build_hf_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(HF_USER_AGENT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("build hf http client")
}

pub(crate) fn hf_resolve_download_url(repo: &str, revision: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{repo}/resolve/{revision}/{}?download=true",
        encode_hf_path(filename)
    )
}

pub(crate) fn encode_hf_path(path: &str) -> String {
    path.split('/')
        .map(encode_hf_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_hf_path_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

pub(crate) fn select_gguf_files_for_quant(ggufs: &[String], quant: Option<&str>) -> Vec<String> {
    let matching: Vec<String> = if let Some(quant) = quant {
        let quant_lower = quant.to_ascii_lowercase();
        ggufs
            .iter()
            .filter(|name| name.to_ascii_lowercase().contains(&quant_lower))
            .cloned()
            .collect()
    } else {
        ggufs
            .iter()
            .find(|name| name.to_ascii_uppercase().contains("Q4_K_M"))
            .cloned()
            .or_else(|| ggufs.first().cloned())
            .into_iter()
            .collect()
    };

    if matching.is_empty() {
        return Vec::new();
    }

    let singles: Vec<String> = matching
        .iter()
        .filter(|name| !is_split_gguf_shard(name))
        .cloned()
        .collect();
    if !singles.is_empty() {
        return singles;
    }

    let shards: Vec<String> = matching
        .into_iter()
        .filter(|name| is_split_gguf_shard(name))
        .collect();
    if shards.is_empty() {
        return Vec::new();
    }

    let Some(group_key) = split_gguf_group_key(shards.first().unwrap()) else {
        return shards;
    };

    let mut grouped: Vec<String> = ggufs
        .iter()
        .filter(|name| split_gguf_group_key(name).as_deref() == Some(group_key.as_str()))
        .cloned()
        .collect();
    grouped.sort();
    grouped
}

fn is_split_gguf_shard(name: &str) -> bool {
    name.contains("-of-") && name.ends_with(".gguf")
}

fn split_gguf_group_key(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf")?;
    let (prefix, of_suffix) = stem.rsplit_once("-of-")?;
    if !of_suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (group, shard) = prefix.rsplit_once('-')?;
    if !shard.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(group.to_string())
}

async fn download_hf_hub_file(
    repo: &str,
    remote_name: &str,
    staging_dir: &Path,
    hf_token: Option<&str>,
    max_bytes: u64,
) -> Result<PathBuf> {
    let mut builder = HFClient::builder().cache_enabled(false);
    if let Some(token) = normalized_hf_token(hf_token) {
        builder = builder.token(token);
    }
    let client = builder
        .build()
        .map_err(|err| anyhow::anyhow!("build hf-hub client: {err}"))?;

    let (owner, name) = split_id(repo);
    let path = client
        .model(owner, name)
        .download_file()
        .filename(remote_name)
        .local_dir(staging_dir.to_path_buf())
        .send()
        .await
        .map_err(|err| anyhow::anyhow!("hf-hub download {repo}/{remote_name}: {err}"))?;

    let metadata = fs::metadata(&path)
        .await
        .with_context(|| format!("stat downloaded hf file {}", path.display()))?;
    if metadata.len() > max_bytes {
        let _ = fs::remove_file(&path).await;
        bail!("model exceeds MODEL_MAX_BYTES limit");
    }

    Ok(path)
}

fn normalized_hf_token(token: Option<&str>) -> Option<String> {
    token
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn hf_hub_host(url: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
}

pub(crate) fn hf_should_attach_auth(url: &str, token: Option<&str>) -> Option<String> {
    let token = normalized_hf_token(token)?;
    match hf_hub_host(url)?.as_str() {
        "huggingface.co" | "www.huggingface.co" => Some(token),
        _ => None,
    }
}

async fn hf_get(
    client: &reqwest::Client,
    url: &str,
    hf_token: Option<&str>,
) -> Result<reqwest::Response> {
    hf_get_follow(client, url, hf_token).await
}

async fn hf_get_follow(
    client: &reqwest::Client,
    url: &str,
    hf_token: Option<&str>,
) -> Result<reqwest::Response> {
    let mut current = url.to_string();
    let mut auth = normalized_hf_token(hf_token);

    for _ in 0..16 {
        let attach_auth = hf_should_attach_auth(&current, auth.as_deref());
        let mut req = client.get(&current);
        if let Some(token) = attach_auth.as_ref() {
            req = req.header(reqwest_header::AUTHORIZATION, format!("Bearer {token}"));
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("hf GET {current} (auth={})", attach_auth.is_some()))?;
        if resp.status().is_redirection() {
            let location = resp
                .headers()
                .get(reqwest_header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
                .with_context(|| format!("hf redirect missing Location header for {current}"))?;
            current = resolve_hf_redirect(&current, &location)?;
            auth = None;
            continue;
        }
        return Ok(resp);
    }

    bail!("hf request exceeded redirect limit for {url}");
}

fn resolve_hf_redirect(current: &str, location: &str) -> Result<String> {
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }
    let base = reqwest::Url::parse(current).context("parse hf redirect base url")?;
    base.join(location)
        .context("resolve hf redirect location")
        .map(|url| url.to_string())
}

fn validate_repo(repo: &str) -> Result<()> {
    if repo.is_empty()
        || repo.contains("..")
        || repo.starts_with('/')
        || repo.contains('\\')
        || repo.split('/').count() < 2
    {
        bail!("invalid hugging face repo");
    }
    Ok(())
}

pub(crate) fn validate_filename(name: &str) -> Result<()> {
    if name.is_empty() || !name.ends_with(".gguf") {
        bail!("filename must end with .gguf");
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        bail!("invalid filename");
    }
    Ok(())
}

pub(crate) fn safe_join(base: &str, relative: &str) -> Result<PathBuf> {
    let base = PathBuf::from(base);
    let joined = base.join(relative);
    for component in Path::new(relative).components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            bail!("path traversal rejected");
        }
    }
    if !joined.starts_with(&base) {
        bail!("path escapes models directory");
    }
    Ok(joined)
}

fn json_response(status: StatusCode, value: serde_json::Value) -> Result<Response<Full<Bytes>>> {
    let body = Bytes::from(serde_json::to_vec(&value)?);
    let mut resp = Response::new(Full::new(body));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    Ok(resp)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
