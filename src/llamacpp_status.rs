use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Returns model id -> router status string (e.g. `loaded`, `unloaded`).
pub fn parse_router_model_states(payload: &Value) -> HashMap<String, String> {
    let Some(data) = payload.get("data").and_then(|d| d.as_array()) else {
        return HashMap::new();
    };

    let mut states = HashMap::new();
    for entry in data {
        let Some(id) = entry.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let status = entry
            .get("status")
            .and_then(parse_router_status_value)
            .unwrap_or_else(|| "unloaded".to_string());
        states.insert(id.to_string(), status);
    }
    states
}

fn parse_router_status_value(status: &Value) -> Option<String> {
    if let Some(value) = status.as_str() {
        return Some(value.to_string());
    }
    status
        .get("value")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

pub fn router_status_is_loaded(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "loaded" | "loading" | "sleeping" | "running"
    )
}

pub fn model_path_matches(loaded_path: &str, model_id: &str) -> bool {
    let loaded = Path::new(loaded_path);
    let model = Path::new(model_id);
    if loaded == model {
        return true;
    }
    if loaded.ends_with(model) {
        return true;
    }
    loaded
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == model_id)
}

pub fn is_model_loaded(
    model_id: &str,
    single_model_path: Option<&str>,
    router_states: &HashMap<String, String>,
) -> bool {
    if single_model_path
        .is_some_and(|path| model_path_matches(path, model_id))
    {
        return true;
    }
    if router_state_is_loaded(router_states, model_id) {
        return true;
    }
    let stem = gguf_stem(model_id);
    if stem != model_id && router_state_is_loaded(router_states, stem) {
        return true;
    }
    router_states.keys().any(|router_id| {
        router_id_matches_local_ref(router_id, model_id)
            && router_state_is_loaded(router_states, router_id)
    })
}

fn router_state_is_loaded(states: &HashMap<String, String>, key: &str) -> bool {
    states
        .get(key)
        .is_some_and(|status| router_status_is_loaded(status))
}

fn router_id_matches_local_ref(router_id: &str, model_ref: &str) -> bool {
    let stem = gguf_stem(model_ref);
    router_id == model_ref
        || router_id == stem
        || Path::new(router_id)
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name == model_ref || name == stem)
}

pub fn loaded_status_json() -> Value {
    serde_json::json!({
        "loaded": true,
        "processor": "gpu",
        "cpu_pct": 0,
        "gpu_pct": 100
    })
}

pub fn unloaded_status_json() -> Value {
    serde_json::json!({
        "loaded": false,
        "processor": "unknown",
        "cpu_pct": 0,
        "gpu_pct": 0
    })
}

pub async fn fetch_router_models(upstream_base: &str, reload: bool) -> Option<Value> {
    let base = upstream_base.trim_end_matches('/');
    let url = if reload {
        format!("{base}/models?reload=1")
    } else {
        format!("{base}/models")
    };
    let resp = router_http_client().get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<Value>().await.ok()
}

pub async fn fetch_router_model_states(upstream_base: &str) -> HashMap<String, String> {
    fetch_router_model_states_with_reload(upstream_base, false).await
}

pub async fn fetch_router_model_states_with_reload(
    upstream_base: &str,
    reload: bool,
) -> HashMap<String, String> {
    let Some(payload) = fetch_router_models(upstream_base, reload).await else {
        return HashMap::new();
    };
    parse_router_model_states(&payload)
}

fn router_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

fn gguf_stem(filename: &str) -> &str {
    filename
        .strip_suffix(".gguf")
        .or_else(|| filename.strip_suffix(".GGUF"))
        .unwrap_or(filename)
}

fn router_entry_matches_ref(id: &str, entry: &Value, model_ref: &str) -> bool {
    let stem = gguf_stem(model_ref);
    if id == model_ref || id == stem {
        return true;
    }
    if Path::new(id)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == model_ref || name == stem)
    {
        return true;
    }
    if let Some(aliases) = entry.get("aliases").and_then(|v| v.as_array()) {
        for alias in aliases {
            if let Some(alias) = alias.as_str() {
                if alias == model_ref || alias == stem {
                    return true;
                }
            }
        }
    }
    false
}

/// Map a local `.gguf` filename (or stem) to the router model id from `GET /models`.
pub fn resolve_router_model_id(payload: &Value, model_ref: &str) -> Option<String> {
    let data = payload.get("data").and_then(|d| d.as_array())?;
    for entry in data {
        let Some(id) = entry.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        if router_entry_matches_ref(id, entry, model_ref) {
            return Some(id.to_string());
        }
    }
    None
}

pub fn router_load_error_is_already_running(status: reqwest::StatusCode, body: &str) -> bool {
    status == reqwest::StatusCode::BAD_REQUEST
        && body.to_ascii_lowercase().contains("already running")
}

pub fn collect_model_ids(
    models_dir: &str,
    router_states: &HashMap<String, String>,
) -> Vec<String> {
    let mut ids = HashSet::new();
    if let Ok(rd) = std::fs::read_dir(models_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            let is_gguf = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.to_ascii_lowercase().ends_with(".gguf"));
            if !is_gguf {
                continue;
            }
            if let Some(id) = path.file_name().and_then(|n| n.to_str()) {
                ids.insert(id.to_string());
            }
        }
    }
    for id in router_states.keys() {
        if id.to_ascii_lowercase().ends_with(".gguf") {
            ids.insert(id.clone());
        }
    }
    let mut sorted: Vec<_> = ids.into_iter().collect();
    sorted.sort();
    sorted
}

pub fn model_entry(id: &str, loaded: bool) -> Value {
    serde_json::json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": "inference-cell",
        "_status": if loaded { loaded_status_json() } else { unloaded_status_json() },
    })
}

pub fn build_v1_models_payload(
    models_dir: &str,
    single_model_path: Option<&str>,
    router_states: &HashMap<String, String>,
) -> Value {
    let ids = collect_model_ids(models_dir, router_states);
    let data: Vec<Value> = ids
        .iter()
        .map(|id| {
            let loaded = is_model_loaded(id, single_model_path, router_states);
            model_entry(id, loaded)
        })
        .collect();
    serde_json::json!({ "object": "list", "data": data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_router_status_object_and_string() {
        let payload = serde_json::json!({
            "data": [
                { "id": "a.gguf", "status": { "value": "loaded" } },
                { "id": "b.gguf", "status": "unloaded" }
            ]
        });
        let states = parse_router_model_states(&payload);
        assert_eq!(states.get("a.gguf").map(String::as_str), Some("loaded"));
        assert_eq!(states.get("b.gguf").map(String::as_str), Some("unloaded"));
    }

    #[test]
    fn router_loaded_status_values() {
        assert!(router_status_is_loaded("loaded"));
        assert!(router_status_is_loaded("loading"));
        assert!(!router_status_is_loaded("unloaded"));
    }

    #[test]
    fn model_path_matches_basename_and_suffix() {
        assert!(model_path_matches("/models/foo.gguf", "foo.gguf"));
        assert!(model_path_matches("models/foo.gguf", "foo.gguf"));
        assert!(!model_path_matches("/models/foo.gguf", "bar.gguf"));
    }

    #[test]
    fn is_model_loaded_uses_router_state_in_router_mode() {
        let mut states = HashMap::new();
        states.insert("a.gguf".to_string(), "loaded".to_string());
        assert!(is_model_loaded("a.gguf", None, &states));
        assert!(!is_model_loaded("b.gguf", None, &states));
    }

    #[test]
    fn is_model_loaded_uses_single_model_path() {
        assert!(is_model_loaded(
            "foo.gguf",
            Some("/models/foo.gguf"),
            &HashMap::new()
        ));
    }

    #[test]
    fn is_model_loaded_matches_router_stem_ids() {
        let mut states = HashMap::new();
        states.insert("gemma-4".to_string(), "loaded".to_string());
        assert!(is_model_loaded("gemma-4.gguf", None, &states));
    }

    #[test]
    fn resolve_router_model_id_matches_stem_and_filename() {
        let payload = serde_json::json!({
            "data": [
                { "id": "gemma-4-12b-it-qat-q4_0", "status": { "value": "unloaded" } },
                { "id": "other.gguf", "aliases": ["alias-model"], "status": { "value": "unloaded" } }
            ]
        });
        assert_eq!(
            resolve_router_model_id(&payload, "gemma-4-12b-it-qat-q4_0.gguf").as_deref(),
            Some("gemma-4-12b-it-qat-q4_0")
        );
        assert_eq!(
            resolve_router_model_id(&payload, "alias-model.gguf").as_deref(),
            Some("other.gguf")
        );
    }

    #[test]
    fn router_load_error_is_already_running_detects_message() {
        assert!(router_load_error_is_already_running(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"model is already running"}"#
        ));
    }

    #[test]
    fn build_payload_marks_router_loaded_models() {
        let mut states = HashMap::new();
        states.insert("remote.gguf".to_string(), "loaded".to_string());
        let payload = build_v1_models_payload("/models", None, &states);
        let data = payload.get("data").and_then(|d| d.as_array()).unwrap();
        let remote = data
            .iter()
            .find(|m| m.get("id").and_then(|v| v.as_str()) == Some("remote.gguf"))
            .unwrap();
        assert_eq!(
            remote
                .get("_status")
                .and_then(|s| s.get("loaded"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }
}
