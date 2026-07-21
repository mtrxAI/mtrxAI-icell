#[cfg(test)]
mod auth_tests {
    use super::super::{
        encode_hf_path, hf_hub_host, hf_resolve_download_url, hf_should_attach_auth, safe_join,
        select_gguf_files_for_quant, validate_filename,
    };

    #[test]
    fn rejects_path_traversal_in_model_paths() {
        assert!(validate_filename("../evil.gguf").is_err());
        assert!(safe_join("/models", "ok.gguf").is_ok());
        assert!(safe_join("/models", "../evil.gguf").is_err());
    }

    #[test]
    fn encodes_hf_path_segments() {
        assert_eq!(
            encode_hf_path("foo bar/Q4_K_M/file name.gguf"),
            "foo%20bar/Q4_K_M/file%20name.gguf"
        );
    }

    #[test]
    fn hf_resolve_download_url_includes_download_query() {
        let url = hf_resolve_download_url("org/repo", "main", "model.gguf");
        assert!(url.ends_with("model.gguf?download=true"));
    }

    #[test]
    fn auth_only_attaches_on_huggingface_host() {
        assert!(hf_should_attach_auth(
            "https://huggingface.co/org/repo/resolve/main/a.gguf?download=true",
            Some("hf_test")
        )
        .is_some());
        assert!(hf_should_attach_auth(
            "https://cas-bridge.xethub.hf.co/x?origin=https://huggingface.co/org/repo",
            Some("hf_test")
        )
        .is_none());
        assert!(hf_should_attach_auth("https://huggingface.co/x", Some("  ")).is_none());
    }

    #[test]
    fn hf_hub_host_parses_authority() {
        assert_eq!(
            hf_hub_host("https://cdn-lfs.hf.co/x?origin=https://huggingface.co/a/b").as_deref(),
            Some("cdn-lfs.hf.co")
        );
    }

    #[test]
    fn select_gguf_prefers_single_file_over_split_shards() {
        let files = vec![
            "model-q4_k_m.gguf".to_string(),
            "model-q8_0-00001-of-00003.gguf".to_string(),
            "model-q8_0-00002-of-00003.gguf".to_string(),
        ];
        let selected = select_gguf_files_for_quant(&files, Some("Q4_K_M"));
        assert_eq!(selected, vec!["model-q4_k_m.gguf".to_string()]);
    }

    #[test]
    fn select_gguf_downloads_all_split_shards_for_quant() {
        let files = vec![
            "model-q8_0-00001-of-00003.gguf".to_string(),
            "model-q8_0-00002-of-00003.gguf".to_string(),
            "model-q8_0-00003-of-00003.gguf".to_string(),
            "model-q4_k_m.gguf".to_string(),
        ];
        let selected = select_gguf_files_for_quant(&files, Some("Q8_0"));
        assert_eq!(
            selected,
            vec![
                "model-q8_0-00001-of-00003.gguf".to_string(),
                "model-q8_0-00002-of-00003.gguf".to_string(),
                "model-q8_0-00003-of-00003.gguf".to_string(),
            ]
        );
    }
}
