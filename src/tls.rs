use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio_rustls::rustls;
use tracing::{info, warn};

pub fn init_crypto_provider() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("failed to install rustls ring crypto provider"))?;
    Ok(())
}

pub struct RotatingTls {
    server_config: Arc<ArcSwap<ServerConfig>>,
}

impl RotatingTls {
    pub fn new(cn: &str, rotate_secs: u64) -> Result<Self> {
        let initial = Arc::new(build_server_config(cn)?);
        let server_config = Arc::new(ArcSwap::from(initial));

        if rotate_secs > 0 {
            let rotate_target = Arc::clone(&server_config);
            let cn = cn.to_string();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(rotate_secs));
                interval.tick().await;
                loop {
                    interval.tick().await;
                    match build_server_config(&cn) {
                        Ok(config) => {
                            rotate_target.store(Arc::new(config));
                            info!(cn = %cn, rotate_secs, "rotated in-memory TLS certificate");
                        }
                        Err(err) => warn!(error = %err, "TLS certificate rotation failed"),
                    }
                }
            });
        }

        Ok(Self { server_config })
    }

    pub fn server_config(&self) -> Arc<ServerConfig> {
        self.server_config.load_full()
    }
}

fn build_server_config(cn: &str) -> Result<ServerConfig> {
    let certified = generate_cert_key(cn)?;
    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certified.certs, certified.key)
        .context("build TLS server config")?;
    server_config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(server_config)
}

struct GeneratedCert {
    certs: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
}

fn generate_cert_key(cn: &str) -> Result<GeneratedCert> {
    use rcgen::{CertificateParams, KeyPair, SanType};
    use std::net::{IpAddr, Ipv4Addr};

    // ring (rcgen default) supports ECDSA P-256, not RSA key generation.
    let key_pair = KeyPair::generate().context("generate ECDSA P-256 key pair")?;
    let mut params = CertificateParams::new(vec![
        cn.to_string(),
        "localhost".into(),
        "127.0.0.1".into(),
    ])
    .context("build certificate params")?;
    params
        .subject_alt_names
        .push(SanType::IpAddress(IpAddr::V4(Ipv4Addr::LOCALHOST)));
    let cert = params
        .self_signed(&key_pair)
        .context("sign self-signed certificate")?;

    Ok(GeneratedCert {
        certs: vec![CertificateDer::from(cert.der().to_vec())],
        key: PrivateKeyDer::Pkcs8(key_pair.serialize_der().into()),
    })
}
