use anyhow::{bail, Context, Result};
use std::ffi::CString;
use std::path::Path;
use tracing::info;

const CELL_UID: u32 = 65532;
const CELL_GID: u32 = 65532;

pub fn prepare_models_volume(models_dir: &str) -> Result<()> {
    #[cfg(not(unix))]
    {
        let _ = models_dir;
        return Ok(());
    }

    #[cfg(unix)]
    {
        let uid = unsafe { libc::getuid() };
        if uid != CELL_UID && uid != 0 {
            return Ok(());
        }

        let path = Path::new(models_dir);
        if uid == 0 {
            if !path.exists() {
                std::fs::create_dir_all(path)
                    .with_context(|| format!("create models dir {}", path.display()))?;
            }
            chown_recursive(path, CELL_UID, CELL_GID)?;
            drop_privileges()?;
            info!(dir = %models_dir, "prepared models volume ownership");
            return Ok(());
        }

        if !path.exists() {
            std::fs::create_dir_all(path)
                .with_context(|| format!("create models dir {}", path.display()))?;
        }
        Ok(())
    }
}

#[cfg(unix)]
fn drop_privileges() -> Result<()> {
    if unsafe { libc::setgid(CELL_GID) } != 0 {
        bail!("setgid({CELL_GID}) failed");
    }
    if unsafe { libc::setuid(CELL_UID) } != 0 {
        bail!("setuid({CELL_UID}) failed");
    }
    Ok(())
}

#[cfg(unix)]
fn chown_recursive(path: &Path, uid: u32, gid: u32) -> Result<()> {
    let c_path = CString::new(path.to_str().context("non-utf8 models path")?)?;
    if unsafe { libc::chown(c_path.as_ptr(), uid, gid) } != 0 {
        bail!("chown failed for {}", path.display());
    }

    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            chown_recursive(&entry?.path(), uid, gid)?;
        }
    }
    Ok(())
}
