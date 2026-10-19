//! Where the API key comes from: `TYPESTRICT_API_KEY`, or the one `typestrict login` saved (only readable by the user).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn path() -> Result<PathBuf> {
    Ok(dirs::config_dir().context("no config directory on this system")?.join("typestrict").join("credentials"))
}

pub fn key() -> Result<String> {
    if let Ok(k) = std::env::var("TYPESTRICT_API_KEY")
        && !k.trim().is_empty()
    {
        return Ok(k.trim().to_owned());
    }
    match std::fs::read_to_string(path()?) {
        Ok(k) if !k.trim().is_empty() => Ok(k.trim().to_owned()),
        _ => bail!(
            "no API key: run `typestrict login`, or set TYPESTRICT_API_KEY (create a key at https://typestrict.ai/settings#api-keys)"
        ),
    }
}

pub fn save(key: &str) -> Result<PathBuf> {
    let p = path()?;
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(&p, format!("{key}\n"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(p)
}

pub fn forget() -> Result<bool> {
    let p = path()?;
    if p.exists() {
        std::fs::remove_file(p)?;
        return Ok(true);
    }
    Ok(false)
}
