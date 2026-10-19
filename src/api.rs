//! The typestrict API over HTTPS: every call carries the API key, every error comes back as `{error, message}`.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::{Method, StatusCode};
use serde_json::Value;

pub const DEFAULT_URL: &str = "https://api.typestrict.ai";

#[derive(Clone)]
pub struct Api {
    base: String,
    key: String,
    http: reqwest::Client,
}

/// An error answered by the API, with its stable code (`unauthorized`, `not_found`, `bad_request`, `no_credits`,
/// `expired`, `internal`) and its message.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code.as_str() {
            "unauthorized" => write!(f, "the API key was refused: run `typestrict login` or set TYPESTRICT_API_KEY"),
            "no_credits" => write!(f, "not enough credits: buy more at https://typestrict.ai/billing"),
            "expired" => write!(f, "expired under the retention policy: {}", self.message),
            _ => write!(f, "{} ({})", self.message, self.status.as_u16()),
        }
    }
}

impl std::error::Error for ApiError {}

impl Api {
    pub fn new(base: &str, key: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("typestrict-cli/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self { base: base.trim_end_matches('/').to_owned(), key: key.to_owned(), http })
    }

    pub async fn get(&self, path: &str) -> Result<Value> {
        self.call(Method::GET, path, None).await
    }

    pub async fn post(&self, path: &str, body: Value) -> Result<Value> {
        self.call(Method::POST, path, Some(body)).await
    }

    async fn call(&self, method: Method, path: &str, body: Option<Value>) -> Result<Value> {
        let mut req = self.http.request(method, format!("{}{path}", self.base)).bearer_auth(&self.key);
        if let Some(b) = body {
            req = req.json(&b);
        }
        let res = req.send().await.with_context(|| format!("could not reach {}", self.base))?;
        let status = res.status();
        let text = res.text().await?;
        if !status.is_success() {
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            return Err(ApiError {
                status,
                code: v["error"].as_str().unwrap_or("internal").to_owned(),
                message: v["message"].as_str().map(str::to_owned).unwrap_or(text),
            }
            .into());
        }
        Ok(if text.is_empty() { Value::Null } else { serde_json::from_str(&text)? })
    }

    /// Sends a file to a presigned upload URL (no key: the URL is the permission).
    pub async fn upload(&self, url: &str, bytes: Vec<u8>) -> Result<()> {
        let res = self.http.put(url).body(bytes).timeout(Duration::from_secs(1800)).send().await?;
        if !res.status().is_success() {
            bail!("upload failed: {}", res.status());
        }
        Ok(())
    }

    /// Downloads a presigned link.
    pub async fn download(&self, url: &str) -> Result<Vec<u8>> {
        let res = self.http.get(url).timeout(Duration::from_secs(1800)).send().await?;
        if !res.status().is_success() {
            bail!("download failed: {}", res.status());
        }
        Ok(res.bytes().await?.to_vec())
    }
}
