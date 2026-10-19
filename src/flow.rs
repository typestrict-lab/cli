//! The steps of a training, shared by the commands and the MCP tools so both do exactly the same.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::api::{Api, ApiError};

/// The base model trained on when none is given.
pub const DEFAULT_ENCODER: &str = "ts-moberto-150m";

fn content_type(name: &str) -> &'static str {
    match Path::new(name).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("csv") => "text/csv",
        Some("tsv") => "text/tab-separated-values",
        Some("json") => "application/json",
        Some("jsonl" | "ndjson") => "application/x-ndjson",
        Some("xlsx") => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        Some("xls") => "application/vnd.ms-excel",
        Some("ods") => "application/vnd.oasis.opendocument.spreadsheet",
        Some("pdf") => "application/pdf",
        Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        Some("html" | "htm") => "text/html",
        Some("txt" | "md" | "log") => "text/plain",
        Some("zip") => "application/zip",
        _ => "application/octet-stream",
    }
}

/// A public source to train from.
pub enum Remote {
    /// A Hugging Face dataset and the files of it to use.
    Hf {
        repo: String,
        files: Vec<String>,
    },
    Github {
        repo: String,
    },
}

/// A file to send: one on this machine, or its contents (what an assistant passes to the remote MCP server).
pub enum File {
    Path(PathBuf),
    Inline { name: String, bytes: Vec<u8> },
}

impl File {
    fn name(&self) -> Result<String> {
        match self {
            File::Path(p) => Ok(p.file_name().and_then(|n| n.to_str()).context("file name")?.to_owned()),
            File::Inline { name, .. } => Ok(name.clone()),
        }
    }

    fn size(&self) -> Result<u64> {
        match self {
            File::Path(p) => {
                let meta = std::fs::metadata(p).with_context(|| format!("cannot read {}", p.display()))?;
                if !meta.is_file() {
                    bail!("{} is not a file", p.display());
                }
                Ok(meta.len())
            }
            File::Inline { bytes, .. } => Ok(bytes.len() as u64),
        }
    }

    fn bytes(&self) -> Result<Vec<u8>> {
        match self {
            File::Path(p) => Ok(std::fs::read(p)?),
            File::Inline { bytes, .. } => Ok(bytes.clone()),
        }
    }
}

pub struct NewDataset {
    pub name: String,
    pub goal: Option<String>,
    pub files: Vec<File>,
    pub remotes: Vec<Remote>,
    pub questions: Vec<String>,
    pub anonymize: bool,
}

/// Creates the dataset, uploads every file to its URL and confirms it. Returns the dataset's id.
pub async fn create_dataset(api: &Api, d: NewDataset, mut progress: impl FnMut(&str)) -> Result<String> {
    if d.files.is_empty() && d.remotes.is_empty() {
        bail!("give at least one file or public source");
    }
    let mut files = Vec::new();
    for f in &d.files {
        let filename = f.name()?;
        files.push(json!({ "filename": filename, "content_type": content_type(&filename), "size": f.size()? }));
    }
    let remotes: Vec<Value> = d
        .remotes
        .iter()
        .map(|r| match r {
            Remote::Hf { repo, files } => json!({ "kind": "hf", "repo": repo, "files": files }),
            Remote::Github { repo } => json!({ "kind": "github", "repo": repo }),
        })
        .collect();
    let mut body = json!({ "name": d.name, "files": files, "remotes": remotes, "questions": d.questions, "anonymize": d.anonymize });
    if let Some(g) = &d.goal {
        body["goal"] = json!(g);
    }
    let created = api.post("/v1/datasets", body).await?;
    let id = created["id"].as_str().context("no dataset id in the answer")?.to_owned();
    for (file, up) in d.files.iter().zip(created["uploads"].as_array().cloned().unwrap_or_default()) {
        progress(&format!("uploading {}", file.name()?));
        api.upload(up["url"].as_str().context("upload url")?, file.bytes()?).await?;
        api.post(
            &format!("/v1/datasets/{id}/files/{}/complete", up["file_id"].as_str().context("file id")?),
            json!({}),
        )
        .await?;
    }
    Ok(id)
}

/// A training by the id of its dataset or of its model, with `kind` set to which one it is.
pub async fn training(api: &Api, id: &str) -> Result<Value> {
    match api.get(&format!("/v1/datasets/{id}")).await {
        Ok(mut d) => {
            d["kind"] = json!("dataset");
            Ok(d)
        }
        Err(e) if e.downcast_ref::<ApiError>().is_some_and(|a| a.code == "not_found") => {
            let mut m = api.get(&format!("/v1/models/{id}")).await?;
            m["kind"] = json!("model");
            Ok(m)
        }
        Err(e) => Err(e),
    }
}

pub async fn trainings(api: &Api, limit: u32, cursor: Option<&str>) -> Result<Value> {
    let mut path = format!("/v1/trainings?limit={limit}");
    if let Some(c) = cursor {
        path.push_str(&format!("&cursor={c}"));
    }
    api.get(&path).await
}

/// Reads the training again until it no longer waits on us, telling each new step. Returns the last read.
pub async fn wait(api: &Api, id: &str, every: Duration, mut progress: impl FnMut(&Value)) -> Result<Value> {
    let mut last = String::new();
    loop {
        let t = training(api, id).await?;
        let now = format!("{} {}", t["step"], t["next_action"]);
        if now != last {
            progress(&t);
            last = now;
        }
        if t["next_action"] != "wait" {
            return Ok(t);
        }
        tokio::time::sleep(every).await;
    }
}

pub async fn structure(api: &Api, dataset: &str, questions: Vec<String>, feedback: Option<String>) -> Result<()> {
    let mut body = json!({});
    if !questions.is_empty() {
        body["questions"] = json!(questions);
    }
    if let Some(f) = feedback {
        body["feedback"] = json!(f);
    }
    api.post(&format!("/v1/datasets/{dataset}/structure"), body).await?;
    Ok(())
}

pub async fn suggest(api: &Api, dataset: &str) -> Result<()> {
    api.post(&format!("/v1/datasets/{dataset}/suggest"), json!({})).await?;
    Ok(())
}

pub async fn next_label(api: &Api, dataset: &str, after: Option<i64>) -> Result<Value> {
    let q = after.map(|a| format!("?after={a}")).unwrap_or_default();
    api.get(&format!("/v1/datasets/{dataset}/labels/next{q}")).await
}

pub async fn save_label(api: &Api, dataset: &str, task: i64, label: &str) -> Result<()> {
    api.post(&format!("/v1/datasets/{dataset}/labels/{task}"), json!({ "label": label })).await?;
    Ok(())
}

/// What starting a training costs, with the balance and whether the base model can be picked now.
pub async fn quote(api: &Api, dataset: &str, encoder: &str) -> Result<Value> {
    let encoders = api.get("/v1/encoders").await?;
    let base = encoders.as_array().and_then(|es| es.iter().find(|e| e["id"] == encoder)).cloned();
    let Some(base) = base else {
        let ids: Vec<&str> =
            encoders.as_array().map(|es| es.iter().filter_map(|e| e["id"].as_str()).collect()).unwrap_or_default();
        bail!("unknown base model {encoder}; the base models are: {}", ids.join(", "));
    };
    if base["available"] != true {
        bail!("{} cannot be trained on yet (coming soon)", base["name"].as_str().unwrap_or(encoder));
    }
    let credits = api.get("/v1/credits").await?;
    let d = training(api, dataset).await?;
    if d["next_action"] != "start_training" {
        bail!("this training is not ready to start: its next action is {}", d["next_action"]);
    }
    Ok(json!({
        "dataset": dataset,
        "name": d["dataset"]["name"],
        "base_model": base["name"],
        "encoder": encoder,
        "cost_credits": credits["per_model"],
        "balance": credits["balance"],
        "enough": credits["balance"].as_i64() >= credits["per_model"].as_i64(),
        "guarantee": "the credit comes back if the model does not beat Jev on the held-out examples, or if the training fails",
    }))
}

pub async fn start(api: &Api, dataset: &str, encoder: &str) -> Result<Value> {
    api.post(&format!("/v1/datasets/{dataset}/models"), json!({ "encoder": encoder })).await
}

/// The package's files and their download links (valid for a while), for a caller that cannot write to this machine.
pub async fn download_links(api: &Api, model: &str) -> Result<Value> {
    api.get(&format!("/v1/models/{model}/files")).await
}

/// Saves the model package (decision_ai.json, tokenizer.json, onnx/model_8bit.onnx) under `dir`.
pub async fn download(api: &Api, model: &str, dir: &Path) -> Result<Vec<PathBuf>> {
    let files = api.get(&format!("/v1/models/{model}/files")).await?;
    let mut written = Vec::new();
    for rel in ["decision_ai.json", "tokenizer.json", "onnx/model_8bit.onnx"] {
        let url = files[rel].as_str().with_context(|| format!("{rel} is not in the package"))?;
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, api.download(url).await?)?;
        written.push(path);
    }
    Ok(written)
}

/// The short story of a training for people: its step and what it needs next.
pub fn describe(t: &Value) -> String {
    let steps = ["data", "review", "base model", "training", "result"];
    let step = t["step"].as_u64().unwrap_or(1) as usize;
    let name = t["dataset"]["name"].as_str().or(t["model"]["name"].as_str()).or(t["name"].as_str()).unwrap_or("");
    let next = match t["next_action"].as_str().unwrap_or("") {
        "upload_files" => "the files are still to be uploaded".to_owned(),
        "wait" => "working on it".to_owned(),
        "start_over" => format!("the data could not be read: {}", t["dataset"]["error"].as_str().unwrap_or("")),
        "structure" => "ready to be structured: `typestrict structure`".to_owned(),
        "pick_decisions" => {
            "pick the decisions to build: `typestrict review` then `typestrict structure --question ...`".to_owned()
        }
        "label_examples" => "label the test examples: `typestrict label`".to_owned(),
        "start_training" => "ready to train: `typestrict run`".to_owned(),
        "retry" => "the training failed and the credit came back: run it again".to_owned(),
        "download" => match t["model"]["verdict"].as_str() {
            Some("beats") => "done, it beats Jev: `typestrict result`, `typestrict pull`".to_owned(),
            _ => "done: `typestrict result`, `typestrict pull`".to_owned(),
        },
        "none" => "done; its files are no longer kept".to_owned(),
        "expired" => "its data expired: upload it again to train".to_owned(),
        other => other.to_owned(),
    };
    format!("{name}: step {step} of 5 ({}), {next}", steps[step.clamp(1, 5) - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_go_up_with_their_type() {
        assert_eq!(content_type("a/Tickets.CSV"), "text/csv");
        assert_eq!(content_type("pack.zip"), "application/zip");
        assert_eq!(content_type("notes"), "application/octet-stream");
    }

    #[test]
    fn a_training_reads_as_its_step_and_what_it_needs() {
        let t = json!({ "dataset": { "name": "tickets" }, "step": 3, "next_action": "start_training" });
        assert_eq!(describe(&t), "tickets: step 3 of 5 (base model), ready to train: `typestrict run`");
        let m = json!({ "model": { "name": "churn", "verdict": "beats" }, "step": 5, "next_action": "download" });
        assert!(describe(&m).contains("it beats Jev"));
        let row = json!({ "name": "raw", "step": 1, "next_action": "start_over", "dataset": { "error": "bad zip" } });
        assert!(describe(&row).ends_with("the data could not be read: bad zip"));
    }
}
