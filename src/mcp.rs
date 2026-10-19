//! `typestrict mcp`: the typestrict flow as MCP tools over stdio, with the user's API key. Every tool is a step of
//! `flow`, so an assistant does exactly what the commands do; starting a training takes a quote, then its confirmation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::api::Api;
use crate::flow;

// The server's instructions to the assistant are on `#[tool_handler]` below (it takes a literal).


/// A quote waiting for the user's yes: what it starts and until when it holds.
struct Quote {
    dataset: String,
    encoder: String,
    until: Instant,
}

#[derive(Clone)]
pub struct Server {
    api: Api,
    quotes: Arc<Mutex<HashMap<String, Quote>>>,
}

fn ok(v: Value) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![ContentBlock::text(serde_json::to_string_pretty(&v).unwrap_or_default())]))
}

/// A failed step is a tool result the assistant reads and explains, not a protocol error.
fn failed(e: anyhow::Error) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))]))
}

macro_rules! step {
    ($e:expr) => {
        match $e {
            Ok(v) => ok(v),
            Err(e) => failed(e),
        }
    };
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Id {
    /// The id of a dataset (a training being prepared) or of a model (a training that ran).
    pub id: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Dataset {
    /// The dataset's id.
    pub dataset: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Model {
    /// The model's id.
    pub model: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct ListArgs {
    /// The `next` cursor of the previous page, for the following one.
    pub cursor: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct HfSource {
    /// A public Hugging Face dataset: org/name.
    pub repo: String,
    /// The files of it to use, as paths inside the repository.
    pub files: Vec<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct CreateArgs {
    /// A name for the training.
    pub name: String,
    /// The decision in plain words, e.g. "Which team handles the ticket?".
    pub goal: Option<String>,
    /// Paths of local files to send: tables (CSV, TSV, Excel, ODS, JSON, JSON lines), documents (PDF, DOCX, HTML, text), logs, zip.
    #[serde(default)]
    pub files: Vec<String>,
    /// Public Hugging Face datasets to train from.
    #[serde(default)]
    pub hf: Vec<HfSource>,
    /// Public GitHub repositories to train from (org/name).
    #[serde(default)]
    pub github: Vec<String>,
    /// Decisions to build, when the user already knows them (up to 12).
    #[serde(default)]
    pub questions: Vec<String>,
    /// false only when the user states the files hold no personal data (masking is skipped). Default true.
    pub anonymize: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct StructureArgs {
    pub dataset: String,
    /// The decisions to build (picked from the suggestions or written by the user).
    #[serde(default)]
    pub questions: Vec<String>,
    /// What the user wants changed in the agent's proposal.
    pub feedback: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct NextLabelArgs {
    pub dataset: String,
    /// Skip to the example after this task id.
    pub after: Option<i64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct LabelArgs {
    pub dataset: String,
    /// The task id from next_example_to_label.
    pub task_id: i64,
    /// The user's answer: one of the labels' names.
    pub label: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct QuoteArgs {
    pub dataset: String,
    /// The base model (list_base_models); ts-moberto-150m by default.
    pub encoder: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct StartArgs {
    /// The confirmation of a quote from quote_training, after the user agreed to spend the credit.
    pub confirmation: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct WaitArgs {
    /// A dataset or model id.
    pub id: String,
    /// How long to wait for a change, in seconds (up to 50, 45 by default).
    pub timeout_seconds: Option<u64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct DownloadArgs {
    pub model: String,
    /// The folder to save the package in (decision_ai.json, tokenizer.json, onnx/model_8bit.onnx).
    pub dir: String,
}

#[tool_router]
impl Server {
    pub fn new(api: Api) -> Self {
        Self { api, quotes: Arc::default() }
    }

    #[tool(description = "Every training of the user, newest first, each with its step (1 to 5) and next_action.", annotations(read_only_hint = true))]
    async fn list_trainings(&self, Parameters(a): Parameters<ListArgs>) -> Result<CallToolResult, McpError> {
        step!(flow::trainings(&self.api, 25, a.cursor.as_deref()).await)
    }

    #[tool(description = "One training, by dataset or model id: its data totals, the agent's structure (proposal), the result, the timeline, step and next_action.", annotations(read_only_hint = true))]
    async fn get_training(&self, Parameters(a): Parameters<Id>) -> Result<CallToolResult, McpError> {
        step!(flow::training(&self.api, &a.id).await)
    }

    #[tool(description = "Waits until a training's next_action is no longer `wait` (or the timeout), then returns it. Call it in a loop to follow a training.", annotations(read_only_hint = true))]
    async fn wait_for_change(&self, Parameters(a): Parameters<WaitArgs>) -> Result<CallToolResult, McpError> {
        let limit = Duration::from_secs(a.timeout_seconds.unwrap_or(45).min(50));
        let follow = flow::wait(&self.api, &a.id, Duration::from_secs(5), |_| {});
        match tokio::time::timeout(limit, follow).await {
            Ok(r) => step!(r),
            Err(_) => step!(flow::training(&self.api, &a.id).await),
        }
    }

    #[tool(description = "Starts a training: sends local files and/or public sources (Hugging Face, GitHub) and returns the dataset id. Personal data is masked first unless anonymize is false.")]
    async fn create_dataset(&self, Parameters(a): Parameters<CreateArgs>) -> Result<CallToolResult, McpError> {
        let mut remotes: Vec<flow::Remote> = a.hf.into_iter().map(|h| flow::Remote::Hf { repo: h.repo, files: h.files }).collect();
        remotes.extend(a.github.into_iter().map(|repo| flow::Remote::Github { repo }));
        let new = flow::NewDataset {
            name: a.name,
            goal: a.goal,
            files: a.files.into_iter().map(PathBuf::from).collect(),
            remotes,
            questions: a.questions,
            anonymize: a.anonymize.unwrap_or(true),
        };
        step!(flow::create_dataset(&self.api, new, |_| {}).await.map(|id| json!({ "dataset": id, "next_action": "wait" })))
    }

    #[tool(description = "Runs the structuring agent on a ready dataset (next_action structure), or again with the user's feedback, or with the decisions the user picked (pick_decisions).")]
    async fn structure_dataset(&self, Parameters(a): Parameters<StructureArgs>) -> Result<CallToolResult, McpError> {
        step!(flow::structure(&self.api, &a.dataset, a.questions, a.feedback).await.map(|_| json!({ "dataset": a.dataset, "next_action": "wait" })))
    }

    #[tool(description = "Asks the agent for decisions this data can teach; they show up in get_training as structure.suggestions.")]
    async fn suggest_decisions(&self, Parameters(a): Parameters<Dataset>) -> Result<CallToolResult, McpError> {
        step!(flow::suggest(&self.api, &a.dataset).await.map(|_| json!({ "dataset": a.dataset, "next_action": "wait" })))
    }

    #[tool(description = "The next held-out example to label (next_action label_examples): its text, the labels and a suggestion. Ask the user, then label_example.", annotations(read_only_hint = true))]
    async fn next_example_to_label(&self, Parameters(a): Parameters<NextLabelArgs>) -> Result<CallToolResult, McpError> {
        step!(flow::next_label(&self.api, &a.dataset, a.after).await)
    }

    #[tool(description = "Saves the user's answer to a held-out example.")]
    async fn label_example(&self, Parameters(a): Parameters<LabelArgs>) -> Result<CallToolResult, McpError> {
        step!(flow::save_label(&self.api, &a.dataset, a.task_id, &a.label).await.map(|_| json!({ "saved": true })))
    }

    #[tool(description = "The base models to train on, with which can be picked now.", annotations(read_only_hint = true))]
    async fn list_base_models(&self) -> Result<CallToolResult, McpError> {
        step!(self.api.get("/v1/encoders").await)
    }

    #[tool(description = "The user's credit balance and what a training costs.", annotations(read_only_hint = true))]
    async fn get_credits(&self) -> Result<CallToolResult, McpError> {
        step!(self.api.get("/v1/credits").await)
    }

    #[tool(description = "What starting a training costs, with the balance. Returns a confirmation, valid 10 minutes: tell the user the cost and pass it to start_training only after they agree.", annotations(read_only_hint = true))]
    async fn quote_training(&self, Parameters(a): Parameters<QuoteArgs>) -> Result<CallToolResult, McpError> {
        let encoder = a.encoder.unwrap_or_else(|| flow::DEFAULT_ENCODER.to_owned());
        match flow::quote(&self.api, &a.dataset, &encoder).await {
            Ok(mut q) => {
                let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
                let confirmation = format!("q{nanos:x}");
                let mut quotes = self.quotes.lock().unwrap();
                quotes.retain(|_, q| q.until > Instant::now());
                quotes.insert(confirmation.clone(), Quote { dataset: a.dataset, encoder, until: Instant::now() + Duration::from_secs(600) });
                q["confirmation"] = json!(confirmation);
                ok(q)
            }
            Err(e) => failed(e),
        }
    }

    #[tool(description = "Starts the training a quote describes and spends the credit. Only with the user's agreement to that quote.", annotations(destructive_hint = false, idempotent_hint = false))]
    async fn start_training(&self, Parameters(a): Parameters<StartArgs>) -> Result<CallToolResult, McpError> {
        let quote = self.quotes.lock().unwrap().remove(&a.confirmation);
        let Some(q) = quote.filter(|q| q.until > Instant::now()) else {
            return failed(anyhow::anyhow!("no valid quote for this confirmation: call quote_training again and ask the user"));
        };
        step!(flow::start(&self.api, &q.dataset, &q.encoder).await)
    }

    #[tool(description = "A model's result: the verdict against Jev, both systems' scores, the timeline and until when its files can be downloaded.", annotations(read_only_hint = true))]
    async fn get_model(&self, Parameters(a): Parameters<Model>) -> Result<CallToolResult, McpError> {
        step!(self.api.get(&format!("/v1/models/{}", a.model)).await)
    }

    #[tool(description = "A link to the audit of a model's verdict, which anyone can recompute with https://typestrict.ai/verify_audit.py.", annotations(read_only_hint = true))]
    async fn get_audit(&self, Parameters(a): Parameters<Model>) -> Result<CallToolResult, McpError> {
        step!(self.api.get(&format!("/v1/models/{}/audit", a.model)).await)
    }

    #[tool(description = "Downloads a model's package (decision_ai.json, tokenizer.json, onnx/model_8bit.onnx) into a local folder.")]
    async fn download_model(&self, Parameters(a): Parameters<DownloadArgs>) -> Result<CallToolResult, McpError> {
        step!(flow::download(&self.api, &a.model, &PathBuf::from(&a.dir)).await.map(|files| json!({ "files": files })))
    }
}

#[tool_handler(
    name = "typestrict",
    instructions = "typestrict trains a small decision model on the user's data that beats Jev, the market reference model, \
or the credit comes back. Drive each training by its `next_action` (get_training, wait_for_change): upload_files, wait, structure, \
pick_decisions, label_examples, start_training, retry, download, none, start_over, expired. Show the user the agent's proposal \
(get_training: structure.spec and structure.stats) before training. Starting a training costs a credit: call quote_training, tell the \
user the cost and balance, and only call start_training with the quote's confirmation after the user agrees. \
Docs for assistants: https://docs.typestrict.ai/llms-full.txt"
)]
impl ServerHandler for Server {}

pub async fn serve(api: Api) -> Result<()> {
    let service = Server::new(api).serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
