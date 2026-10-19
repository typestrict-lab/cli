//! typestrict: train, follow and download typestrict decision models from the terminal; `typestrict mcp` is the same flow
//! as an MCP server (stdio) for AI assistants. Docs: https://docs.typestrict.ai/cli

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use typestrict::{config, flow, mcp};

use typestrict::api::Api;

#[derive(Parser)]
#[command(
    name = "typestrict",
    version,
    about = "Train decision models that beat Jev, or the credit comes back. Docs: https://docs.typestrict.ai/cli"
)]
struct Cli {
    /// Print the API's answers as JSON, for scripts.
    #[arg(long, global = true)]
    json: bool,
    /// The API to talk to.
    #[arg(long, global = true, env = "TYPESTRICT_API_URL", default_value = typestrict::api::DEFAULT_URL, hide_default_value = true)]
    api_url: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Save an API key (create one at https://typestrict.ai/settings#api-keys).
    Login {
        /// The key; asked for (hidden) when left out.
        #[arg(long)]
        key: Option<String>,
    },
    /// Forget the saved key.
    Logout,
    /// Send data and start preparing a training: files, Hugging Face datasets or GitHub repositories.
    Train {
        /// Files to send (tables, documents, logs, zip).
        files: Vec<PathBuf>,
        /// The decision, in plain words: "Which team handles the ticket?"
        #[arg(long)]
        goal: Option<String>,
        /// Name of the training (the first file's name by default).
        #[arg(long)]
        name: Option<String>,
        /// A decision to build; repeat for several.
        #[arg(long = "question")]
        questions: Vec<String>,
        /// A Hugging Face dataset and its files: org/name:path/a.csv,path/b.csv
        #[arg(long)]
        hf: Vec<String>,
        /// A GitHub repository: org/name
        #[arg(long)]
        github: Vec<String>,
        /// The files hold no personal data: skip masking (on your responsibility).
        #[arg(long)]
        no_anonymize: bool,
        /// Follow it until it needs you.
        #[arg(long)]
        watch: bool,
    },
    /// Every training and its step, or one of them (by dataset or model id).
    Status {
        id: Option<String>,
        /// Follow it until it needs you or ends; exits 1 if it failed.
        #[arg(long)]
        watch: bool,
        /// How many trainings to list.
        #[arg(long, default_value_t = 25)]
        limit: u32,
    },
    /// The agent's proposal for a dataset: the decision, the answers and how many examples each has.
    Review { dataset: String },
    /// Start the structuring agent, or run it again with your feedback or the decisions you picked.
    Structure {
        dataset: String,
        /// What to change in the proposal.
        #[arg(long)]
        feedback: Option<String>,
        /// A decision to build; repeat for several.
        #[arg(long = "question")]
        questions: Vec<String>,
    },
    /// Ask the agent for decisions this data can teach.
    Suggest { dataset: String },
    /// Answer the held-out examples, one by one.
    Label { dataset: String },
    /// The base models.
    Encoders,
    /// Your credits.
    Credits,
    /// Start the training (costs a credit, asked before spending).
    Run {
        dataset: String,
        /// The base model.
        #[arg(long, default_value = flow::DEFAULT_ENCODER)]
        base: String,
        /// Do not ask before spending the credit.
        #[arg(long)]
        yes: bool,
        /// Follow it until the result.
        #[arg(long)]
        watch: bool,
    },
    /// The result of a model against Jev.
    Result { model: String },
    /// Download a model's package (decision_ai.json, tokenizer.json, onnx/model_8bit.onnx).
    Pull {
        model: String,
        /// Where to save it (./<name>-v<version> by default).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// The typestrict MCP server over stdio, for Claude Code, Cursor, Codex and other assistants.
    Mcp,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn api(cli: &Cli) -> Result<Api> {
    Api::new(&cli.api_url, &config::key()?)
}

fn print(json: bool, v: &Value, human: impl FnOnce()) {
    if json {
        println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
    } else {
        human();
    }
}

fn ask(prompt: &str) -> Result<String> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_owned())
}

async fn watch(api: &Api, id: &str, json: bool) -> Result<ExitCode> {
    let t = flow::wait(api, id, Duration::from_secs(10), |t| {
        if !json {
            println!("{}", flow::describe(t));
        }
    })
    .await?;
    print(json, &t, || {});
    Ok(if matches!(t["next_action"].as_str(), Some("start_over" | "retry" | "expired")) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

async fn run(cli: Cli) -> Result<ExitCode> {
    let json = cli.json;
    match &cli.command {
        Command::Login { key } => {
            let key = match key {
                Some(k) => k.clone(),
                None => {
                    println!("Create a key at https://typestrict.ai/settings#api-keys and paste it here.");
                    rpassword::prompt_password("API key: ")?
                }
            };
            let credits = Api::new(&cli.api_url, key.trim())?.get("/v1/credits").await?;
            let path = config::save(key.trim())?;
            println!("Signed in ({} credits). Key saved to {}.", credits["balance"], path.display());
        }
        Command::Logout => println!("{}", if config::forget()? { "Key removed." } else { "No saved key." }),
        Command::Train { files, goal, name, questions, hf, github, no_anonymize, watch: follow } => {
            let api = api(&cli)?;
            let mut remotes = Vec::new();
            for h in hf {
                let (repo, paths) = h.split_once(':').context("--hf takes org/name:file,file")?;
                remotes.push(flow::Remote::Hf {
                    repo: repo.to_owned(),
                    files: paths.split(',').map(str::to_owned).collect(),
                });
            }
            remotes.extend(github.iter().map(|r| flow::Remote::Github { repo: r.clone() }));
            let name = name
                .clone()
                .or_else(|| files.first().and_then(|f| f.file_stem()).and_then(|s| s.to_str()).map(str::to_owned))
                .or_else(|| hf.first().or(github.first()).map(|r| r.split(':').next().unwrap_or(r).to_owned()))
                .context("give a file, a public source or --name")?;
            let new = flow::NewDataset {
                name,
                goal: goal.clone(),
                files: files.iter().cloned().map(flow::File::Path).collect(),
                remotes,
                questions: questions.clone(),
                anonymize: !no_anonymize,
            };
            let id = flow::create_dataset(&api, new, |m| {
                if !json {
                    println!("{m}");
                }
            })
            .await?;
            if !json {
                println!(
                    "Training {id} started: reading and masking the data. Follow it with `typestrict status {id} --watch`."
                );
            }
            if *follow {
                return watch(&api, &id, json).await;
            }
            print(json, &json!({ "dataset": id }), || {});
        }
        Command::Status { id, watch: follow, limit } => {
            let api = api(&cli)?;
            match id {
                Some(id) if *follow => return watch(&api, id, json).await,
                Some(id) => {
                    let t = flow::training(&api, id).await?;
                    print(json, &t, || println!("{}", flow::describe(&t)));
                }
                None => {
                    let page = flow::trainings(&api, *limit, None).await?;
                    print(json, &page, || {
                        let rows = page["data"].as_array().cloned().unwrap_or_default();
                        if rows.is_empty() {
                            println!("No trainings yet: `typestrict train <file> --goal \"...\"`.");
                        }
                        for r in rows {
                            println!("{}  {}", r["id"].as_str().unwrap_or(""), flow::describe(&r));
                        }
                    });
                }
            }
        }
        Command::Review { dataset } => {
            let d = flow::training(&api(&cli)?, dataset).await?;
            print(json, &d, || review(&d));
        }
        Command::Structure { dataset, feedback, questions } => {
            flow::structure(&api(&cli)?, dataset, questions.clone(), feedback.clone()).await?;
            print(json, &json!({ "dataset": dataset, "status": "structuring" }), || {
                println!("The agent is structuring the data. Follow it with `typestrict status {dataset} --watch`.")
            });
        }
        Command::Suggest { dataset } => {
            flow::suggest(&api(&cli)?, dataset).await?;
            print(json, &json!({ "dataset": dataset, "status": "suggesting" }), || {
                println!("The agent is looking for decisions. Then `typestrict review {dataset}`.")
            });
        }
        Command::Label { dataset } => label(&api(&cli)?, dataset).await?,
        Command::Encoders => {
            let es = api(&cli)?.get("/v1/encoders").await?;
            print(json, &es, || {
                for e in es.as_array().cloned().unwrap_or_default() {
                    let state = if e["available"] == true { "available" } else { "coming soon" };
                    let s = |k: &str| e[k].as_str().unwrap_or("").to_owned();
                    println!("{:<18} {:<18} {:<6} {:<8} {state}", s("id"), s("name"), s("size"), s("languages"));
                }
            });
        }
        Command::Credits => {
            let c = api(&cli)?.get("/v1/credits").await?;
            print(json, &c, || println!("{} credits; a training costs {}.", c["balance"], c["per_model"]));
        }
        Command::Run { dataset, base, yes, watch: follow } => {
            let api = api(&cli)?;
            let q = flow::quote(&api, dataset, base).await?;
            if q["enough"] != true {
                bail!(
                    "not enough credits ({} of {}): buy more at https://typestrict.ai/billing",
                    q["balance"],
                    q["cost_credits"]
                );
            }
            if !yes {
                let answer = ask(&format!(
                    "Train {} on {} for {} credit (balance {})? The credit comes back if it does not beat Jev. [y/N] ",
                    q["name"].as_str().unwrap_or(dataset),
                    q["base_model"].as_str().unwrap_or(base),
                    q["cost_credits"],
                    q["balance"]
                ))?;
                if !matches!(answer.to_lowercase().as_str(), "y" | "yes" | "s" | "sim" | "si") {
                    println!("Nothing spent.");
                    return Ok(ExitCode::SUCCESS);
                }
            }
            let m = flow::start(&api, dataset, base).await?;
            let id = m["id"].as_str().unwrap_or("").to_owned();
            if *follow {
                return watch(&api, &id, json).await;
            }
            print(json, &m, || {
                println!("Training {id} (v{}) started. Follow it with `typestrict status {id} --watch`.", m["version"])
            });
        }
        Command::Result { model } => {
            let m = api(&cli)?.get(&format!("/v1/models/{model}")).await?;
            print(json, &m, || result(&m));
        }
        Command::Pull { model, dir } => {
            let api = api(&cli)?;
            let m = api.get(&format!("/v1/models/{model}")).await?;
            let dir = dir.clone().unwrap_or_else(|| {
                PathBuf::from(format!("{}-v{}", m["model"]["name"].as_str().unwrap_or("model"), m["model"]["version"]))
            });
            let files = flow::download(&api, model, &dir).await?;
            print(json, &json!({ "files": files }), || {
                for f in &files {
                    println!("saved {}", f.display());
                }
            });
        }
        Command::Mcp => mcp::serve(api(&cli)?).await?,
    }
    Ok(ExitCode::SUCCESS)
}

fn review(d: &Value) {
    println!("{}", flow::describe(d));
    let s = &d["structure"];
    if s.is_null() {
        return;
    }
    if let Some(sugg) = s["suggestions"].as_array().filter(|a| !a.is_empty()) {
        println!("\nDecisions the agent suggests:");
        for q in sugg {
            println!("  - {}", q["question"].as_str().or(q.as_str()).unwrap_or(""));
        }
    }
    let spec = &s["spec"];
    if let Some(q) = spec["question"].as_str() {
        println!("\nThe model will answer: {q}");
    }
    let per = &s["stats"]["per_label"];
    for l in spec["labels"].as_array().cloned().unwrap_or_default() {
        let name = l["name"].as_str().unwrap_or("");
        println!(
            "  {name:<24} {:>8} examples  {}",
            per[name].as_i64().unwrap_or(0),
            l["description"].as_str().unwrap_or("")
        );
    }
    if let Some(tasks) = spec["tasks"].as_array() {
        println!("\n{} decisions to build:", tasks.len());
        for t in tasks {
            println!("  - {}", t["question"].as_str().unwrap_or(""));
        }
    }
    for w in s["stats"]["warnings"].as_array().cloned().unwrap_or_default() {
        println!("  ! {}", w.as_str().unwrap_or(""));
    }
    if let Some(e) = spec["explanation"].as_str() {
        println!("\n{e}");
    }
}

async fn label(api: &Api, dataset: &str) -> Result<()> {
    let mut after = None;
    loop {
        let next = flow::next_label(api, dataset, after).await?;
        let Some(task) = next["task"].as_object() else {
            println!("Every example is labeled: `typestrict run {dataset}`.");
            return Ok(());
        };
        let labels: Vec<String> = next["spec"]["labels"]
            .as_array()
            .map(|ls| ls.iter().filter_map(|l| l["name"].as_str().map(str::to_owned)).collect())
            .unwrap_or_default();
        println!(
            "\n[{} of {}] {}\n{}",
            next["done"],
            next["total"],
            next["spec"]["question"].as_str().unwrap_or(""),
            task["state"].as_str().unwrap_or("")
        );
        for (i, l) in labels.iter().enumerate() {
            let guess = if task["suggestion"].as_str() == Some(l) { "  (suggested)" } else { "" };
            println!("  {}. {l}{guess}", i + 1);
        }
        let answer = ask("Answer (number or name, s to skip, q to quit): ")?;
        let id = task["id"].as_i64().context("task id")?;
        match answer.as_str() {
            "q" => return Ok(()),
            "s" | "" => after = Some(id),
            a => {
                let chosen = a
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| labels.get(n.wrapping_sub(1)).cloned())
                    .or_else(|| labels.iter().find(|l| l.as_str() == a).cloned());
                match chosen {
                    Some(l) => flow::save_label(api, dataset, id, &l).await?,
                    None => println!("Not an option."),
                }
            }
        }
    }
}

fn result(m: &Value) {
    let model = &m["model"];
    println!("{}", flow::describe(m));
    let metric = model["jev"]["metric"].as_str().unwrap_or("macro_f1");
    if let (Some(o), Some(j)) = (model["metrics"][metric].as_f64(), model["jev"]["scores"][metric].as_f64()) {
        println!("{metric}: this model {o:.3}, Jev {j:.3}, on {} held-out examples", model["holdout_n"]);
    }
    match model["verdict"].as_str() {
        Some("beats") => println!("Verdict: beats Jev."),
        Some(_) => println!("Verdict: did not beat Jev; the credit came back."),
        None => {}
    }
    if let Some(until) = model["files_expire_at"].as_str().filter(|_| model["files_deleted_at"].is_null()) {
        println!("Download with `typestrict pull {}` until {until}.", model["id"].as_str().unwrap_or(""));
    }
}
