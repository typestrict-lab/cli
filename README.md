# typestrict CLI

Train a decision model on your data that beats Jev, the market reference model, or the credit comes back. From the
terminal, from scripts, and from AI assistants through MCP.

```bash
curl -fsSL https://typestrict.ai/install.sh | sh
typestrict login                     # paste an API key from https://typestrict.ai/settings#api-keys
```

On Windows, download the zip from the [latest release](https://github.com/typestrict-lab/cli/releases/latest).
The key can also come from `TYPESTRICT_API_KEY`.

## The flow

```bash
typestrict train tickets.csv --goal "Which team handles the ticket?" --watch   # upload, mask, read
typestrict structure <dataset>                     # the agent turns the data into training examples
typestrict review <dataset>                        # its proposal; adjust with: structure <dataset> --feedback "..."
typestrict label <dataset>                         # when asked, answer the held-out examples
typestrict run <dataset> --watch                   # shows the cost and asks before spending the credit
typestrict result <model>                          # the verdict against Jev
typestrict pull <model>                            # decision_ai.json, tokenizer.json, onnx/model_8bit.onnx
```

`typestrict status` lists every training with its step and what it needs next; `status <id> --watch` follows one and
exits with 1 if it failed. `suggest`, `encoders` and `credits` do what they say. Every command takes `--json`.

## For AI assistants

`typestrict mcp` is the typestrict MCP server over stdio: the same flow as tools, for Claude Code, Cursor, Codex,
GitHub Copilot, Windsurf and Gemini CLI. Starting a training always takes a quote first and the user's confirmation.

```bash
claude mcp add typestrict -- typestrict mcp
```

The setup for every assistant: https://docs.typestrict.ai/agents. Skills: https://github.com/typestrict-lab/skills.

## Docs

https://docs.typestrict.ai/cli

## License

Apache-2.0.
