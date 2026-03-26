# Hugging Face AI skills

This folder holds **agent skills** for **Cursor** and **Claude Code** (and the Claude CLI where applicable) for working with the Hugging Face Hub: creating model cards, publishing quantized models, and using the Hugging Face MCP tools in a consistent way.

Each skill is a directory containing a `SKILL.md` file with YAML front matter and step-by-step instructions for the model. Some skills include additional reference files (templates, evaluation protocols, etc.) in the same directory.

## Skills in this folder

| Skill | Description |
|--------|-------------|
| **create-model-card-compression** | Interactive, conversational workflow to produce a HuggingFace model card for a RedHatAI quantized model. Walks through gathering model info (config, recipe, base model), collecting evaluation results, filling a standardized template, iterating on the draft, and optionally uploading to the Hub. Supports INT W8A8, INT W4A16, FP8, NVFP4, and mixed-precision formats. |

Paths:

- `create-model-card-compression/SKILL.md` — main skill
- `create-model-card-compression/template.md` — model card template with placeholders
- `create-model-card-compression/recipe-parsing.md` — reference for detecting quantization format from `recipe.yaml`
- `create-model-card-compression/evaluations.md` — evaluation protocol (lm-eval, lighteval, standard benchmarks)

---

## Hugging Face MCP setup

The skills assume the Hugging Face Hub is reachable through the **Hugging Face MCP Server**. The server provides built-in tools for model/dataset/Space/paper search, documentation lookup, and Hub repository operations.

You can manage which tools and Spaces are available from your [MCP settings](https://huggingface.co/settings/mcp).

### Cursor

Cursor reads MCP servers from **`mcp.json`** ([docs](https://cursor.com/docs/context/mcp)):

| Scope | File |
|--------|------|
| Project (shared with the repo) | `.cursor/mcp.json` at the project root |
| Global (all workspaces) | `~/.cursor/mcp.json` |

**Ways to add the server**

1. **Marketplace (recommended)**: **Settings** → **Cursor Settings** → **Marketplace** — find the **Hugging Face** MCP integration and install it from there ([cursor.com/marketplace](https://cursor.com/marketplace) in the browser works too).
2. **Edit `mcp.json`**: merge a server under the top-level `mcpServers` key, then **restart Cursor** so the agent picks up the change.

**Hosted Hugging Face MCP (Streamable HTTP)** — Cursor connects to the remote endpoint; authenticate with a [Hugging Face token](https://huggingface.co/settings/tokens):

```json
{
  "mcpServers": {
    "huggingface": {
      "url": "https://huggingface.co/mcp",
      "headers": {
        "Authorization": "Bearer <YOUR_HF_TOKEN>"
      }
    }
  }
}
```

After installation, visit https://huggingface.co/settings/mcp to enable or disable individual tools and community Spaces.

**Troubleshooting**: **View** → **Output** → channel **MCP Logs** for connection errors. Manage or disable Marketplace-installed MCP servers from **Settings** → **Cursor Settings** → **Marketplace**.

---

### Claude Code / Claude CLI

#### Claude CLI: hosted HTTP transport

```bash
claude mcp add hf-mcp-server -t http https://huggingface.co/mcp?login
```

Start Claude and follow the authentication instructions printed by the CLI.

Alternatively, pass your [Hugging Face token](https://huggingface.co/settings/tokens) directly:

```bash
claude mcp add hf-mcp-server \
  -t http https://huggingface.co/mcp \
  -H "Authorization: Bearer <YOUR_HF_TOKEN>"
```

---

## Where to put skill files

Skills must stay in **one folder per skill**, with the main file named **`SKILL.md`**:

```text
skill-name/
├── SKILL.md
└── (optional reference files)
```

### In this repository

Skills live under:

```text
ai_skills/huggingface/<skill-name>/SKILL.md
```

That keeps them versioned and easy to share; it is not always the path your editor reads automatically.

### Cursor

Cursor loads skills from:

```text
.cursor/skills/<skill-name>/SKILL.md
```

Copy or symlink each skill directory (e.g. `create-model-card-compression`) from `ai_skills/huggingface/` into one of those locations. Do not put custom skills in `~/.cursor/skills-cursor/` (reserved for Cursor's built-in skills).

### Claude Code

Claude Code loads project skills from:

```text
.claude/skills/<skill-name>/SKILL.md
```

Copy or symlink each skill directory from `ai_skills/huggingface/` into `.claude/skills/` at the root of the repository (or the Claude Code project you are using).

---

## How to use these skills

1. **Install and verify MCP** using the **Cursor** or **Claude Code / CLI** steps above so Hugging Face tools are available to the agent.
2. **Install the skill** in `.cursor/skills/` (Cursor) or `.claude/skills/` (Claude Code), as described in the previous section.
3. **Invoke explicitly** when you start the task, for example:
   - "Create a model card for my quantized model at `/path/to/model`"
   - "Generate a README for `RedHatAI/Llama-3-8B-Instruct-FP8-dynamic`"
