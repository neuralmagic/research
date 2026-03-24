# Jira AI skills

This folder holds **agent skills** for **Cursor** and **Claude Code** (and the Claude CLI where applicable) for working with Jira: creating epics, breaking work into stories, and using Atlassian MCP tools in a consistent way.

Each skill is a directory containing a `SKILL.md` file with YAML front matter and step-by-step instructions for the model.

## Skills in this folder

| Skill | Description |
|--------|-------------|
| **create-epic** | Walks through planning and creating a Jira **epic** in conversation: goals, description, LLMs/evaluations when relevant, then assignee, priority, components, and optional defaults (reporter, team). Creates the epic only after explicit user approval via Atlassian MCP. |
| **breakdown-epic** | Helps **decompose an epic into Story issues**: find the epic, pull details and existing children, propose a story-level breakdown, refine with the user, then create each new story (with fields like assignee, priority, components, activity type) linked to the epic—again only after approval per story. |

Paths:

- `create-epic/SKILL.md`
- `breakdown-epic/SKILL.md`

> The skill bodies reference a specific Jira project setup (e.g. INFERENG). If you use another project, update field IDs, project key, and defaults inside each `SKILL.md`.

---

## Atlassian MCP setup

The skills assume Jira is reachable through an **Atlassian-compatible MCP** (tool names in the skills look like `mcp__atlassian__…`).

### Cursor

Cursor reads MCP servers from **`mcp.json`** ([docs](https://cursor.com/docs/context/mcp)):

| Scope | File |
|--------|------|
| Project (shared with the repo) | `.cursor/mcp.json` at the project root |
| Global (all workspaces) | `~/.cursor/mcp.json` |

**Ways to add the server**

1. **Marketplace (recommended)**: **Settings** → **Cursor Settings** → **Marketplace** — find the Atlassian (or Jira) MCP integration and install it from there ([cursor.com/marketplace](https://cursor.com/marketplace) in the browser works too).
2. **Edit `mcp.json`**: merge a server under the top-level `mcpServers` key, then **restart Cursor** so the agent picks up the change.

**Hosted Atlassian MCP (Streamable HTTP / OAuth)** — Cursor connects to the remote endpoint; complete any OAuth sign-in when the agent first uses Jira tools:

```json
{
  "mcpServers": {
    "atlassian": {
      "url": "https://mcp.atlassian.com/v1/mcp"
    }
  }
}
```

**Local `mcp-atlassian` (stdio via `uvx`)** — same pattern as Claude; uses Jira URL + email + API token (no OAuth to Atlassian’s hosted MCP):

```json
{
  "mcpServers": {
    "mcp-atlassian": {
      "command": "uvx",
      "args": ["mcp-atlassian"],
      "env": {
        "JIRA_URL": "https://issues.redhat.com",
        "JIRA_USER_EMAIL": "your-email@redhat.com",
        "JIRA_API_TOKEN": "your-jira-api-token"
      }
    }
  }
}
```

Replace `JIRA_URL`, `JIRA_USER_EMAIL`, and `JIRA_API_TOKEN` with your instance and a [Jira API token](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/) (or your org’s equivalent).

**Troubleshooting**: **View** → **Output** → channel **MCP Logs** for connection errors. Manage or disable Marketplace-installed MCP servers from **Settings** → **Cursor Settings** → **Marketplace**.

---

### Claude Code / Claude CLI

#### Claude CLI: hosted HTTP transport

```bash
claude mcp add atlassian --transport http https://mcp.atlassian.com/v1/mcp --scope user
```

Follow any browser or auth steps the CLI prints so the connection is allowed for your Atlassian account.

#### Claude configuration: `mcp-atlassian` via `uvx`

Add or merge this under your Claude MCP config (location depends on your Claude app / Claude Code version):

```json
{
  "mcpServers": {
    "mcp-atlassian": {
      "command": "uvx",
      "args": ["mcp-atlassian"],
      "env": {
        "JIRA_URL": "https://issues.redhat.com",
        "JIRA_USER_EMAIL": "your-email@redhat.com",
        "JIRA_API_TOKEN": "your-jira-api-token"
      }
    }
  }
}
```

Replace `JIRA_URL`, `JIRA_USER_EMAIL`, and `JIRA_API_TOKEN` with your instance and a [Jira API token](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/) (or your org’s equivalent).

---

## Where to put skill files

Skills must stay in **one folder per skill**, with the main file named **`SKILL.md`**:

```text
skill-name/
└── SKILL.md
```

### In this repository

Skills live under:

```text
ai_skills/jira/<skill-name>/SKILL.md
```

That keeps them versioned and easy to share; it is not always the path your editor reads automatically.

### Cursor

Cursor loads skills from:

```text
.cursor/skills/<skill-name>/SKILL.md
```

Copy or symlink each skill directory (e.g. `create-epic`, `breakdown-epic`) from `ai_skills/jira/` into one of those locations. Do not put custom skills in `~/.cursor/skills-cursor/` (reserved for Cursor’s built-in skills).

### Claude Code

Claude Code loads project skills from:

```text
.claude/skills/<skill-name>/SKILL.md
```

Copy or symlink each skill directory from `ai_skills/jira/` into `.claude/skills/` at the root of the repository (or the Claude Code project you are using).

---

## How to use these skills

1. **Install and verify MCP** using the **Cursor** or **Claude Code / CLI** steps above so Jira tools are available to the agent.
2. **Install the skill** in `.cursor/skills/` (Cursor) or `.claude/skills/` (Claude Code), as described in the previous section.
3. **Invoke explicitly** when you start the task, for example:
   - /create-epic
   - /breakdown-epic