# Jira AI skills

This folder holds **agent skills** for **Cursor** and **Claude Code** (and the Claude CLI where applicable) for working with Jira: creating epics, breaking work into stories, and using the Atlassian CLI (acli) in a consistent way.

Each skill is a directory containing a `SKILL.md` file with YAML front matter and step-by-step instructions for the model.

## Skills in this folder

| Skill | Description |
|--------|-------------|
| **create-epic** | Walks through planning and creating a Jira **epic** in conversation: goals, description, LLMs/evaluations when relevant, then assignee, priority, components, and optional defaults (reporter, team). Creates the epic only after explicit user approval via Atlassian CLI (acli). |
| **breakdown-epic** | Helps **decompose an epic into Story issues**: find the epic, pull details and existing children, propose a story-level breakdown, refine with the user, then create each new story (with fields like assignee, priority, components, activity type) linked to the epic—again only after approval per story. |

Paths:

- `create-epic/SKILL.md`
- `breakdown-epic/SKILL.md`

> The skill bodies reference a specific Jira project setup (e.g. INFERENG). If you use another project, update field IDs, project key, and defaults inside each `SKILL.md`.

---

## Atlassian CLI (acli) setup

The skills assume you have the **Atlassian CLI (acli)** installed and authenticated with your Jira instance.

### Installation and Authentication

Follow the installation and authentication guide here:
**[Atlassian CLI (acli) Setup Guide](https://docs.google.com/document/d/1ilv5NgMS06SK7kS6jXtaELdxE9Nbh_8-nZcEeC9bXPI/)**

The guide covers:
- Installing acli on your system
- Authenticating with your Jira instance (e.g., issues.redhat.com)
- Configuring API tokens and credentials
- Verifying your setup

### Verification

Once installed and authenticated, verify that acli is working:

```bash
acli jira --help
```

You should see the acli Jira command help output. You can also test authentication with:

```bash
acli jira --action getServerInfo
```

This should return information about your Jira instance if authentication is successful.

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

1. **Install and authenticate acli** using the setup guide linked above so that Jira commands are available.
2. **Install the skill** in `.cursor/skills/` (Cursor) or `.claude/skills/` (Claude Code), as described in the previous section.
3. **Invoke explicitly** when you start the task, for example:
   - /create-epic
   - /breakdown-epic