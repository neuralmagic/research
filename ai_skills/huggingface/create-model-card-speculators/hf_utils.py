#!/usr/bin/env python3
"""
Shared helpers for the create-model-card-speculators skill.

Provides utilities to load model files (local folder or HuggingFace repo),
parse README frontmatter / sections / tables, and extract shell commands.

Only the Python standard library is required. PyYAML is used for frontmatter
parsing when available, with a minimal fallback otherwise.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")


def get_token():
    for var in HF_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def fetch_url(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "model-card-speculators-skill/1.0"}
    )
    token = get_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def fetch_optional(url):
    try:
        return fetch_url(url)
    except Exception:
        return None


def _read_optional(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json_optional(path):
    text = _read_optional(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_repo(source):
    """
    Load README.md, config.json and optional API metadata for a model source.

    source: a local directory path, or a HuggingFace repo id
            (e.g. "RedHatAI/Qwen3-8B-speculator.eagle3").

    Returns a dict with keys:
      source, is_local, readme (str), config (dict), card_data (dict),
      api (dict), files (list[str])
    """
    p = Path(str(source))
    if p.exists() and p.is_dir():
        readme = _read_optional(p / "README.md")
        config = _read_json_optional(p / "config.json")
        files = sorted(x.name for x in p.iterdir() if not x.name.startswith("."))
        return {
            "source": str(p),
            "is_local": True,
            "readme": readme or "",
            "config": config or {},
            "card_data": {},
            "api": {},
            "files": files,
        }

    repo_id = str(source).strip().strip("/")
    api = {}
    api_text = fetch_optional(f"https://huggingface.co/api/models/{repo_id}")
    if api_text:
        try:
            api = json.loads(api_text)
        except json.JSONDecodeError:
            api = {}
    readme = fetch_optional(f"https://huggingface.co/{repo_id}/resolve/main/README.md")
    raw_config = fetch_optional(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
    config = {}
    if raw_config:
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError:
            config = {}
    if not config:
        config = api.get("config") or {}
    return {
        "source": repo_id,
        "is_local": False,
        "readme": readme or "",
        "config": config,
        "card_data": api.get("cardData") or {},
        "api": api,
        "files": [
            s.get("rfilename", "") for s in api.get("siblings", []) if s.get("rfilename")
        ],
    }


def parse_frontmatter(text):
    """
    Split a README into (frontmatter_dict, body_str).

    Frontmatter is the YAML between the leading '---' fences. Returns ({}, text)
    when the document has no frontmatter.
    """
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, text
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(fm_text)
        return (data if isinstance(data, dict) else {}), body
    except ImportError:
        return _minimal_yaml(fm_text), body


def _minimal_yaml(text):
    """Parse a small YAML subset: scalars, flat lists, one-level nested maps."""
    data = {}
    current_key = None
    current_mode = None
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0:
            if line.startswith("- "):
                if current_key is None:
                    continue
                if current_mode is None:
                    data[current_key] = []
                    current_mode = "list"
                if isinstance(data.get(current_key), list):
                    data[current_key].append(_scalar(line[2:].strip()))
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val == "":
                    data[key] = None
                    current_key, current_mode = key, None
                else:
                    data[key] = _scalar(val)
                    current_key, current_mode = None, None
            continue
        if line.startswith("- "):
            if current_key is None:
                continue
            if current_mode is None:
                data[current_key] = []
                current_mode = "list"
            if isinstance(data.get(current_key), list):
                data[current_key].append(_scalar(line[2:].strip()))
            continue
        if ":" in line and current_key is not None:
            if current_mode is None:
                data[current_key] = {}
                current_mode = "dict"
            if isinstance(data.get(current_key), dict):
                key, _, val = line.partition(":")
                data[current_key][key.strip()] = _scalar(val.strip())
    return data


def _scalar(val):
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    low = val.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def normalize_html_headings(text):
    """Convert <hN>title</hN> lines to markdown headings."""

    def _repl(m):
        level = int(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        return "#" * level + " " + title

    return re.sub(
        r"<h([1-6])[^>]*>(.*?)</h\1>", _repl, text, flags=re.IGNORECASE | re.DOTALL
    )


def extract_section(body, title):
    """
    Return the markdown body of the section whose heading contains `title`
    (case-insensitive), up to the next heading of the same or higher level.
    Headings inside fenced code blocks are ignored (e.g. shell comments).
    Returns "" when the section is not found.
    """
    lines = body.splitlines()
    result = []
    capturing = False
    cap_level = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if capturing:
                result.append(line)
            continue
        m = _HEADING_RE.match(stripped)
        if m and not in_fence:
            level = len(m.group(1))
            if capturing:
                if level <= cap_level:
                    break
            elif title.strip().lower() in m.group(2).strip().lower():
                capturing = True
                cap_level = level
            continue
        if capturing:
            result.append(line)
    return "\n".join(result).strip()


def parse_md_tables(text):
    """Find all markdown tables. Returns [{"headers": [...], "rows": [[...]]}, ...]."""
    tables = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|") and len(line) > 1 and i + 1 < n:
            sep = lines[i + 1].strip()
            if re.match(r"^\|[\s:|-]+\|$", sep):
                headers = [c.strip() for c in line.strip("|").split("|")]
                rows = []
                j = i + 2
                while j < n and lines[j].strip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    rows.append(cells)
                    j += 1
                tables.append({"headers": headers, "rows": rows})
                i = j
                continue
        i += 1
    return tables


def _clean_cell(s):
    s = re.sub(r"<[^>]+>", "", s)
    for ent, ch in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&nbsp;", " "),
    ):
        s = s.replace(ent, ch)
    return re.sub(r"\s+", " ", s).strip()


def parse_html_tables(text):
    """Find simple HTML tables. Returns the same structure as parse_md_tables."""
    tables = []
    for tm in re.finditer(r"<table[^>]*>(.*?)</table>", text, flags=re.DOTALL | re.IGNORECASE):
        block = tm.group(1)
        header = None
        rows = []
        for tr in re.finditer(r"<tr[^>]*>(.*?)</tr>", block, flags=re.DOTALL | re.IGNORECASE):
            cells = re.findall(
                r"<t([dh])(?:[^>]*)?>(.*?)</t\1>", tr.group(1), flags=re.DOTALL | re.IGNORECASE
            )
            cleaned = [_clean_cell(c[1]) for c in cells]
            if not cleaned:
                continue
            if any(c[0].lower() == "h" for c in cells) and header is None:
                header = cleaned
            else:
                rows.append(cleaned)
        if header is None and rows:
            header, rows = rows[0], rows[1:]
        if header or rows:
            tables.append({"headers": header or [], "rows": rows})
    return tables


def parse_tables(text):
    """All tables in the text, markdown or HTML."""
    return parse_md_tables(text) + parse_html_tables(text)


def table_to_dicts(table):
    """Convert a parsed table into a list of {header: cell} dicts."""
    out = []
    for row in table["rows"]:
        out.append(
            {h: (row[i] if i < len(row) else "") for i, h in enumerate(table["headers"])}
        )
    return out


def extract_code_blocks(text):
    """Return a list of (language, code) tuples for fenced code blocks."""
    blocks = []
    for m in re.finditer(
        r"^```([^\n`]*)\n(.*?)^```[ \t]*$", text, flags=re.MULTILINE | re.DOTALL
    ):
        blocks.append((m.group(1).strip(), m.group(2)))
    return blocks


def extract_command_blocks(text):
    """
    Map sub-heading titles (lowercased) to the code blocks under them.
    Useful for '### Prepare data' / '### Launch vLLM' / '### Launch training'
    style command sections inside <details> blocks.
    """
    lines = text.splitlines()
    blocks = {}
    current = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^#{1,6}\s+(.*)$", line.strip())
        if m:
            current = m.group(1).strip().lower()
            i += 1
            continue
        if line.strip().startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            if current is not None:
                blocks.setdefault(current, []).append("\n".join(buf))
            i += 1
            continue
        i += 1
    return {k: "\n\n".join(v) for k, v in blocks.items()}


def parse_cli_flags(code):
    """
    Best-effort parse of `--flag value` pairs in a shell command.
    Handles line continuations and single/double-quoted values.
    Repeated flags become lists. Only the first following token is taken as
    the value (multi-value flags keep only the first value).
    """
    text = re.sub(r"\\\s*\n", " ", code)
    flags = {}
    pattern = re.compile(r"(?<![\w-])(--[\w-]+)(?:\s+((?:'[^']*'|\"[^\"]*\"|\S+)))?")
    for m in pattern.finditer(text):
        name = m.group(1)[2:]
        val = m.group(2)
        if val is not None:
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if val.startswith("-"):
                val = True
        else:
            val = True
        if name in flags:
            existing = flags[name]
            if isinstance(existing, list):
                existing.append(val)
            else:
                flags[name] = [existing, val]
        else:
            flags[name] = val
    return flags


def extract_json_after(text, marker):
    """Extract the first balanced {...} JSON object appearing after `marker`."""
    idx = text.find(marker)
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def md_links_to_text(s):
    """Strip markdown links, keeping the link text."""
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)


def strip_code_blocks(text):
    """Remove fenced code blocks from markdown text."""
    return re.sub(r"^```[^\n`]*\n.*?^```[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL)


def emit(data, output):
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if output == "-":
        print(text)
    else:
        Path(output).write_text(text + "\n", encoding="utf-8")
        print(f"Written to {output}", file=sys.stderr)
