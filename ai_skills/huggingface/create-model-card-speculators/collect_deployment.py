#!/usr/bin/env python3
"""
Category 3 — Deployment (vLLM serving) for a speculator model.

Extracts ONLY what is already documented in the model's README:
  - vLLM install requirements (e.g. pip install from a PR branch)
  - The `vllm serve` command (raw and one-line)
  - The speculative decoding config (parsed `--speculative-config` JSON, or
    legacy `--spec-model` / `--spec-tokens` / `--spec-method` flags)
  - Model specifications table (base model, chat template, format, license,
    validation hardware)

When the deployment section or serve command is missing, the corresponding
output value is null and the skill should prompt the user for it.

Usage:
  python collect_deployment.py RedHatAI/Qwen3-8B-speculator.dflash
  python collect_deployment.py /local/model --output deployment.json
"""

import argparse
import json
import re

from hf_utils import (
    load_repo,
    normalize_html_headings,
    extract_section,
    parse_tables,
    extract_json_after,
    parse_cli_flags,
    emit,
)


def split_statements(block):
    segments = []
    cur = []
    for line in block.splitlines():
        if line.strip() == "":
            if cur:
                segments.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        segments.append("\n".join(cur))
    return segments


def strip_comments(segment):
    return "\n".join(l for l in segment.splitlines() if not l.strip().startswith("#"))


def to_oneline(code):
    code = re.sub(r"\\\s*\n", " ", code)
    return re.sub(r"\s{2,}", " ", code).strip()


def find_model_specs(body):
    specs = {}
    for table in parse_tables(body):
        headers = table["headers"]
        if not all(h == "" for h in headers):
            continue
        rows = [r for r in table["rows"] if len(r) == 2]
        if not rows:
            continue
        for r in rows:
            key = r[0].replace("*", "").strip()
            if key:
                specs[key] = r[1].strip()
        if specs:
            break
    return specs or None


def main():
    parser = argparse.ArgumentParser(
        description="Collect deployment details for a speculator model (category 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Local model folder or HuggingFace repo id")
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    repo = load_repo(args.source)
    body = normalize_html_headings(repo["readme"])

    section = extract_section(body, "Deployment") or extract_section(body, "Use with vLLM")

    install = []
    serve_raw = None
    spec_config = None
    spec_source = None
    notes = []

    if section:
        lines = section.splitlines()
        in_fence = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                buf = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    buf.append(lines[i])
                    i += 1
                block = "\n".join(l.rstrip() for l in buf)
                for seg in split_statements(block):
                    clean = strip_comments(seg)
                    clean = "\n".join(l.rstrip() for l in clean.splitlines()).strip()
                    if not clean:
                        continue
                    if "vllm serve" in clean and "pip install" in clean:
                        idx = clean.find("vllm serve")
                        line_start = clean.rfind("\n", 0, idx) + 1
                        pre = strip_comments(clean[:line_start]).strip()
                        if pre:
                            install.append(to_oneline(pre))
                        serve_raw = clean[line_start:]
                    elif clean.startswith("pip install"):
                        install.append(to_oneline(clean))
                    elif "vllm serve" in clean:
                        serve_raw = clean
                continue
            if not in_fence and stripped.startswith("#") and stripped != "#":
                notes.append(stripped.lstrip("# "))
            i += 1

    if serve_raw:
        raw_json = extract_json_after(serve_raw, "--speculative-config")
        if raw_json:
            try:
                spec_config = json.loads(raw_json)
                spec_source = "speculative-config"
            except json.JSONDecodeError:
                spec_config = None
        if spec_config is None:
            flags = parse_cli_flags(serve_raw)
            if "spec-model" in flags or "spec-method" in flags:
                spec_config = {
                    "model": flags.get("spec-model"),
                    "num_speculative_tokens": flags.get("spec-tokens"),
                    "method": flags.get("spec-method"),
                }
                spec_source = "legacy-flags"

    out = {
        "source": repo["source"],
        "install": install or None,
        "serve_command": to_oneline(serve_raw) if serve_raw else None,
        "serve_command_raw": serve_raw,
        "speculative_config": spec_config,
        "speculative_config_source": spec_source,
        "model_specs": find_model_specs(body),
        "notes": notes or None,
        "missing": [
            k
            for k, v in {
                "serve_command": serve_raw,
                "speculative_config": spec_config,
            }.items()
            if not v
        ],
    }
    emit(out, args.output)


if __name__ == "__main__":
    main()
