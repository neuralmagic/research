#!/usr/bin/env python3
"""
Category 2 — Creation details (training library, datasets, scripts, commands).

Extracts ONLY what is already documented in the model's README — nothing is
guessed. When a field is missing from the card, the corresponding output
value is null and the skill should prompt the user for it.

Fields:
  - Training library (name + repo url)
  - Datasets used (HF dataset ids, with splits when mentioned)
  - Data preparation notes (e.g. response regeneration)
  - Warm-start source (e.g. warm-started from a DFlash checkpoint)
  - Sponsors / compute providers
  - Training commands: prepare data / launch vLLM / launch training
  - Best-effort training flags parsed from the training command

Usage:
  python collect_creation.py RedHatAI/Qwen3-8B-speculator.dflash
  python collect_creation.py /local/model --output creation.json
"""

import argparse
import re

from hf_utils import (
    load_repo,
    normalize_html_headings,
    strip_code_blocks,
    md_links_to_text,
    parse_cli_flags,
    emit,
)

DATASET_LINK_RE = re.compile(
    r"\[([^\]]*)\]\(\s*(https?://huggingface\.co/datasets/[^)\s]+?)\s*\)"
)
DATASET_BARE_RE = re.compile(r"https?://huggingface\.co/datasets/([\w.-]+/[\w.-]+)")
SPLIT_RE = re.compile(
    r"the\s+([\w-]+)\s+split\s+of\s*\[[^\]]*\]\(\s*(https?://huggingface\.co/datasets/[^)\s]+?)\s*\)",
    re.IGNORECASE,
)
LIBRARY_RE = re.compile(
    r"\[([Ss]peculators)\]\((https?://github\.com/[^)\s]+)\)"
)
LIBRARY_URL_RE = re.compile(r"https?://github\.com/[^)\s\]`]*speculators[^)\s\]`]*")


def _sentences(text):
    text = md_links_to_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) > 15:
            out.append(p)
    return out


def find_datasets(body):
    splits = {}
    for m in SPLIT_RE.finditer(body):
        splits[m.group(2)] = m.group(1)

    found = []
    seen = set()
    for m in DATASET_LINK_RE.finditer(body):
        url = m.group(2)
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        ds_id = key.replace("https://huggingface.co/datasets/", "", 1)
        found.append(
            {
                "id": ds_id,
                "display": m.group(1) or ds_id,
                "url": url,
                "split": splits.get(key),
            }
        )
    for m in DATASET_BARE_RE.finditer(body):
        ds_id = m.group(1)
        key = f"https://huggingface.co/datasets/{ds_id}"
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {"id": ds_id, "display": ds_id, "url": key, "split": splits.get(key)}
        )
    return found


def find_library(body):
    m = LIBRARY_RE.search(body)
    if m:
        return {"name": m.group(1), "url": m.group(2)}
    m = LIBRARY_URL_RE.search(body)
    if m:
        return {"name": "Speculators", "url": m.group(0)}
    return None


def find_notes(body):
    """Extract data-preparation, warm-start and sponsor sentences."""
    plain = strip_code_blocks(body)
    notes = {"data_preparation": None, "warm_start": None, "sponsors": []}
    for s in _sentences(plain):
        low = s.lower()
        if "regenerat" in low and notes["data_preparation"] is None:
            notes["data_preparation"] = s
        if "warm-start" in low and notes["warm_start"] is None:
            notes["warm_start"] = s
        if ("sponsored by" in low or "provided by" in low) and "training compute" in low:
            notes["sponsors"].append(s)
    return notes


def find_commands(body):
    from hf_utils import extract_command_blocks

    blocks = extract_command_blocks(body)
    mapping = {
        "prepare_data": None,
        "launch_vllm": None,
        "launch_training": None,
    }
    for key, code in blocks.items():
        if "prepare" in key and "data" in key:
            mapping["prepare_data"] = code
        elif "vllm" in key and "launch" in key:
            mapping["launch_vllm"] = code
        elif "launch" in key and "train" in key:
            mapping["launch_training"] = code
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Collect creation details for a speculator model (category 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Local model folder or HuggingFace repo id")
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    repo = load_repo(args.source)
    body = normalize_html_headings(repo["readme"])

    datasets = find_datasets(body)
    library = find_library(body)
    notes = find_notes(body)
    commands = find_commands(body)
    train_code = commands.get("launch_training")
    train_flags = parse_cli_flags(train_code) if train_code else None

    out = {
        "source": repo["source"],
        "library": library,
        "datasets": datasets or None,
        **notes,
        "commands": {k: v for k, v in commands.items() if v} or None,
        "training_flags": train_flags,
        "missing": [
            k
            for k, v in {
                "library": library,
                "datasets": datasets,
                "prepare_data": commands.get("prepare_data"),
                "launch_vllm": commands.get("launch_vllm"),
                "launch_training": commands.get("launch_training"),
            }.items()
            if not v
        ],
    }
    emit(out, args.output)


if __name__ == "__main__":
    main()
