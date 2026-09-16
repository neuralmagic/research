#!/usr/bin/env python3
"""
Category 1 — Model basics for a speculator model.

Extracts (only from the model's own files — nothing is invented):
  - Model architecture: draft architecture (config.json `architectures[0]`)
    and verifier architecture (config.json `speculators_config.verifier`
    -> the verifier's own config.json when not listed)
  - Speculative decoding algorithm (eagle3, dflash, dflash2, dspark, peagle, ...)
  - Verifier (target) model — from config.json `speculators_config.verifier`,
    overridable with --verifier (user-defined)
  - Draft model shape: number of layers + backbone family (llama, qwen3, ...)
    — no parameter estimation
  - Speculators library version and proposal settings
  - README frontmatter (tags, license, base_model)

Usage:
  python collect_basics.py RedHatAI/Qwen3-8B-speculator.dflash
  python collect_basics.py /local/model --verifier Qwen/Qwen3-8B
  python collect_basics.py /local/model --output basics.json
"""

import argparse
import re

from hf_utils import fetch_optional, load_repo, parse_frontmatter, emit

ALGORITHM_DISPLAY = {
    "eagle": "EAGLE",
    "eagle2": "EAGLE-2",
    "eagle3": "EAGLE-3",
    "peagle": "P-EAGLE",
    "dflash": "DFlash",
    "dflash2": "DFlash2",
    "dspark": "DSpark",
}


def detect_algorithm(source_name, config):
    sc = config.get("speculators_config") or {}
    alg = sc.get("algorithm") or config.get("speculators_model_type")
    if not alg:
        m = re.search(r"-speculator\.([a-z0-9]+)$", source_name, re.IGNORECASE)
        if m:
            alg = m.group(1).lower()
    return (alg or "").lower() or None


def display_name(alg):
    if not alg:
        return None
    return ALGORITHM_DISPLAY.get(alg, alg.title())


def resolve_verifier(repo, config, user_verifier):
    """Return (verifier_id, architectures, source_of_truth)."""
    sc = config.get("speculators_config") or {}
    verifier_cfg = sc.get("verifier") or {}
    archs = verifier_cfg.get("architectures") or []

    if user_verifier:
        return user_verifier, archs, "user-provided"

    verifier = verifier_cfg.get("name_or_path")
    if verifier:
        return verifier, archs, "config.json"

    fm, _ = parse_frontmatter(repo["readme"])
    base = fm.get("base_model")
    if isinstance(base, list) and base:
        verifier = base[0]
    elif isinstance(base, str):
        verifier = base
    if verifier:
        return verifier, archs, "frontmatter base_model"

    return None, archs, None


def fetch_verifier_architecture(verifier_id):
    if not verifier_id or "://" in verifier_id or verifier_id.startswith("."):
        return None
    raw = fetch_optional(
        f"https://huggingface.co/{verifier_id}/resolve/main/config.json"
    )
    if not raw:
        return None
    try:
        import json

        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    archs = cfg.get("architectures") or []
    return archs[0] if archs else None


def main():
    parser = argparse.ArgumentParser(
        description="Collect model basics for a speculator model (category 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Local model folder or HuggingFace repo id")
    parser.add_argument(
        "--verifier",
        help="Verifier (target) model id. Defaults to the value in config.json.",
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    repo = load_repo(args.source)
    config = repo["config"]

    model_name = repo["source"].rstrip("/").split("/")[-1]
    alg = detect_algorithm(model_name, config)

    verifier_id, verifier_archs, verifier_source = resolve_verifier(
        repo, config, args.verifier
    )
    verifier_arch = verifier_archs[0] if verifier_archs else None
    verifier_arch_source = "speculator-config" if verifier_arch else None
    if not verifier_arch and verifier_id and not repo["is_local"]:
        verifier_arch = fetch_verifier_architecture(verifier_id)
        if verifier_arch:
            verifier_arch_source = "verifier-config"

    tlc = config.get("transformer_layer_config") or {}
    draft = {
        "num_layers": tlc.get("num_hidden_layers"),
        "backbone": tlc.get("model_type"),
    }

    sc = config.get("speculators_config") or {}
    fm, _body = parse_frontmatter(repo["readme"])
    if not fm and repo["card_data"]:
        fm = repo["card_data"]

    out = {
        "source": repo["source"],
        "model_name": model_name,
        "created_at": (repo["api"] or {}).get("createdAt"),
        "model_architecture": {
            "draft": (config.get("architectures") or [None])[0],
            "verifier": verifier_arch,
            "verifier_resolution": verifier_arch_source or "unresolved",
        },
        "algorithm": {
            "id": alg,
            "display": display_name(alg),
        },
        "verifier": {
            "name_or_path": verifier_id,
            "architectures": verifier_archs,
            "source": verifier_source or "unresolved",
        },
        "draft_model": draft,
        "speculators_version": config.get("speculators_version"),
        "proposal": {
            "default_method": sc.get("default_proposal_method"),
            "methods": sc.get("proposal_methods") or [],
        },
        "frontmatter": fm,
        "files": repo["files"],
    }
    emit(out, args.output)


if __name__ == "__main__":
    main()
