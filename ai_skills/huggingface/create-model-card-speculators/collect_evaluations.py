#!/usr/bin/env python3
"""
Category 4 — Evaluations and optimization details for a speculator model.

Extracts only what is already documented in the model's README or saved in a
Speculators AIPerf artifact directory — no results are guessed. When nothing
of a kind is present, the corresponding output value is null and the skill
should prompt the user for the results.

Fields:
  - Acceptance rate tables (per-position acceptance % + average accepted
    length), one entry per dataset
  - Acceptance length tables (k=1..k=N per use case)
  - Use case tables (dataset + sample counts)
  - Performance benchmark details (hardware, sampling config, library
    versions, benchmark commands)
  - Benchmark plot images
  - AIPerf acceptance/category/prompt-length tables, commands, summaries, plots
  - References (papers)

Usage:
  python collect_evaluations.py RedHatAI/Qwen3-8B-speculator.dflash
  python collect_evaluations.py /local/model --output evaluations.json
  python collect_evaluations.py /local/model --eval-dir /path/to/eval-output
"""

import argparse
import csv
import re
from pathlib import Path

from hf_utils import (
    load_repo,
    normalize_html_headings,
    extract_section,
    parse_tables,
    extract_code_blocks,
    md_links_to_text,
    emit,
)


def _pos_header(h):
    m = re.match(r"pos\s*(\d+)$", h.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _k_header(h):
    m = re.match(r"k\s*=\s*(\d+)$", h.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _num(val):
    val = val.strip()
    if val.endswith("%"):
        val = val[:-1].strip()
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except ValueError:
        return val.strip()


PERF_HEADER_RE = re.compile(
    r"throughput|speedup|speed-?up|ttft|ttfb|itl|tokens?/s|tok/s|tok/s|latency|p50|p99|p999|req/s|\brps\b",
    re.IGNORECASE,
)

AIPERF_TABLE_FILES = (
    "acceptance.csv",
    "acceptance_by_prompt_length.csv",
    "acceptance_length_by_category.csv",
    "acceptance_rate_by_category.csv",
)
AIPERF_COMMAND_FILES = ("eval_command.txt", "aiperf_command.txt")
AIPERF_PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _relative(path, root):
    return path.relative_to(root).as_posix()


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _detect_aiperf_modes(commands):
    modes = set()
    for command in commands:
        content = command["content"].lower()
        if re.search(
            r"(?:^|\s)(?:\S*/)?evaluate\.py\s+acceptance(?:\s|$)", content
        ):
            modes.add("acceptance")
        if re.search(
            r"(?:^|\s)(?:\S*/)?evaluate\.py\s+speedup(?:\s|$)", content
        ):
            modes.add("speedup")
    return sorted(modes)


def collect_aiperf_artifacts(directory):
    """Collect model-card inputs from a Speculators AIPerf output tree."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"AIPerf artifact directory does not exist: {root}")

    tables = {}
    for filename in AIPERF_TABLE_FILES:
        entries = [
            {"path": _relative(path, root), "rows": _read_csv(path)}
            for path in sorted(root.rglob(filename))
        ]
        if entries:
            tables[filename] = entries

    commands = []
    for filename in AIPERF_COMMAND_FILES:
        for path in sorted(root.rglob(filename)):
            commands.append(
                {
                    "path": _relative(path, root),
                    "content": path.read_text(encoding="utf-8"),
                }
            )

    plots = sorted(
        _relative(path, root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AIPERF_PLOT_SUFFIXES
    )
    summaries = sorted(
        _relative(path, root)
        for path in root.rglob("profile_export_aiperf.json")
    )
    found_anything = bool(tables or commands or plots or summaries)
    return {
        "root": str(root),
        "modes": _detect_aiperf_modes(commands),
        "tables": tables or None,
        "plots": plots or None,
        "commands": commands or None,
        "native_summaries": summaries or None,
        "found_anything": found_anything,
    }


def _aiperf_datasets(artifact_sets):
    datasets = set()
    for artifacts in artifact_sets:
        tables = artifacts.get("tables") or {}
        for entry in tables.get("acceptance.csv", []):
            for row in entry["rows"]:
                dataset = row.get("dataset")
                if dataset:
                    datasets.add(dataset)
    return datasets


def classify_tables(body):
    acceptance_rates = {}
    acceptance_lengths = {}
    use_cases = []

    for table in parse_tables(body):
        headers = table["headers"]
        pos_idx = [(i, _pos_header(h)) for i, h in enumerate(headers)]
        pos_idx = [(i, p) for i, p in pos_idx if p is not None]
        k_idx = [(i, _k_header(h)) for i, h in enumerate(headers)]
        k_idx = [(i, k) for i, k in k_idx if k is not None]
        low = [h.lower() for h in headers]

        if pos_idx:
            for row in table["rows"]:
                dataset = row[0].strip() if row else ""
                if not dataset:
                    continue
                entry = {"positions": {}, "avg_length": None}
                for i, p in pos_idx:
                    if i < len(row):
                        entry["positions"][str(p)] = _num(row[i])
                for i, h in enumerate(headers):
                    if "avg" in h.lower() and i < len(row):
                        entry["avg_length"] = _num(row[i])
                acceptance_rates[dataset] = entry
            continue

        if k_idx:
            for row in table["rows"]:
                use_case = row[0].strip() if row else ""
                if not use_case:
                    continue
                entry = {}
                for i, k in k_idx:
                    if i < len(row):
                        entry[f"k={k}"] = _num(row[i])
                acceptance_lengths[use_case] = entry
            continue

        if any("use case" in h for h in low) and any("dataset" in h for h in low):
            for row in table["rows"]:
                cells = {
                    h.strip().lower(): (row[i] if i < len(row) else "")
                    for i, h in enumerate(headers)
                }
                use_cases.append(
                    {
                        "use_case": cells.get("use case", ""),
                        "dataset": next(
                            (v for k, v in cells.items() if "dataset" in k), ""
                        ),
                        "samples": next(
                            (
                                _num(v)
                                for k, v in cells.items()
                                if "sample" in k
                            ),
                            None,
                        ),
                    }
                )
    return acceptance_rates, acceptance_lengths, use_cases


def find_performance_tables(body):
    """Tables whose headers look like performance benchmark results."""
    out = []
    for table in parse_tables(body):
        if not PERF_HEADER_RE.search(" ".join(table["headers"])):
            continue
        out.append({"headers": table["headers"], "rows": table["rows"]})
    return out or None


def find_performance(body):
    section = extract_section(body, "Performance") or ""
    config = {}
    for line in section.splitlines():
        m = re.match(r"^\s*[-*]\s*([A-Za-z][\w ]*?)\s*:\s*(.+)$", line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            config[key] = _num(m.group(2))
    hw_m = re.search(r"\((\d+\s*[xX×]\s*[A-Z0-9]+)\)", section[:200])
    if hw_m and not config.get("hardware"):
        config["hardware"] = hw_m.group(1).replace(" ", "")
    commands = [code for lang, code in extract_code_blocks(section) if code.strip()]
    if not config and not commands:
        return None
    return {"config": config or None, "commands": commands or None}


def find_images(body):
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', body)
    imgs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", body)
    seen = set()
    out = []
    for img in imgs:
        if img not in seen:
            seen.add(img)
            out.append(img)
    return out or None


def find_references(body):
    section = extract_section(body, "References")
    if not section:
        return None
    refs = []
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", section):
        refs.append({"title": m.group(1), "url": m.group(2)})
    return refs or None


def main():
    parser = argparse.ArgumentParser(
        description="Collect evaluation details for a speculator model (category 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Local model folder or HuggingFace repo id")
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        help="Speculators AIPerf artifact directory; repeat for multiple runs",
    )
    args = parser.parse_args()

    repo = load_repo(args.source)
    body = normalize_html_headings(repo["readme"])

    acceptance_rates, acceptance_lengths, use_cases = classify_tables(body)
    performance = find_performance(body)
    perf_tables = find_performance_tables(body)
    if perf_tables:
        if performance:
            performance["tables"] = perf_tables
        else:
            performance = {"tables": perf_tables}
    images = find_images(body)
    references = find_references(body)
    aiperf_artifacts = [collect_aiperf_artifacts(path) for path in args.eval_dir]
    aiperf_artifacts = [item for item in aiperf_artifacts if item["found_anything"]]

    has_aiperf_acceptance = any(
        "acceptance.csv" in (item.get("tables") or {})
        or "acceptance_length_by_category.csv" in (item.get("tables") or {})
        or "acceptance_rate_by_category.csv" in (item.get("tables") or {})
        for item in aiperf_artifacts
    )
    has_aiperf_performance = any(
        "speedup" in item["modes"]
        and (item.get("native_summaries") or item.get("plots"))
        for item in aiperf_artifacts
    )

    found_any = any(
        [
            acceptance_rates,
            acceptance_lengths,
            use_cases,
            performance,
            references,
            aiperf_artifacts,
        ]
    )

    if aiperf_artifacts and (acceptance_rates or acceptance_lengths):
        style = "mixed"
    elif aiperf_artifacts:
        style = "aiperf"
    elif acceptance_rates and acceptance_lengths:
        style = "mixed"
    elif acceptance_rates:
        style = "acceptance_rates"
    elif acceptance_lengths:
        style = "acceptance_lengths"
    else:
        style = None

    out = {
        "source": repo["source"],
        "style": style,
        "datasets": sorted(
            set(list(acceptance_rates) + list(acceptance_lengths))
            | _aiperf_datasets(aiperf_artifacts)
        )
        or None,
        "acceptance_rates": acceptance_rates or None,
        "acceptance_lengths": acceptance_lengths or None,
        "use_cases": use_cases or None,
        "performance": performance,
        "images": images,
        "aiperf_artifacts": aiperf_artifacts or None,
        "references": references,
        "missing": [
            k
            for k, v in {
                "acceptance_rates": acceptance_rates
                or acceptance_lengths
                or has_aiperf_acceptance,
                "acceptance_lengths": acceptance_lengths or has_aiperf_acceptance,
                "performance": performance or has_aiperf_performance,
                "references": references,
            }.items()
            if not v
        ],
        "found_anything": found_any,
    }
    emit(out, args.output)


if __name__ == "__main__":
    main()
