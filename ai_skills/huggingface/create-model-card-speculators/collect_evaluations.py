#!/usr/bin/env python3
"""
Category 4 — Evaluations and optimization details for a speculator model.

Extracts ONLY what is already documented in the model's README — no results
are guessed. When nothing of a kind is present, the corresponding output
value is null and the skill should prompt the user for the results.

Fields:
  - Acceptance rate tables (per-position acceptance % + average accepted
    length), one entry per dataset
  - Acceptance length tables (k=1..k=N per use case)
  - Use case tables (dataset + sample counts)
  - Performance benchmark details (hardware, sampling config, library
    versions, benchmark commands)
  - Benchmark plot images
  - References (papers)

Usage:
  python collect_evaluations.py RedHatAI/Qwen3-8B-speculator.dflash
  python collect_evaluations.py /local/model --output evaluations.json
"""

import argparse
import re

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

    found_any = any(
        [
            acceptance_rates,
            acceptance_lengths,
            use_cases,
            performance,
            references,
        ]
    )

    if acceptance_rates and acceptance_lengths:
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
        "datasets": sorted(set(list(acceptance_rates) + list(acceptance_lengths)))
        or None,
        "acceptance_rates": acceptance_rates or None,
        "acceptance_lengths": acceptance_lengths or None,
        "use_cases": use_cases or None,
        "performance": performance,
        "images": images,
        "references": references,
        "missing": [
            k
            for k, v in {
                "acceptance_rates": acceptance_rates,
                "acceptance_lengths": acceptance_lengths,
                "performance": performance,
                "references": references,
            }.items()
            if not v
        ],
        "found_anything": found_any,
    }
    emit(out, args.output)


if __name__ == "__main__":
    main()
