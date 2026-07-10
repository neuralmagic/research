#!/usr/bin/env python3
"""
Compute averaged benchmark scores from evaluation result files.

Supported input formats:
  - summary_data.json  Internal aggregated format (has run_id + results with mean/values/count).
                       Scores are already averaged across seeds; mean is used directly.
  - lm-eval JSON       Top-level "results" key, values are {metric: float} dicts.
                       When multiple files cover the same task, scores are averaged.
  - BFCL CSV          data_overall.csv with Overall/Non-Live/Multi-Turn/... columns.
                       Agentic is computed as avg(Web Search, Memory) when not present.

Usage:
  # Single summary_data.json (handles all tasks in one file)
  python compute_averages.py run_dir/summary_data.json

  # Directory — searched recursively for summary_data.json and results*.json
  python compute_averages.py ./run_dir/

  # Multiple lm-eval JSON files (one per seed) for the same task
  python compute_averages.py seed1.json seed2.json seed3.json

  # BFCL CSV — specify model name substring to extract
  python compute_averages.py score/data_overall.csv --bfcl-model "gemma-4-26B-A4B-it-FP8"

  # Write output to file
  python compute_averages.py run_dir/ --output quantized_scores.json

Output JSON (scores in percentage, 0–100):
  {
    "task:metric": {"mean": 85.93, "n": 3},
    ...
  }
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Format detection ──────────────────────────────────────────────────────────

def detect_json_format(data: dict) -> str:
    """Return 'summary_data', 'lm_eval', or 'unknown'."""
    if "run_id" in data and isinstance(data.get("results"), dict):
        # Check that values look like {metric: {mean, values, count}}
        for task_metrics in data["results"].values():
            if not isinstance(task_metrics, dict):
                break
            for stats in task_metrics.values():
                if isinstance(stats, dict) and "mean" in stats:
                    return "summary_data"
            break
    if isinstance(data.get("results"), dict):
        for task_val in data["results"].values():
            if isinstance(task_val, dict):
                for v in task_val.values():
                    if isinstance(v, (int, float)):
                        return "lm_eval"
            break
    return "unknown"


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_summary_data(data: dict) -> dict:
    """
    Parse internal summary_data.json.
    Returns {task:metric -> (mean_fraction, count)}.
    Already averaged — uses stored mean directly.
    """
    scores = {}
    for task, metrics in data.get("results", {}).items():
        for metric, stats in metrics.items():
            if metric.endswith("_stderr") or not isinstance(stats, dict):
                continue
            if "mean" in stats:
                scores[f"{task}:{metric}"] = (stats["mean"], stats.get("count", 1))
    return scores


def parse_lm_eval(data: dict) -> dict:
    """
    Parse lm-eval output JSON.
    Returns {task:metric -> float} on 0–1 scale.
    """
    scores = {}
    for task, metrics in data.get("results", {}).items():
        for metric, value in metrics.items():
            if metric == "alias":
                continue
            if metric.endswith("_stderr") or not isinstance(value, (int, float)):
                continue
            scores[f"{task}:{metric}"] = value
    return scores


def parse_bfcl_csv(filepath: Path, model_filter: str) -> dict:
    """
    Parse a BFCL data_overall.csv.
    Returns {bfcl:ColumnName -> float} on 0–100 scale for the matched model.
    Agentic = avg(Web Search, Memory) when not already present.
    """
    matched = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("Model") or row.get("model") or "").strip()
            if model_filter.lower() not in model.lower():
                continue
            raw = {}
            for col, val in row.items():
                col = col.strip()
                if col.lower() in ("model", "rank", "notes", ""):
                    continue
                try:
                    raw[col] = float(val)
                except (ValueError, TypeError):
                    pass
            if "Agentic" not in raw:
                web = raw.get("Web Search")
                mem = raw.get("Memory")
                if web is not None and mem is not None:
                    raw["Agentic"] = (web + mem) / 2.0
            matched = {f"bfcl:{k}": v for k, v in raw.items()}
            break  # use first match

    if not matched:
        print(
            f"Warning: no model matching '{model_filter}' found in {filepath}",
            file=sys.stderr,
        )
    return matched


# ── File discovery ─────────────────────────────────────────────────────────────

def find_result_files(path: Path) -> list:
    """Find summary_data.json and results*.json files under path."""
    if path.is_file():
        return [path]
    files = []
    for f in sorted(path.rglob("*.json")):
        if f.name == "summary_data.json" or f.name.startswith("results"):
            files.append(f)
    return files


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute averaged benchmark scores from evaluation result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Result files, directories, or BFCL CSVs")
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--bfcl-model",
        metavar="MODEL",
        default="",
        help="Model name substring to match in BFCL CSV (case-insensitive)",
    )
    args = parser.parse_args()

    # Scores from summary_data (already averaged): {key -> (mean_01, count)}
    final = {}
    # Scores from individual lm-eval files: {key -> [raw_01, ...]}
    accumulated = defaultdict(list)
    # BFCL scores (already 0-100): {key -> float}
    bfcl_scores = {}

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping", file=sys.stderr)
            continue

        # BFCL CSV
        if path.suffix == ".csv":
            if not args.bfcl_model:
                print(
                    f"BFCL CSV provided but --bfcl-model not specified; skipping {path}",
                    file=sys.stderr,
                )
                continue
            bfcl_scores.update(parse_bfcl_csv(path, args.bfcl_model))
            continue

        result_files = find_result_files(path)
        if not result_files:
            print(f"Warning: no result files found under {path}", file=sys.stderr)
            continue

        for filepath in result_files:
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {filepath}: {e}", file=sys.stderr)
                continue

            fmt = detect_json_format(data)
            if fmt == "summary_data":
                for key, (mean_01, count) in parse_summary_data(data).items():
                    if key in final:
                        print(
                            f"Warning: duplicate summary_data entry for '{key}', keeping first",
                            file=sys.stderr,
                        )
                    else:
                        final[key] = (mean_01, count)
            elif fmt == "lm_eval":
                for key, val in parse_lm_eval(data).items():
                    accumulated[key].append(val)
            else:
                print(f"Warning: unknown format in {filepath}, skipping", file=sys.stderr)

    # Build output (all scores in percentage, 0–100)
    output = {}

    for key, (mean_01, n) in sorted(final.items()):
        output[key] = {"mean": round(mean_01 * 100, 2), "n": n}

    for key, values in sorted(accumulated.items()):
        if key in output:
            continue  # summary_data takes precedence
        avg = sum(values) / len(values)
        output[key] = {
            "mean": round(avg * 100, 2),
            "n": len(values),
            "values": [round(v * 100, 2) for v in values],
        }

    for key, val in sorted(bfcl_scores.items()):
        output[key] = {"mean": round(val, 2), "n": 1}

    text = json.dumps(output, indent=2)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Averages written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
