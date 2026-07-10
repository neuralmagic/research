#!/usr/bin/env python3
"""
Compute recovery percentages between baseline and quantized model scores.

Both inputs must be JSON files produced by compute_averages.py:
  {"task:metric": {"mean": <float_pct_0_to_100>, "n": <int>}, ...}

Usage:
  python compute_recovery.py baseline_scores.json quantized_scores.json
  python compute_recovery.py baseline_scores.json quantized_scores.json --output recovery.json

Output JSON:
  {
    "task:metric": {
      "baseline": 85.93,
      "quantized": 84.10,
      "recovery_pct": 97.87
    },
    ...
  }

Notes:
  - Keys present only in the quantized file are included with baseline=null, recovery_pct=null.
  - Keys present only in the baseline file are included with quantized=null, recovery_pct=null.
  - If baseline is 0 or null, recovery_pct is set to null (shown as N/A in the model card table).
"""

import argparse
import json
import sys
from pathlib import Path


def load_averages(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Compute recovery percentages from compute_averages.py output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("baseline", help="Baseline averages JSON (from compute_averages.py)")
    parser.add_argument("quantized", help="Quantized averages JSON (from compute_averages.py)")
    parser.add_argument("--output", "-o", default="-", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    baseline = load_averages(Path(args.baseline))
    quantized = load_averages(Path(args.quantized))

    all_keys = sorted(set(baseline) | set(quantized))
    output = {}

    for key in all_keys:
        b_entry = baseline.get(key)
        q_entry = quantized.get(key)

        b_mean = b_entry["mean"] if b_entry else None
        q_mean = q_entry["mean"] if q_entry else None

        if b_mean is not None and q_mean is not None and b_mean != 0:
            recovery = round(q_mean / b_mean * 100, 2)
        else:
            recovery = None

        output[key] = {
            "baseline": b_mean,
            "quantized": q_mean,
            "recovery_pct": recovery,
        }

    text = json.dumps(output, indent=2)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Recovery written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
