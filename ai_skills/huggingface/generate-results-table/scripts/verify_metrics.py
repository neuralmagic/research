#!/usr/bin/env python3
"""
Verification script to ensure correct metrics are being extracted from benchmark files.
Run this script to verify that each benchmark is using the correct evaluation_name and evaluation_description.
"""

import json
from pathlib import Path
import sys

# Define the expected metrics for each benchmark
EXPECTED_METRICS = {
    'gsm8k_platinum_cot_llama': {
        'evaluation_name': 'gsm8k_platinum_cot_llama/strict-match',
        'evaluation_description': 'exact_match (filter: strict-match)',
    },
    'mmlu_pro_chat': {
        'evaluation_name': 'mmlu_pro_chat/custom-extract',
        'evaluation_description': 'exact_match (filter: custom-extract)',
    },
    'ifeval': {
        'evaluation_name': 'ifeval',
        'evaluation_description': 'inst_level_strict_acc',
    },
    'aime25': {
        'evaluation_name': 'aime25',
        'evaluation_description': 'avg@n:n=1',
    },
    'gpqa_diamond': {
        'evaluation_name': 'gpqa:diamond',
        'evaluation_description': 'gpqa_pass@k:k=1',
    },
    'math_500': {
        'evaluation_name': 'math_500',
        'evaluation_description': 'pass@k:k=1&n=1',
    },
}


def verify_directory(directory_path: Path) -> bool:
    """
    Verify that all benchmarks in a directory use the correct metrics.

    Args:
        directory_path: Path to directory containing evaluation JSON files

    Returns:
        True if all verifications pass, False otherwise
    """
    if not directory_path.exists():
        print(f"❌ Directory not found: {directory_path}")
        return False

    print(f"\nVerifying metrics in: {directory_path}")
    print("=" * 80)

    all_passed = True

    for benchmark_name, expected in EXPECTED_METRICS.items():
        json_file = directory_path / f"{benchmark_name}.json"

        if not json_file.exists():
            print(f"⚠️  {benchmark_name}: File not found (skipping)")
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            eval_results = data.get('evaluation_results', [])

            # Search for the matching metric
            found = False
            for result in eval_results:
                eval_name = result.get('evaluation_name', '')
                metric_config = result.get('metric_config', {})
                eval_desc = metric_config.get('evaluation_description', '')

                if (eval_name == expected['evaluation_name'] and
                    eval_desc == expected['evaluation_description']):
                    score = result.get('score_details', {}).get('score')
                    if score is not None:
                        print(f"✅ {benchmark_name:30} {score:.4f} ({score*100:.2f}%)")
                        found = True
                        break

            if not found:
                print(f"❌ {benchmark_name:30} Metric not found!")
                print(f"   Expected: {expected['evaluation_name']} / {expected['evaluation_description']}")
                print(f"   Available metrics in file:")
                for result in eval_results[:3]:  # Show first 3 results
                    metric_config = result.get('metric_config', {})
                    print(f"     - {result.get('evaluation_name')} / {metric_config.get('evaluation_description')}")
                all_passed = False

        except Exception as e:
            print(f"❌ {benchmark_name:30} Error: {e}")
            all_passed = False

    return all_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify that benchmark metrics are correctly extracted"
    )
    parser.add_argument(
        'directories',
        nargs='+',
        type=Path,
        help='One or more directories containing evaluation JSON files'
    )

    args = parser.parse_args()

    all_passed = True
    for directory in args.directories:
        if not verify_directory(directory):
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All verifications PASSED")
        return 0
    else:
        print("❌ Some verifications FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
