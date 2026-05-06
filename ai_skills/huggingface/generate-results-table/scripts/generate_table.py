#!/usr/bin/env python3
"""
Generate benchmark comparison tables from every_eval_ever JSON results.

Reads JSON files containing evaluation results and produces a markdown table
showing accuracies and optional recovery percentages.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_eval_file(filepath: Path) -> Optional[Tuple[str, float]]:
    """
    Parse an every_eval JSON file and extract the evaluation name and score.

    Args:
        filepath: Path to the JSON file

    Returns:
        Tuple of (evaluation_name, score) or None if parsing fails
    """
    # Define benchmark-specific evaluation criteria
    # Maps filename stem to (expected evaluation_name, expected evaluation_description)
    BENCHMARK_METRICS = {
        'gsm8k_platinum_cot_llama': ('gsm8k_platinum_cot_llama/strict-match', 'exact_match (filter: strict-match)'),
        'mmlu_pro_chat': ('mmlu_pro_chat/custom-extract', 'exact_match (filter: custom-extract)'),
        'ifeval': ('ifeval', 'inst_level_strict_acc'),
        'aime25': ('aime25', 'avg@n:n=1'),
        'gpqa_diamond': ('gpqa:diamond', 'gpqa_pass@k:k=1'),
        'math_500': ('math_500', 'pass@k:k=1&n=1'),
    }

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract from evaluation_results
        eval_results = data.get('evaluation_results', [])
        if not eval_results:
            print(f"Warning: No evaluation_results in {filepath.name}")
            return None

        # Get the expected evaluation_name and description for this benchmark
        benchmark_key = filepath.stem
        if benchmark_key in BENCHMARK_METRICS:
            expected_name, expected_description = BENCHMARK_METRICS[benchmark_key]

            # Search through all evaluation results to find the matching one
            for result in eval_results:
                eval_name = result.get('evaluation_name', '')
                metric_config = result.get('metric_config', {})
                eval_description = metric_config.get('evaluation_description', '')

                # Check if both the evaluation_name and description match
                if eval_name == expected_name and eval_description == expected_description:
                    score = result.get('score_details', {}).get('score')
                    if score is not None:
                        return (eval_name, score)
                    else:
                        print(f"Warning: Found matching evaluation but no score in {filepath.name}")
                        return None

            # If we didn't find a match, warn the user
            print(f"Warning: Could not find evaluation with name '{expected_name}' and description '{expected_description}' in {filepath.name}")
            return None
        else:
            # For unrecognized benchmarks, use the first result as fallback
            target_result = eval_results[0]
            eval_name = target_result.get('evaluation_name', filepath.stem)
            score = target_result.get('score_details', {}).get('score')

            if score is None:
                print(f"Warning: No score found in {filepath.name}")
                return None

            return (eval_name, score)

    except Exception as e:
        print(f"Error parsing {filepath.name}: {e}")
        return None


def load_directory_results(directory: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Load all JSON files from a directory and extract their scores and model name.

    Args:
        directory: Path to directory containing JSON files

    Returns:
        Tuple of (model_name, results_dict) where model_name is extracted from model_info
    """
    results = {}
    model_name = None
    json_files = list(directory.glob('*.json'))

    if not json_files:
        print(f"Warning: No JSON files found in {directory}")
        return (None, results)

    for json_file in json_files:
        parsed = parse_eval_file(json_file)
        if parsed:
            eval_name, score = parsed
            results[eval_name] = score

        # Extract model name from the first JSON file
        if model_name is None:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    model_info = data.get('model_info', {})
                    model_name = model_info.get('name') or model_info.get('id')
            except Exception:
                pass

    return (model_name, results)


def verify_metrics(directories: List[Path]) -> bool:
    """
    Verify that all benchmarks use the expected metrics.

    Args:
        directories: List of directories to verify

    Returns:
        True if all verifications pass, False otherwise
    """
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

    print("\n" + "=" * 80)
    print("VERIFICATION: Checking that correct metrics are being used")
    print("=" * 80)

    all_passed = True

    for directory in directories:
        print(f"\nDirectory: {directory}")
        for benchmark_name, expected in EXPECTED_METRICS.items():
            json_file = directory / f"{benchmark_name}.json"

            if not json_file.exists():
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
                            print(f"  ✅ {benchmark_name:30} {score:.4f} ({score*100:.2f}%)")
                            found = True
                            break

                if not found:
                    print(f"  ❌ {benchmark_name:30} Metric not found!")
                    all_passed = False

            except Exception as e:
                print(f"  ❌ {benchmark_name:30} Error: {e}")
                all_passed = False

    print("=" * 80)
    if all_passed:
        print("✅ VERIFICATION PASSED: All metrics are correct")
    else:
        print("❌ VERIFICATION FAILED: Some metrics are incorrect")
    print("=" * 80 + "\n")

    return all_passed


def format_benchmark_name(eval_name: str) -> str:
    """
    Convert evaluation name to a more readable benchmark name.

    Args:
        eval_name: Raw evaluation name from JSON

    Returns:
        Formatted benchmark name
    """
    # Simple formatting - can be enhanced based on patterns
    name_map = {
        'gsm8k_platinum_cot_llama/strict-match': 'GSM8k Platinum (0-shot)',
        'mmlu_pro_chat': 'MMLU Pro Chat',
        'mmlu_pro_chat/custom-extract': 'MMLU Pro Chat',
        'ifeval': 'IfEval (0-shot)',
        'aime25': 'AIME 2025',
        'gpqa:diamond': 'GPQA diamond',
        'math_500': 'Math 500',
    }

    return name_map.get(eval_name, eval_name.replace('_', ' ').title())


def calculate_recovery(base_score: float, comparison_score: float) -> float:
    """
    Calculate recovery percentage.

    Args:
        base_score: Score from base model
        comparison_score: Score from comparison model

    Returns:
        Recovery percentage
    """
    if base_score == 0:
        return 0.0
    return (comparison_score / base_score) * 100


def generate_markdown_table(
    directory_results: List[Tuple[str, Dict[str, float]]],
    output_path: Path,
    include_recovery: bool = True
) -> None:
    """
    Generate a markdown table comparing results across directories.

    Args:
        directory_results: List of (directory_name, results_dict) tuples
        output_path: Where to write the markdown table
        include_recovery: Whether to include recovery column
    """
    if not directory_results:
        print("Error: No results to generate table from")
        return

    # Determine which evaluations to include
    if len(directory_results) == 1:
        # Single directory - just show accuracies
        all_evals = set(directory_results[0][1].keys())
        include_recovery = False
    else:
        # Two directories - only include evals present in both
        all_evals = set(directory_results[0][1].keys()) & set(directory_results[1][1].keys())

    if not all_evals:
        print("Error: No common evaluations found across directories")
        return

    # Define preferred order for benchmarks
    benchmark_order = [
        'gsm8k_platinum_cot_llama/strict-match',
        'mmlu_pro_chat',
        'ifeval',
        'aime25',
        'gpqa:diamond',
        'math_500'
    ]

    # Sort evaluations according to preferred order, with unrecognized ones at the end
    def sort_key(eval_name):
        try:
            return benchmark_order.index(eval_name)
        except ValueError:
            return len(benchmark_order)  # Put unrecognized items at the end

    sorted_evals = sorted(all_evals, key=sort_key)

    # Build the table
    lines = []

    # Header row
    header = ["Benchmark"]
    for dir_name, _ in directory_results:
        header.append(dir_name)
    if include_recovery and len(directory_results) >= 2:
        header.append("Recovery (%)")
    lines.append("| " + " | ".join(header) + " |")

    # Separator row
    lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|")

    # Data rows
    for eval_name in sorted_evals:
        row = [format_benchmark_name(eval_name)]

        base_score = None
        comparison_score = None

        for idx, (_, results) in enumerate(directory_results):
            score = results.get(eval_name)
            if score is not None:
                row.append(f"{score * 100:.2f}")
                if idx == 0:
                    base_score = score
                elif idx == 1:
                    comparison_score = score
            else:
                row.append("")

        # Add recovery column if applicable
        if include_recovery and len(directory_results) >= 2 and base_score and comparison_score:
            recovery = calculate_recovery(base_score, comparison_score)
            row.append(f"{recovery:.2f}")
        elif include_recovery and len(directory_results) >= 2:
            row.append("")

        lines.append("| " + " | ".join(row) + " |")

    # Write to file
    output_path.write_text('\n'.join(lines) + '\n')
    print(f"Table written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison table from every_eval_ever results"
    )
    parser.add_argument(
        'directories',
        nargs='+',
        type=Path,
        help='One or two directories containing JSON result files (max 2)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('table.md'),
        help='Output markdown file (default: table.md)'
    )
    parser.add_argument(
        '--no-recovery',
        action='store_true',
        help='Disable recovery calculation'
    )
    parser.add_argument(
        '--names',
        nargs='+',
        type=str,
        help='Custom names for the model columns (must match number of directories)'
    )

    args = parser.parse_args()

    # Validate number of directories
    if len(args.directories) > 2:
        print("Error: Maximum of 2 directories supported")
        return 1

    # Validate directories
    for directory in args.directories:
        if not directory.exists():
            print(f"Error: Directory does not exist: {directory}")
            return 1
        if not directory.is_dir():
            print(f"Error: Not a directory: {directory}")
            return 1

    # Validate custom names if provided
    if args.names and len(args.names) != len(args.directories):
        print(f"Error: Number of names ({len(args.names)}) must match number of directories ({len(args.directories)})")
        return 1

    # Load results from each directory
    directory_results = []
    for idx, directory in enumerate(args.directories):
        model_name, results = load_directory_results(directory)
        if results:
            # Use custom name if provided, otherwise use model name from JSON, otherwise use directory name
            if args.names:
                name = args.names[idx]
            elif model_name:
                name = model_name
            else:
                name = directory.name
            directory_results.append((name, results))
        else:
            print(f"Warning: No valid results found in {directory}")

    if not directory_results:
        print("Error: No valid results found in any directory")
        return 1

    # Generate the table
    include_recovery = not args.no_recovery and len(directory_results) >= 2
    generate_markdown_table(directory_results, args.output, include_recovery)

    # Run verification
    verification_passed = verify_metrics(args.directories)

    if not verification_passed:
        print("\n⚠️  Warning: Verification found issues. Please review the metrics above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
