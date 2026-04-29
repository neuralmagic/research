---
name: generate-results-table
description: Generate benchmark comparison tables from every_eval_ever evaluation results. Use this skill when users want to create accuracy tables, recovery tables, or results tables from directories containing JSON evaluation files. Trigger when users mention "generate results table", "generate accuracy table", "generate recovery table", "compare benchmark results", "create evaluation table", or when they reference every_eval_ever format data and want tabular output.
---

# Generate Results Table

This skill helps you create formatted markdown tables comparing benchmark evaluation results across different models or configurations.

## When to Use This Skill

Use this skill whenever the user wants to:
- Generate a results table from evaluation data
- Compare accuracy scores across models
- Calculate recovery percentages between model variants
- Create markdown tables from every_eval_ever JSON results
- Compare benchmark performance (especially for quantized models vs. base models)

## Input Format

The skill expects directories containing JSON files in the **every_eval_ever** format. Each JSON file should have this structure:

```json
{
  "evaluation_results": [
    {
      "evaluation_name": "benchmark_name",
      "score_details": {
        "score": 0.8733
      }
    }
  ]
}
```

The accuracy score is located at: `evaluation_results[0].score_details.score`

## How to Generate the Table

Use the bundled `scripts/generate_table.py` script to process the evaluation directories and create the markdown table. This script handles:
- Parsing JSON files in the every_eval_ever format
- Extracting accuracy scores
- Matching evaluations across directories
- Calculating recovery percentages
- Formatting the output as a markdown table

### Basic Usage

For a **single directory** (accuracy table only):
```bash
python scripts/generate_table.py /path/to/results/directory -o table.md
```

For **two directories** (with recovery):
```bash
python scripts/generate_table.py /path/to/base/model /path/to/comparison/model -o table.md
```

### Recovery Calculation Rules

- **Two directories**: Recovery is calculated only for benchmarks present in **both** directories
  - Recovery = (comparison_score / base_score) × 100
  - The first directory is treated as the base model, the second as the comparison
- **Single directory**: No recovery column is generated
- **Maximum**: The skill supports up to 2 model directories

### Common Patterns

**Comparing quantized model to base model:**
```bash
# Base model is MiniMax-M2.5-BF16, comparison is MiniMax-M2.5-NVFP4
python scripts/generate_table.py base_model_results/ quantized_model_results/ -o table.md
```

**Downloading results from HuggingFace:**
```bash
# Clone or download the every_eval_ever directories first
git clone https://huggingface.co/inference-optimization/MiniMax-M2.5-BF16
git clone https://huggingface.co/inference-optimization/MiniMax-M2.5-NVFP4

# Then generate the table
python scripts/generate_table.py \
  MiniMax-M2.5-BF16/every_eval_ever \
  MiniMax-M2.5-NVFP4/every_eval_ever \
  -o results_comparison.md
```

**Disabling recovery calculation:**
```bash
python scripts/generate_table.py /path/to/results --no-recovery -o table.md
```

## Output Format

The generated markdown table will look like this:

```markdown
| Benchmark | Base Model | Comparison Model | Recovery (%) |
|-----------|------------|------------------|--------------|
| AIME 2025 | 87.14 | 77.08 | 88.46 |
| GPQA diamond | 83.08 | 80.30 | 96.66 |
| Math 500 | 87.33 | 87.73 | 100.46 |
```

Scores are displayed as percentages (multiplied by 100 from the 0-1 range in the JSON files).

## Important Notes

- The script automatically filters to include only benchmarks present across all specified directories
- If a JSON file is missing in one directory but present in others, that benchmark is excluded from the final table
- Directory names are used as column headers in the table
- Benchmark names are automatically formatted for readability (e.g., "gsm8k_platinum_cot_llama" → "GSM8k Platinum (0-shot)")
- The output file defaults to `table.md` in the current directory if not specified

## Error Handling

If the script encounters issues:
- **No JSON files found**: Check that the directory path is correct and contains `.json` files
- **No common evaluations**: When using multiple directories, ensure they have overlapping benchmark files
- **Missing score data**: Verify the JSON structure matches the every_eval_ever format
- **Permission errors**: Ensure read access to input directories and write access to output location

## Workflow

When a user asks you to generate a results table:

1. **Identify the data source**: Ask where the evaluation results are located (local directories or remote URLs like HuggingFace)

2. **Download if needed**: If results are on HuggingFace or another remote location, download them first using `git clone` or direct file downloads

3. **Run the script**: Use the appropriate command based on how many directories are being compared

4. **Review the output**: Read the generated `table.md` file and present it to the user

5. **Handle edge cases**: If certain benchmarks are missing from some models, explain which ones were excluded and why

## Example Interaction

**User**: "Generate a results table comparing MiniMax-M2.5-BF16 and MiniMax-M2.5-NVFP4 from HuggingFace"

**Your response**:
1. Download both model evaluation directories from HuggingFace
2. Run: `python scripts/generate_table.py MiniMax-M2.5-BF16/every_eval_ever MiniMax-M2.5-NVFP4/every_eval_ever -o table.md`
3. Read and display the generated table
4. Explain the recovery percentages if the user asks

Remember: this skill is designed for the every_eval_ever JSON format specifically. If the user has results in a different format, you may need to adapt the approach or help them convert their data first.
