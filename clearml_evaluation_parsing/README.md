# Scripts to Push Results from Evaluation Harnesses into ClearML

## Getting Started

To push results to ClearML, you need to install ClearML and initialize a session locally.

```bash
pip install clearml
clearml-init
```

## lm-evaluation-harness

The `lm_evaluation_harness.py` script allows you to push results obtained using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to ClearML. If you use the `--output_path` argument, `lm_eval` will create a `.json` file containing the results. You can then use the following command to push these results to ClearML:

```bash
python lm_evaluation_harness.py <path_to_json_file> <clearml_project_name> <clearml_task_name>
```

- `<path_to_json_file>`: Path to the `.json` file containing the results.
- `<clearml_project_name>`: Name of the ClearML project where results will be pushed.
- `<clearml_task_name>`: Name of the ClearML task under which results will be stored.

This script will push the results in two formats:
1. It will upload the raw content of the `.json` file as an artifact.
2. It will list the numeric values from the "results" field of the `.json` file as scalar results.


## bigcode-evaluation-harness

The `bigcode_evaluation_harness.py` script allows you to push results obtained using the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to ClearML. If you use the `--metric_output_path` argument, `main.py` will create a `.json` file containing the results. You can then use the following command to push these results to ClearML:

```bash
python bigcode_evaluation_harness.py <path_to_json_file> <clearml_project_name> <clearml_task_name>
```

- `<path_to_json_file>`: Path to the `.json` file containing the results.
- `<clearml_project_name>`: Name of the ClearML project where results will be pushed.
- `<clearml_task_name>`: Name of the ClearML task under which results will be stored.

This script will push the results in two formats:
1. It will upload the raw content of the `.json` file as an artifact.
2. It will list the numeric values for the metrics contained in `.json` file as scalar results. It assums that every field in the root of the `.json` file is a different task, with exception of "configs".

## llm-foundry

The `llm_foundry_gauntlet.py` script allows you to push results obtained using the [llm-foundry](https://github.com/mosaicml/llm-foundry) evaluation gauntlet to ClearML. If one replaces the script `scripts/eval/eval.py` by `eval.py` contained in this folder, the results will be reported in two `.json` files: one for summary (score per category) and one with detailed scores. You can then use the following command to push these results to ClearML:

```bash
python llm_foundry_gauntlet.py <path_to_summary_json_file> <path_to_detailed_json_file> <clearml_project_name> <clearml_task_name>
```

- `<path_to_summary_json_file>`: Path to the `.json` file containing the summary of results.
- `<path_to_detailed_json_file>`: Path to the `.json` file containing detailed results.
- `<clearml_project_name>`: Name of the ClearML project where results will be pushed.
- `<clearml_task_name>`: Name of the ClearML task under which results will be stored.

This script will push the results in three formats:
1. It will upload the raw content of the `.json` files as artifacts.
2. It will list the numeric values for the metrics contained in summary results file as scalar results.
3. It will create a pandas table for each of the categories using the detailed results and push as a ClearML table, which appears under the "PLOTS" tab.