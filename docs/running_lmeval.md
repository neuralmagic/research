# How to run LMEvalTask

The `LMEvalTask` class allows running evaluations using `lm_eval` within a ClearML task.
It supports both general ClearML task arguments and lm_eval-specific parameters.

## ClearML task arguments
The following arguments configure the ClearML environment for the evaluation task:
- **project_name (required)**: name of ClearML project
- **task_name (required)**: name of ClearML task
- docker_image (optional): path to docker image. Will use default image if one is not provided.
- packages (optional): list of additional Python packages to be installed in the evaluation environment. Syntax similar to a `requirements.txt` file for pip install.

## Model ID
Instead of specifying the model ID in `model_args` (specifically the `pretrained` field), use the following arguments:
- **model_id (required):** Model ID (HF or ClearML) or local path
- clearml_model: boolean to indicate whether the model is a ClearML model

## lm_eval arguments
The following arguments are passed directly to `lm_eval`:
- model_args
- tasks
- num_fewshot
- batch_size
- etc.

⚠️ **Important:**  
The `pretrained` field **must not be included** in `model_args`, as `LMEvalTask` handles model initialization separately.

⚠️ **Note:**  
The `model` argument is not user-configurable. It is **hardcoded to `model=vllm`** in `lmeval_script.py` to ensure compatibility with `vLLM`-based inference.

## Groups

This class supports computing **weighted averages** of multiple metrics.  
For example, this feature can be used to compute an **aggregate OpenLLM or LeaderboardV2 score** instead of displaying the separate metrics individually.

The feature is controlled via the optional `group` argument, which is a nested dictionary.  
- The **top-level keys** represent the names of the aggregate metrics.
- The **lower-level dictionaries** define the individual metrics to be included.
- Each metric entry must include a `series` key, which specifies the name of the metric used in aggregation.

### **Example: OpenLLM Aggregation**
```python
group = {
    "openllm": {
        "arc_challenge": {"series": "acc,none"},
        "gsm8k": {"series": "exact_match,strict-match"},
        "hellaswag": {"series": "acc_norm,none"},
        "mmlu": {"series": "acc,none"},
        "winogrande": {"series": "acc,none"},
        "truthfulqa_mc2": {"series": "acc,none"},
    }
}
```



## GSM8k Example

```python
from automation.tasks import LMEvalTask

task = LMEvalTask(
    project_name="alexandre_debug",
    task_name="test_lmeval_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    tasks="gsm8k",
    model_args="dtype=auto,max_model_len=8192",
    batch_size="auto",
    num_fewshot=5,
)

task.execute_remotely("oneshot-a100x1")
```
