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
