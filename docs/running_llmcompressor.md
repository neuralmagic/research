# How to run LLMCompressorTask

The `LLMCompressorTask` class applies one-shot quantization and optimization algorithms using `llm-compressor`.
In addition to general arguments needed to define ClearML classes, it takes arguments related to calibration dataset and oneshot recipe.

## ClearML task arguments
The following arguments configure the ClearML environment for the compression task:
- **project_name (required)**: name of ClearML project
- **task_name (required)**: name of ClearML task
- docker_image (optional): path to docker image. Will use default image if one is not provided.
- packages (optional): list of additional Python packages to be installed in the evaluation environment. Syntax similar to a `requirements.txt` file for pip install.

## llm-compressor arguments

### Model-loading arguments
The following arguments are used to load the model:
- **model_id** (required): Model identifier, which can be:
  - A **Hugging Face** model ID (e.g., `"meta-llama/Llama-3.2-1B-Instruct"`)
  - A **ClearML model ID**
  - A **local file path**
- clearml_model (optional, boolean): If `True`, indicates that `model_id` refers to a ClearML model rather than a Hugging Face or local model.
- trust_remote_code (optional, boolean): Enables loading custom model architectures from Hugging Face repositories.
- model_class (optional, string): Transformers class used to load model. Defaults to AutoModelForCausalLM.
- tracing_class (optiona, string): Class used to override model definition to enable tracing. Needed when running sequential methods with some multi-modal models. See [tracing guide](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/transformers/tracing/GUIDE.md) for more info.
- max_memory_per_gpu*(optional): Defines how the model is sharded across multiple devices:
  - `None`: Uses `device_map="auto"` (automatic allocation).
  - `"hessian"`: Uses `calculate_offload_device_map` to optimize memory allocation, considering Hessian storage.
  - *(integer)*: Specifies a fixed memory limit (in GB) per device.

### Recipe arguments
The following arguments are used to define the oneshot recipe:
- **recipe (required):** oneshot recipe. One of the following formats is accepted:
  - string in `yaml` format
  - path to recipe file in `yaml` format
  - dictionary
  - list of `Modifier` classes 
- recipe_args (optional): dictionary of recipe variables to be ovewritten

⚠️ **Dynamic Recipe Variables**  
This class extends `llm-compressor` by allowing **runtime variable substitution** in recipes via `recipe_args`.

- Variables are prefixed with `$` in the recipe.
- Their values can be assigned dynamically in `recipe_args`.

**Example:**  
If your recipe contains:
```yaml
dampening_frac: $damp
```
You can define damp dynamically as:
```python
recipe_args={"damp": 0.1}
```

### Dataset arguments
The following arguments are used to define the calibration dataset:
- dataset_name (optional, string): name of dataset. It can be one of the shortcuts registered in this library or a HF dataset
- text_samples (optional, integer): number of text samples
- vision_samples (optional, integer): number of vision samples
- max_seq_len (optional, integer): maximum number of tokens per sample
- dataset_loader (optional, callable): callable used to load the dataset. If not provided default data loader is used.
- data_collator (optiona, callabale): callable used to override data collator.

### Output arguments
- save_directory (optional): path to save output model. Model will be uploaded to ClearML regardless of the path specified here.
- tags (optional): list of tags used to locate model in ClearML UI.


## W4A16 Quantization Example
```python
from automation.tasks import LLMCompressorTask

recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["lm_head"]
      scheme: "W4A16"
      targets: "Linear"
      observer: $observer
      dampening_frac: $damp
"""

task = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="test_llmcompressor_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    recipe=recipe,
    recipe_args={"observer": "mse", "damp": 0.1},
)

task.execute_remotely("oneshot-a100x1")
```
