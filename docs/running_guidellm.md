# How to run GuideLLMTask

The `GuideLLMTask` class controls the execution of GuideLLM.
The current implementation is focused on performance benchmarking, although in principle the class should support any use of GuideLLM.
The GuideLLM script will initialize a vLLM server, wait for the server to be active, and then start the GuideLLM process.
It supports general ClearML task arguments, GuideLLM-specific parameters, and arguments related to the vLLM server.

## ClearML task arguments
The following arguments configure the ClearML environment for the evaluation task:
- **project_name (required)**: name of ClearML project
- **task_name (required)**: name of ClearML task
- docker_image (optional): path to docker image. Will use default image if one is not provided.
- packages (optional): list of additional Python packages to be installed in the evaluation environment. Syntax similar to a `requirements.txt` file for pip install.

## Model ID
These arguments define the model to be used:
  - **model** (required): Model identifier, which can be:
  - A **Hugging Face** model ID (e.g., `"meta-llama/Llama-3.2-1B-Instruct"`)
  - A **ClearML model ID**
  - A **local file path**
- clearml_model (optional): boolean to indicate whether the model is a ClearML model
- force_download (optional): boolean to indicate whether to force a fresh download of the model (valid for HF models only).

## GuideLLM arguments
The following arguments are passed directly to `GuideLLM`:
- target
- data-type
- data
- max_secons
- etc.

# Environment variables
Some parameters in GuideLLM can be controlled with environment variables.
These variables start with "GUIDELLM__" and can be passed as regular arguments.
Example:
```python
GUIDELLM__MAX_CONCURRENCY=512
```

## vLLM arguments
`vllm_kwargs` is a dictionary containing optional arguments for the vLLM server, such as:  
  - enable_chunked_prefill
  - max_batch_tokens
  - etc

## Benchmarking example

```python
from automation.tasks import GuideLLMTask

task = GuideLLMTask(
    project_name="alexandre_debug",
    task_name="test_guidellm_task",
    model="meta-llama/Llama-3.2-1B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    GUIDELLM__MAX_CONCURRENCY=256,
    GUIDELLM__REQUEST_TIMEOUT=21600,
    target="http://localhost:8000/v1",
    data_type="emulated",
    max_seconds=30,
    data="prompt_tokens=512,generated_tokens=256",
    vllm_kwargs={"enable-chunked-prefill": True}
)

task.execute_remotely("oneshot-a100x1")
```