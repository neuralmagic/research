---
name: mxfp4_compress
description: >
  Generate a working MXFP4 quantization example script, set up an environment, install dependencies,
  run quantization with canhazgpu, and verify the compressed model with vLLM.
  Triggers on: "mxfp4", "MXFP4", "mx fp4", "fp4 amd", "mi300x quantization".
allowed-tools: [Read, Write, Glob, Shell, AskQuestion, WebFetch]
---

# MXFP4 Quantization — End-to-End Workflow

Generate a working MXFP4 quantization script, create an isolated environment, run quantization on GPUs, and verify the result with vLLM.

Hardware requirement: AMD MI300X for inference. Quantization can run on any GPU.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Algorithm** — choose one:
   - `QuantizationModifier` — simple PTQ, no calibration data required. Recommended default.
   - `GPTQModifier` — learned weight rounding, requires calibration data. Better accuracy.
3. **Model type** — dense, MoE, or multimodal (vision/audio)
4. **Existing virtual environment** — Ask: "Do you have an existing virtual environment you'd like to use? If so, provide the path (e.g. `/data/user/myenv`). Otherwise a new one will be created."
5. **Additional dependencies** — Ask: "Do you need any additional pip packages beyond `llmcompressor`, `vllm`, and `torchvision`?"
6. **HF repo name (optional)** — Ask: "Do you want to upload the quantized model to a Hugging Face repo? If so, provide the repo name (e.g. `RedHatAI/Qwen3-8B-MXFP4`). Leave blank to skip upload."

## Step 2 — Determine GPU requirements

Check available GPUs:
```bash
nvidia-smi -L 2>/dev/null | wc -l
```

Determine how many GPUs are needed based on model size:
- **Up to 8B parameters** — 1 GPU
- **8B–30B parameters** — 2 GPUs
- **30B–70B parameters** — 4 GPUs
- **70B+ parameters** — 8 GPUs

If the required number of GPUs exceeds what is available, warn the user and ask how to proceed (e.g. reduce model size, or wait for resources).

Set `TENSOR_PARALLEL_SIZE` to the number of GPUs needed (used later for vLLM verification).

## Step 3 — Set up virtual environment and install dependencies

### If the user provided an existing virtual environment path:

Activate it directly:
```bash
source <location>/bin/activate
```

### If no existing environment was provided:

Create a new isolated environment using `uv`:
```bash
uv venv compress_model --python 3.12
source compress_model/bin/activate
```

### Install dependencies

Always install `llmcompressor` from the git source:
```bash
uv pip install git+https://github.com/vllm-project/llm-compressor.git
uv pip install vllm torchvision
```

If the user specified additional dependencies, install them too:
```bash
uv pip install <additional_packages>
```

## Step 4 — Check for existing examples on GitHub

Before writing a script from scratch, check the upstream examples directory for an existing script that targets the same model architecture:

**GitHub directory:** `https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w4a4_mxfp4`

1. Fetch the directory listing using `WebFetch` or browse the GitHub URL to see available example files.
2. Identify if an example exists for the same model family / architecture as the user's MODEL_ID. For example:
   - User wants to quantize `meta-llama/Llama-4-Scout-17B-16E-Instruct` → look for `llama4_example.py`
   - User wants to quantize `Qwen/Qwen3-32B` → look for `qwen3_example.py`
   - User wants to quantize `google/gemma-3-27b-it` → look for `gemma3_example.py`
3. **If a matching example is found:**
   - Download the raw file from GitHub (e.g. `https://raw.githubusercontent.com/vllm-project/llm-compressor/main/examples/quantization_w4a4_mxfp4/<filename>.py`)
   - Replace the `MODEL_ID` value in the script with the user's MODEL_ID
   - Adjust the `SAVE_DIR` if needed
   - Use this as the quantization script — skip Step 5 template generation
4. **If no matching example is found:**
   - Proceed to Step 5 and generate the script from the template

## Step 5 — Choose the quantization template and write the script (fallback)

### QuantizationModifier (no calibration — standard path)

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],  # extend per model type — see Step 6
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### GPTQModifier (with calibration — better accuracy)

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "<MODEL_ID>"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],  # extend per model type — see Step 6
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Place the file under `examples/quantization_w4a4_mxfp4/`.

Name the file `{model_name_slug}_example.py` (e.g. `llama3_mxfp4.py`, `gemma4_mxfp4.py`).

## Step 6 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient in most cases.

### MoE models
Add gate/router layers to `ignore`. For models that need a custom loading path, wrap the load in `load_context`:

```python
from llmcompressor.utils import load_context

with load_context(SpecificModelClass):
    model = SpecificModelClass.from_pretrained(MODEL_ID)
```

Common MoE ignore additions:
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`
- Llama4 / Gemma4 MoE: `"re:.*router"`, `"Llama4TextAttention"`
- General: `"re:.*mlp.router.*"`

For MXFP4 on MoE, also consider skipping attention: `"re:.*self_attn"`.

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Use the appropriate model class (e.g. `Gemma4ForConditionalGeneration`, `Llama4ForConditionalGeneration`) and wrap load in `load_context` if needed
- Add to `ignore`:
  - Vision: `"re:.*vision_tower.*"`, `"re:.*vision_model.*"`, `"re:.*multi_modal_projector.*"`
  - Audio: `"re:.*audio_tower.*"`
  - Embedding projections: `"re:.*embed.*"` (model-specific)

## Step 7 — Run quantization with canhazgpu

Launch the quantization script using `canhazgpu` with the required number of GPUs:

```bash
canhazgpu run --gpus <NUM_GPUS> -- python3 <script_name>.py
```

Wait for the quantization to complete. Monitor the output for errors.

## Step 8 — Write and run vLLM verification test

After quantization completes, create a `vllm_test.py` script to verify the compressed model loads and generates reasonable output:

```python
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    model_name = "<SAVE_DIR>"
    llm = LLM(model=model_name, tensor_parallel_size=<TENSOR_PARALLEL_SIZE>, trust_remote_code=True)

    outputs = llm.generate(
        ["Describe Large Language Models"],
        SamplingParams(temperature=0.8, max_tokens=200)
    )

    print(outputs[0].outputs[0].text)
```

Run the vLLM test:
```bash
canhazgpu run --gpus <NUM_GPUS> -- python3 vllm_test.py
```

## Step 9 — Validate results

After running `vllm_test.py`, check:
1. The model loaded successfully without errors.
2. The generated output is coherent and reasonable (not garbage/random tokens).

If both conditions are met, the MXFP4 quantization is successful. Report the result to the user.

If the output is garbled or the model fails to load, investigate:
- Check if `ignore` layers need adjustment
- Verify the correct algorithm was used (QuantizationModifier vs GPTQModifier)
- If using GPTQModifier, try increasing calibration samples
- Ensure tensor_parallel_size matches available GPUs

## Step 10 — Upload to Hugging Face (optional)

**Skip this step entirely if the user did not provide an HF repo name in Step 1.**

If the user provided a repo name, upload the quantized model directory and a generated README.

### 10a — Authenticate

Pick up the HF token from the user's environment:
```bash
echo $HF_TOKEN
```
If `$HF_TOKEN` is not set, try `$HUGGING_FACE_HUB_TOKEN`. If neither is set, ask the user to provide one or run `huggingface-cli login`.

### 10b — Generate README.md

Create a `README.md` inside the `<SAVE_DIR>` directory. The README must be **specific to the model being quantized**. Use the template below and fill in all placeholders:

- `<HF_REPO>` — the full repo name (e.g. `RedHatAI/Qwen3-8B-MXFP4`)
- `<BASE_MODEL_ID>` — the original unquantized model (e.g. `Qwen/Qwen3-8B`)
- `<MODEL_ARCH>` — the model's architecture class (e.g. `Qwen3ForCausalLM`, `LlamaForCausalLM`). Determine from the model's `config.json` `architectures` field.
- `<ALGORITHM>` — the algorithm used (`QuantizationModifier` or `GPTQModifier`)
- `<ALGORITHM_DESCRIPTION>` — algorithm-specific text:
  - QuantizationModifier: "Weights and activations are quantized using the MXFP4 format with MX block scaling. No calibration data is required."
  - GPTQModifier: "Weights and activations are quantized using the MXFP4 format with MX block scaling. GPTQ learned weight rounding with calibration data is used for improved accuracy."
- `<TENSOR_PARALLEL_SIZE>` — number of GPUs for vLLM
- `<QUANTIZATION_SCRIPT>` — the actual Python code used for quantization (from Step 4 or 5)
- `<LANGUAGES>` — infer from the base model card; if unknown, use `en` only
- `<LICENSE>` — infer from the base model card
- `<TAGS>` — include the model family tag (e.g. `qwen`, `llama`, `gemma`), `mxfp4`, `fp4`, `vllm`, `conversational`, `text-generation-inference`
- `<PIPELINE_TAG>` — typically `text-generation`

**README template:**

````markdown
---
language:
<LANGUAGES_LIST>
base_model:
- <BASE_MODEL_ID>
pipeline_tag: <PIPELINE_TAG>
tags:
<TAGS_LIST>
license: <LICENSE>
---

## Model Overview
- **Model Architecture:** <MODEL_ARCH>
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP4
  - **Weight quantization:** FP4
- **Intended Use Cases:** Intended for commercial and research use. Similarly to the base model, this quantized version is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing activations and weights of [<BASE_MODEL_ID>](https://huggingface.co/<BASE_MODEL_ID>) to MXFP4 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 4, reducing GPU memory requirements (by approximately 75%) and increasing matrix-multiply compute throughput (by approximately 4x).
Weight quantization also reduces disk size requirements by approximately 75%.

Only weights and activations of the linear operators within transformers blocks are quantized.
<ALGORITHM_DESCRIPTION>
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "<HF_REPO>"
number_gpus = <TENSOR_PARALLEL_SIZE>
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)
outputs = llm.generate(prompts, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

## Creation

<details>
  <summary>Creation details</summary>

  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

  ```python
  <QUANTIZATION_SCRIPT>
  ```
</details>
````

Adapt the template as needed:
- For multimodal models, change Input/Output to reflect vision/audio capabilities
- For MoE models, mention the MoE architecture in the overview
- Include any model-family-specific tags or metadata from the base model card

### 10c — Upload to Hugging Face

Upload the entire quantized weight directory (including the generated README.md) to the HF repo:

```bash
huggingface-cli upload <HF_REPO> <SAVE_DIR> --token $HF_TOKEN
```

If the repo does not exist yet, create it first:
```bash
huggingface-cli repo create <REPO_NAME> --organization <ORG> --type model --token $HF_TOKEN
```

Verify the upload succeeded by checking the repo page.

## Notes
- With `QuantizationModifier`: `oneshot(model=model, recipe=recipe)` with no dataset is correct — no calibration data is needed.
- With `GPTQModifier`: 512 samples at 2048 seq-len is a good default.
- MXFP4 targets AMD MI300X for inference; quantization can run on any GPU.
- `save_compressed=True` should be passed to `save_pretrained` for MXFP4 checkpoints.
- The virtual environment name defaults to `compress_model` but can be customized if the user prefers.
