---
name: fp8_compress
description: >
  Generate a working FP8 quantization example script, set up an environment, install dependencies,
  run quantization with canhazgpu, and verify the compressed model with vLLM.
  Triggers on: "fp8", "FP8_DYNAMIC", "FP8_BLOCK", "MXFP8", "fp8 example", "quantize to fp8".
allowed-tools: [Read, Write, Glob, Shell, AskQuestion]
---

# FP8 Quantization — End-to-End Workflow

Generate a working FP8 quantization script, create an isolated environment, run quantization on GPUs, and verify the result with vLLM.

## Step 1 — Gather information

Ask the user (or infer from context) for:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
2. **Scheme variant** — choose one:
   - `FP8_DYNAMIC` — weights fp8 per-channel, activations fp8 dynamic per-token. No calibration. Broadest hardware support (Ampere+). Recommended default.
   - `FP8_BLOCK` — weights fp8 with 128x128 block scaling, activations dynamic. No calibration. Best throughput on Hopper/Blackwell.
   - `MXFP8` — weights and activations in MX fp8 format. No calibration. AMD MI300X target.
3. **Model type** — dense, MoE, or multimodal (vision/audio)
4. **Use `model_free_ptq`?** — yes if the model is very large (70B+) and should avoid full GPU load; otherwise use `oneshot`
5. **Additional dependencies** — Ask: "Do you need any additional pip packages beyond `llmcompressor`, `vllm`, and `torchvision`?"

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

If the required number of GPUs exceeds what is available, warn the user and ask how to proceed (e.g. use `model_free_ptq` instead, or wait for resources).

Set `TENSOR_PARALLEL_SIZE` to the number of GPUs needed (used later for vLLM verification).

## Step 3 — Create virtual environment and install dependencies

Create an isolated environment using `uv`:

```bash
uv venv compress_model --python 3.12
source compress_model/bin/activate
```

Install core dependencies:
```bash
uv pip install llmcompressor vllm torchvision
```

If the user specified additional dependencies, install them too:
```bash
uv pip install <additional_packages>
```

## Step 4 — Choose the quantization template and write the script

### `oneshot` with `QuantizationModifier` (standard path)

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
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK | MXFP8
    ignore=["lm_head"],  # extend per model type — see Step 5
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### `model_free_ptq` (large models, avoids full GPU load)

```python
from llmcompressor import model_free_ptq

MODEL_ID = "<MODEL_ID>"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-<SCHEME>"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="<SCHEME>",  # FP8_DYNAMIC | FP8_BLOCK
    ignore=[
        "model.embed_tokens",
        "lm_head",
        # extend per model type — see Step 5
    ],
    max_workers=15,
    device="cuda:0",
)
```

Place the file in the appropriate directory under `examples/`:
- `quantization_w8a8_fp8/` for FP8_DYNAMIC or FP8_BLOCK (oneshot path)
- `quantization_w8a8_mxfp8/` for MXFP8
- `model_free_ptq/` when using `model_free_ptq()`

Name the file `{model_name_slug}_example.py` (e.g. `llama3_example.py`, `gemma4_example.py`).

## Step 5 — Apply model-type adjustments

### Dense models
`ignore=["lm_head"]` is sufficient in most cases.

### MoE models
Add gate/router layers to `ignore`. For models that require a custom loading path, wrap the load in `load_context`:
```python
from llmcompressor.utils import load_context

with load_context(SpecificModelClass):
    model = SpecificModelClass.from_pretrained(MODEL_ID)
```
Common MoE ignore additions:
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`
- Llama4 / Gemma4 MoE: `"re:.*router"`, `"Llama4TextAttention"`

For FP8_BLOCK on MoE, also skip attention: `"re:.*self_attn"`.

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Use the appropriate model class (e.g. `Gemma4ForConditionalGeneration`, `Llama4ForConditionalGeneration`) and wrap load in `load_context` if needed
- Add to `ignore`:
  - Vision: `"re:.*vision_tower.*"`, `"re:.*vision_model.*"`, `"re:.*multi_modal_projector.*"`
  - Audio: `"re:.*audio_tower.*"`
  - Embedding projections: `"re:.*embed.*"` (model-specific)

## Step 6 — Run quantization with canhazgpu

Launch the quantization script using `canhazgpu` with the required number of GPUs:

```bash
canhazgpu run --gpus <NUM_GPUS> -- python3 <script_name>.py
```

Wait for the quantization to complete. Monitor the output for errors.

## Step 7 — Write and run vLLM verification test

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

## Step 8 — Validate results

After running `vllm_test.py`, check:
1. The model loaded successfully without errors.
2. The generated output is coherent and reasonable (not garbage/random tokens).

If both conditions are met, the FP8 quantization is successful. Report the result to the user.

If the output is garbled or the model fails to load, investigate:
- Check if `ignore` layers need adjustment
- Verify the correct scheme was used for the hardware
- Ensure tensor_parallel_size matches available GPUs

## Notes
- `FP8_DYNAMIC` is the recommended starting point — no calibration required, broad hardware support.
- `FP8_BLOCK` is preferred for Hopper/Blackwell throughput.
- `MXFP8` targets AMD MI300X.
- Neither scheme requires a calibration dataset; `oneshot(model=model, recipe=recipe)` with no `dataset` argument is correct.
- `save_compressed=True` is optional — the checkpoint saves in compressed-tensors format either way. Omit unless explicitly requested.
- The virtual environment name defaults to `compress_model` but can be customized if the user prefers.
