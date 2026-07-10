<!-- TEMPLATE: Model Card for RedHatAI Quantized LLMs

This template is a guide for an AI assistant to produce consistent model cards
for quantized models published by RedHatAI on HuggingFace.

PLACEHOLDERS use the format <PLACEHOLDER_NAME>. Replace each one with the
appropriate value. Sections wrapped in "IF ... ENDIF" comments are conditional
and should only be included when the condition is true.

SUPPORTED QUANTIZATION FORMATS:
  - INT W8A8:           weights INT8, activations INT8, model name suffix "-quantized.w8a8"
  - INT W4A16:          weights INT4, no activation quantization, model name suffix "-quantized.w4a16"
  - FP8 W8A8:           weights FP8, activations FP8 dynamic per-tensor, model name suffix "-FP8-dynamic"
  - FP8 W8A8 block:     weights FP8, activations FP8 block-wise scaling, model name suffix "-FP8-block"
  - NVFP4 (W4A4):       weights FP4, activations FP4, model name suffix "-NVFP4"
  - Mixed precision:     variable per-layer (FP4/FP8/BF16), suffix adapted to specific mix

-->

<!-- ============================================================
     SECTION 1: YAML FRONTMATTER

     Start from the ORIGINAL model's YAML header. Then:
       - REMOVE: gated, extra_gated_*, widget, inference fields
       - ADD to tags: the quantization data type tag(s) (e.g. fp8, fp4, int8, int4),
         plus vllm, llm-compressor, compressed-tensors
       - SET base_model to the original model's HuggingFace ID
       - KEEP everything else (language, license, pipeline_tag, etc.)
     ============================================================ -->
---
tags:
- <QUANTIZATION_TAG>
<!-- Add the appropriate data-type tag for the format:
       INT W8A8       -> int8
       INT W4A16      -> int4
       FP8 W8A8       -> fp8
       FP8 W8A8 block -> fp8
       NVFP4          -> fp4
       Mixed          -> use the primary data type tag(s), e.g. fp4, fp8 -->
- vllm
- llm-compressor
- compressed-tensors
<!-- COPY remaining fields from original model YAML (language, license, pipeline_tag, etc.)
     OMIT: gated, extra_gated_*, widget, inference -->
<ORIGINAL_YAML_FIELDS>
base_model: <ORIGINAL_MODEL_ID>
---

<!-- ============================================================
     SECTION 2: TITLE AND MODEL OVERVIEW
     ============================================================ -->

# <QUANTIZED_MODEL_NAME>

## Model Overview
- **Model Architecture:** <ARCHITECTURE_CLASS>
<!-- ARCHITECTURE_CLASS: the class name from config.json's architectures[0] field,
     e.g. LlamaForCausalLM, Gemma4ForConditionalGeneration, Qwen2ForCausalLM
     NOT the HuggingFace model ID or _name_or_path -->
  - **Input:** <MODEL_INPUT>
  - **Output:** <MODEL_OUTPUT>
<!-- MODEL_INPUT and MODEL_OUTPUT are usually "Text" / "Text".
     For multimodal models adjust accordingly (e.g. "Text / Image" / "Text"). -->
- **Model Optimizations:**
  - **Weight quantization:** <WEIGHT_QUANT_TYPE>
  - **Activation quantization:** <ACTIVATION_QUANT_TYPE>
<!-- WEIGHT_QUANT_TYPE / ACTIVATION_QUANT_TYPE values by format:
       INT W8A8       -> INT8 / INT8
       INT W4A16      -> INT4 / None
       FP8 W8A8       -> FP8 / FP8
       FP8 W8A8 block -> FP8 / FP8
       NVFP4          -> FP4 / FP4
       Mixed          -> describe the mix, e.g. "Mixed (FP4/FP8/BF16 per layer, avg <N> bits)" / same or "N/A"
-->
- **Release Date:** <RELEASE_DATE>
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [<ORIGINAL_MODEL_ID>](https://huggingface.co/<ORIGINAL_MODEL_ID>).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

<!-- IF mixed precision -->
<!-- Add a note like: "This is a mixed-precision quantization where individual layers are
     assigned FP4, FP8, or BF16 precision to achieve an average of <N> bits per parameter
     while preserving accuracy on sensitive layers." -->
<!-- ENDIF -->

<!-- ============================================================
     SECTION 3: MODEL OPTIMIZATIONS
     ============================================================ -->

### Model Optimizations

<!-- Choose the appropriate description based on the quantization format: -->

<!-- IF weights AND activations are quantized (INT W8A8, FP8 W8A8, FP8 W8A8 block, NVFP4) -->
This model was obtained by quantizing the weights and activations of [<ORIGINAL_MODEL_ID>](https://huggingface.co/<ORIGINAL_MODEL_ID>) to <QUANT_DATA_TYPE> data type, ready for inference with vLLM.
<!-- ENDIF -->

<!-- IF only weights are quantized (INT W4A16) -->
This model was obtained by quantizing the weights of [<ORIGINAL_MODEL_ID>](https://huggingface.co/<ORIGINAL_MODEL_ID>) to <QUANT_DATA_TYPE> data type while keeping activations in original precision, ready for inference with vLLM.
<!-- ENDIF -->

<!-- IF mixed precision -->
This model was obtained by applying mixed-precision quantization to [<ORIGINAL_MODEL_ID>](https://huggingface.co/<ORIGINAL_MODEL_ID>), assigning each layer FP4, FP8, or BF16 precision to achieve an average of <AVG_BITS> bits per parameter, ready for inference with vLLM.
<!-- ENDIF -->

This optimization reduces the number of bits per parameter from 16 to <BITS_PER_PARAM_QUANTIZED>, reducing the disk size and GPU memory requirements by approximately <REDUCTION_PERCENT>%.
<!-- Common reductions:
       16 -> 8 bits  = ~50% reduction
       16 -> 4 bits  = ~75% reduction
       For mixed precision, compute based on the average bits. -->

<!-- IF weights AND activations are quantized (INT W8A8, FP8 W8A8, FP8 W8A8 block, NVFP4, Mixed) -->
Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).
<!-- ENDIF -->

<!-- IF only weights are quantized (INT W4A16) -->
Only the weights of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).
<!-- ENDIF -->

<!-- ============================================================
     SECTION 4: DEPLOYMENT
     ============================================================ -->

## Deployment

### Use with vLLM

1. Start the vLLM server:
```
vllm serve RedHatAI/<QUANTIZED_MODEL_NAME> <VLLM_EXTRA_ARGS>
```
<!-- VLLM_EXTRA_ARGS: use flags from the vLLM recipes page for this model
     (https://recipes.vllm.ai/<ORG>/<BASE_MODEL_NAME>) as the primary reference,
     supplemented by the base model's HuggingFace README. Include only flags
     relevant to the model, for example:
       --reasoning-parser         (for reasoning models)
       --tool-call-parser         (for tool-calling models)
       --enable-auto-tool-choice  (for tool-calling models)
       --chat-template <path>     (when a custom chat template was used)
       --tokenizer_mode mistral   (if the model uses a Mistral tokenizer)
       --max_model_len N          (if specified in the base model card)
     Do not add --tensor_parallel_size unless the base model card prescribes it. -->

2. Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/<QUANTIZED_MODEL_NAME>"

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```
<!-- For model-specific request parameters (thinking mode, tool calling format,
     sampling settings), refer to the vLLM recipes page for this model:
     https://recipes.vllm.ai/<ORG>/<BASE_MODEL_NAME> -->

<!-- ============================================================
     SECTION 5: CREATION
     ============================================================ -->

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from <CALIBRATION_DATASET>, as presented in the code snippet below.
<!-- CALIBRATION_DATASET: set to the dataset used in the quantization script (e.g. UltraChat).
     If the quantization scheme requires no calibration data (e.g. FP8 dynamic),
     omit the "with calibration samples from ..." phrase entirely and use model_free_ptq(). -->

<details>

```python
<QUANTIZATION_SCRIPT>
```
<!-- Insert the full Python script used to produce this quantized model.
     The script typically includes:
       - Loading the original model and tokenizer
       - Loading and preprocessing calibration data (usually UltraChat)
       - Configuring the quantization recipe (varies by format):
           INT W8A8       -> GPTQModifier or QuantizationModifier with int8 weights + int8 activations
           INT W4A16      -> GPTQModifier with int4 weights, group_size 128, no activation quantization
           FP8 W8A8       -> model_free_ptq(scheme="FP8_DYNAMIC") — no calibration data needed
           FP8 W8A8 block -> QuantizationModifier with fp8 weights + fp8 block-scaled activations (group_size 128)
           NVFP4          -> QuantizationModifier with scheme="NVFP4", fp4 weights + fp4 activations, group_size 16
           Mixed          -> recipe with per-layer config mapping layers to different precisions
       - Running oneshot() or apply() to apply quantization
       - Saving the model in compressed-tensors format
-->
</details>

<!-- ============================================================
     SECTION 6: EVALUATION
     ============================================================ -->

## Evaluation

<!-- Add a sentence listing the benchmarks the model was evaluated on and the
     evaluation harnesses used. Mention both lm-evaluation-harness and lighteval
     if both were used. Mention BFCLv4 if tool-calling results are available.
     Only list benchmarks for which results are present.

     Example: "This model was evaluated on GSM8K Platinum, MMLU-Pro, IFEval,
     MATH-500, AIME 2025, GPQA Diamond, LiveCodeBench v6, and BFCLv4 using
     lm-evaluation-harness, lighteval, and BFCL — all served with vLLM
     (OpenAI-compatible API)." -->

<EVALUATION_DESCRIPTION>

### Accuracy

<!-- IF results are available for both thinking and non-thinking modes:
     use two subsections as shown below.
     IF only one set of results is available:
     present a single table without the subsection headers. -->

<!-- IF both thinking and non-thinking results are available -->
#### Without thinking
<!-- ENDIF -->

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th><ORIGINAL_MODEL_ID></th>
      <th>RedHatAI/<QUANTIZED_MODEL_NAME></th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval (0-shot, prompt-level strict)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>IFEval (0-shot, inst-level strict)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K Platinum (0-shot, strict-match)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot, custom-extract)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>MATH-500 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>AIME 2025 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <!-- Add or remove rows to match the benchmarks actually evaluated -->
  </tbody>
</table>

<!-- IF both thinking and non-thinking results are available -->
#### With thinking

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th><ORIGINAL_MODEL_ID></th>
      <th>RedHatAI/<QUANTIZED_MODEL_NAME></th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval (0-shot, prompt-level strict)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>IFEval (0-shot, inst-level strict)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K Platinum (0-shot, strict-match)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot, custom-extract)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>MATH-500 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>AIME 2025 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot, pass@1)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <!-- IF BFCLv4 results are available -->
    <tr>
      <td rowspan="4"><b>Tool Calling</b></td>
      <td>BFCLv4 Overall</td>
      <td><!-- baseline % --></td>
      <td><!-- quantized % --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>BFCLv4 Single Turn</td>
      <td><!-- baseline % --></td>
      <td><!-- quantized % --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>BFCLv4 Multi-Turn</td>
      <td><!-- baseline % --></td>
      <td><!-- quantized % --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>BFCLv4 Agentic</td>
      <td><!-- baseline % --></td>
      <td><!-- quantized % --></td>
      <td><!-- recovery % --></td>
    </tr>
    <!-- BFCLv4 scores shown as percentages (e.g. 68.31%), not plain numbers -->
    <!-- ENDIF BFCLv4 -->
    <!-- Add or remove rows to match the benchmarks actually evaluated -->
  </tbody>
</table>
<!-- ENDIF both thinking and non-thinking -->

### Reproduction

The results were obtained using the following commands:

<!-- Add the evaluation commands inside a collapsible <details> block.
     Include one command block per benchmark evaluated.
     Refer to evaluations.md for standard command formats, harness selection
     (lm-eval vs lighteval), and task names for each benchmark.
     Only include commands for benchmarks that were actually evaluated. -->

<details>

#### GSM8K Platinum (lm-eval, 0-shot, 3 repetitions)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/<QUANTIZED_MODEL_NAME>,max_length=<MAX_LENGTH>,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=<TEMPERATURE>,top_p=<TOP_P>,top_k=<TOP_K>,max_gen_toks=<MAX_GEN_TOKS>,seed=1234"
```

#### MATH-500, AIME 2025, GPQA Diamond (lighteval, 3 repetitions; 8 for AIME 2025)

**litellm_config.yaml:**
```yaml
model_parameters:
  provider: hosted_vllm
  model_name: hosted_vllm/RedHatAI/<QUANTIZED_MODEL_NAME>
  base_url: http://0.0.0.0:8000/v1
  api_key: ''
  timeout: 3600
  concurrent_requests: 32
  generation_parameters:
    temperature: <TEMPERATURE>
    max_new_tokens: <MAX_NEW_TOKENS>
    top_p: <TOP_P>
    top_k: <TOP_K>
    seed: 1234
```

Run once per seed (changing `seed` in the config each time):
```
lighteval endpoint litellm litellm_config.yaml 'math_500|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'aime25|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'gpqa:diamond|0' \
  --output-dir results/ --save-details
```

<!-- Add more command blocks for each benchmark evaluated. -->

<!-- IF BFCLv4 results are present -->
#### BFCLv4

BFCL requires the model to be registered in the leaderboard codebase before running.

**Step 1 — Register the model in `bfcl_eval/constants/model_config.py`**

Add the following entry to `api_inference_model_map`:

```python
"<MODEL_SLUG>": ModelConfig(
    model_name="<MODEL_SLUG>",
    display_name="<DISPLAY_NAME> (FC)",
    url="https://huggingface.co/RedHatAI/<QUANTIZED_MODEL_NAME>",
    org="<ORG>",
    license="<LICENSE>",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
),
```

**Step 2 — Add the key to `bfcl_eval/constants/supported_models.py`**

Add `"<MODEL_SLUG>"` to the `SUPPORTED_MODELS` list.

**Step 3 — Start the vLLM server** (use the command at the top of this section)

**Step 4 — Generate responses and evaluate**
```
bfcl generate --model <MODEL_SLUG> --test-category all
bfcl evaluate --model <MODEL_SLUG> --test-category all
```
<!-- ENDIF BFCLv4 -->

</details>
