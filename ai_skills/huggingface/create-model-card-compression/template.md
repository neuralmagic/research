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
- **Model Architecture:** <ORIGINAL_MODEL_ID>
  - **Input:** <MODEL_INPUT>
  - **Output:** <MODEL_OUTPUT>
<!-- MODEL_INPUT and MODEL_OUTPUT are usually "Text" / "Text".
     For multimodal models adjust accordingly (e.g. "Text/Image" / "Text"). -->
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

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

<!-- ============================================================
     SECTION 4: DEPLOYMENT
     ============================================================ -->

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/<QUANTIZED_MODEL_NAME> <VLLM_EXTRA_ARGS>
```
<!-- VLLM_EXTRA_ARGS: include flags relevant to the model, for example:
       --tensor_parallel_size N
       --tokenizer_mode mistral   (if the model uses a Mistral tokenizer)
       --max_model_len N          (if needed)
     Adjust based on the specific model's requirements. -->

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

<!-- ============================================================
     SECTION 5: CREATION
     ============================================================ -->

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from UltraChat, as presented in the code snippet below.

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
           FP8 W8A8       -> QuantizationModifier with fp8 weights + dynamic per-tensor fp8 activations
           FP8 W8A8 block -> QuantizationModifier with fp8 weights + fp8 activations using block scales (group_size 128)
           NVFP4          -> SmoothQuantModifier + QuantizationModifier with fp4 weights + fp4 activations, group_size 16
           Mixed          -> recipe with per-layer config mapping layers to different precisions
       - Running oneshot() to apply quantization
       - Saving the model in compressed-tensors format
-->
</details>

<!-- ============================================================
     SECTION 6: EVALUATION
     ============================================================ -->

## Evaluation

<!-- Add a sentence listing the benchmarks the model was evaluated on and the
     evaluation harnesses used. Mention both lm-evaluation-harness and lighteval
     if both were used. Link to the relevant repos.

     Example: "This model was evaluated on GSM8k-Platinum, MMLU-CoT, IFEval, and
     Math 500 using lm-evaluation-harness and lighteval."

     Refer to the evaluation protocol document (model_cards/Evaluations using vLLM server.md)
     for the full list of standard benchmarks, harness configurations, and commands. -->

<EVALUATION_DESCRIPTION>

### Accuracy

<!-- Add an HTML table with evaluation results.
     Columns: Category | Metric | <ORIGINAL_MODEL_ID> | RedHatAI/<QUANTIZED_MODEL_NAME> | Recovery
     Recovery = (quantized_score / original_score) * 100, shown as percentage.
     Group rows by category using rowspan. Include an Average row per category.

     Example: -->

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th><ORIGINAL_MODEL_ID></th>
      <th>RedHatAI/<QUANTIZED_MODEL_NAME></th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Instruction Following</b></td>
      <td>GSM8k-Platinum (5-shot)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>MMLU-CoT (5-shot)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td>IFEval (0-shot)</td>
      <td><!-- original score --></td>
      <td><!-- quantized score --></td>
      <td><!-- recovery % --></td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b><!-- avg original --></b></td>
      <td><b><!-- avg quantized --></b></td>
      <td><b><!-- avg recovery --></b></td>
    </tr>
    <!-- Add more categories and rows as needed (Reasoning, Coding, etc.) -->
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands:

<!-- Add the evaluation commands inside a collapsible <details> block.
     Include one command block per benchmark evaluated.
     Refer to model_cards/Evaluations using vLLM server.md for the standard
     command formats, harness selection (lm-eval vs lighteval), number of shots,
     and repetition settings for each benchmark.

     Two examples are shown below — one for lm-eval (generative tasks via vLLM server)
     and one for lighteval. Adapt the model name, generation parameters, and
     task names to match the actual evaluations run. -->

<details>

#### GSM8k-Platinum (lm-eval, 5-shot, 3 repetitions)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/<QUANTIZED_MODEL_NAME>,max_length=<MAX_LENGTH>,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results_gsm8k_platinum.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=<TEMPERATURE>,top_p=<TOP_P>,top_k=<TOP_K>,max_gen_toks=<MAX_GEN_TOKS>,seed=1234"
```

#### Math 500 (lighteval, 0-shot, 3 repetitions)
```
lighteval endpoint litellm litellm_config.yaml \
  "math_500@1@3|0" \
  --output-dir <OUTPUT_DIR> \
  --save-details
```

<!-- Add more command blocks for each benchmark evaluated.
     Remember to repeat each evaluation with different seeds as specified
     in the evaluation protocol. -->

</details>
