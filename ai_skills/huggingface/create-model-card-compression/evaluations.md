# Evaluations using vLLM server

This document is a catalogue of known benchmarks and their evaluation protocols.
The skill should search for and report whatever benchmarks are available — only
include results that were actually run. Do not skip or error if a benchmark is missing.

---

## Serving the model

Start the vLLM server using the configuration recommended by the model provider.
The vLLM recipes page (`recipes.vllm.ai/<org>/<model>`) is the primary reference for
model-specific flags such as `--tool-call-parser`, `--reasoning-parser`,
`--tokenizer-mode`, and `--load-format`.

```
vllm serve <model_name> ...
```

---

## lighteval

**Note:** use the neuralmagic fork: https://github.com/neuralmagic/lighteval

To use lighteval with a vLLM server, use the litellm endpoint.

**litellm_config.yaml**

```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/<model_name>"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 1200
  concurrent_requests: 16
  generation_parameters:
    temperature: 0.6
    max_new_tokens: 65536
    top_p: 0.95
    seed: 0
    top_k: 20
    presence_penalty: 1.5
```

**Notes:**

- The model_name must be preceded by `hosted_vllm`. Examples:
  - HF model: `moonshotai/Kimi-K2-Thinking` → `model_name: "hosted_vllm/moonshotai/Kimi-K2-Thinking"`
  - Local model: `/local-dir` → `model_name: "hosted_vllm/local-dir"`
- Generation parameters should match the model provider's recommendations.
- `max_model_length` needs to be specified if the model's default length is overridden when launching the vLLM server.
- lighteval allows generating multiple responses per sample, each with a different seed. Set `concurrent_requests` accordingly.
- The seed is specified in `generation_parameters`.
- The timeout parameter controls time in seconds per request.
- Default `concurrent_requests` is low (10); increase for better throughput.

**Evaluation command:**

```shell
lighteval endpoint litellm litellm_config.yaml \
  "aime25@<k>@<n>|0,math_500@<k>@<n>|0" \
  --output-dir <output-dir> \
  --save-details
```

**Notes:**

- Task format: `<task_name>@<k>@<n>|<num_fewshot>`. If k and n are omitted, they default to 1.
  - `aime25@1@8|0` — 8 seeds, pass@1
  - `math_500@1@3|0` — 3 seeds, pass@1
  - `gpqa:diamond@1@3|0` — 3 seeds
- k and n refer to pass@k with n samples; used to average results over multiple seeds.
- Inline string alternative (useful in scripts):

```shell
lighteval endpoint litellm \
  "model_name=hosted_vllm/${SERVED_MODEL_NAME},provider=hosted_vllm,base_url=http://0.0.0.0:${PORT}/v1,timeout=3600,concurrent_requests=8,generation_parameters={temperature:${TEMP},max_new_tokens:${MAX_NEW_TOKENS},top_p:${TOP_P},seed:${SEED},top_k:${TOP_K}}" \
  "aime25@k=${K}@n=${N_AIME}|0,math_500@k=${K}@n=${N_OTHERS}|0,gpqa:diamond@k=${K}@n=${N_OTHERS}|0,lcb:codegeneration_v6|0" \
  --output-dir ${OUTPUT_DIR} \
  --save-details
```

---

## lm-eval (generative tasks)

**Note:** use the neuralmagic fork: https://github.com/neuralmagic/lm-evaluation-harness

**Evaluation command:**

```shell
lm_eval --model local-chat-completions \
  --tasks gsm8k \
  --model_args "model=<model_name>,max_length=<max_length>,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results_gsm8k.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=1.0,top_k=20,max_gen_toks=64000,seed=1234"
```

**Notes:**

- Always set `max_length` explicitly (the default of 2048 is very low).
- Set `do_sample=True` explicitly when using non-greedy decoding.
- Include `seed` in both `--seed` and `gen_kwargs`.
- Only works with generative tasks; does not work with log-likelihood or multiple-choice tasks.

## lm-eval (multiple-choice tasks, no chat template)

```
lm_eval --model local-completions \
  --tasks mmlu \
  --model_args "model=<model_name>,max_length=<max_length>,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=10,max_retries=3,tokenized_requests=False" \
  --num_fewshot 5 \
  --output_path results_mmlu.json
```

## Debugging tips

- Limit samples for quick tests: `--max-samples 5` (lighteval) or `--limit 5` (lm-eval)
- Verify sampling args reach vLLM:

```shell
VLLM_LOGGING_LEVEL=INFO vllm serve <model> \
    --enable-log-requests \
    --uvicorn-log-level info
```

---

# Standard benchmarks

## Standard protocol

- Always use the chat template (`--apply_chat_template`)
- Use `--fewshot_as_multiturn` when using 1 or more few-shot examples
- Match generation parameters to the model provider's recommendations
- Each benchmark entry lists the recommended number of repetitions (runs with different random seeds)
- For lighteval, use `@1@n` to denote average over n repetitions. Example: AIME 2025 with 8 repetitions: `aime25@1@8`

---

## Instruction Following

### GSM8K Platinum

- **Harness:** lm-eval
- **Task:** `gsm8k_platinum_cot_llama`
- **Shots:** 5
- **Metric:** `exact_match,strict-match`
- **Repetitions:** 3

### MMLU-CoT

- **Harness:** lm-eval
- **Task:** `mmlu_cot_llama`
- **Shots:** 5
- **Metric:** `exact_match,strict_match`
- **Repetitions:** 3

### MMLU-Pro

- **Harness:** lm-eval (neuralmagic fork)
- **Task:** `mmlu_pro_chat`
- **Shots:** 5
- **Metric:** `exact_match,custom-extract`
- **Repetitions:** 3

### IFEval

- **Harness:** lm-eval
- **Task:** `ifeval`
- **Shots:** 0
- **Metrics:** `prompt_level_strict_acc` and `inst_level_strict_acc`
- **Repetitions:** 3

---

## Reasoning

### GSM8K Platinum

- **Harness:** lm-eval
- **Task:** `gsm8k_platinum_cot_llama`
- **Shots:** 0
- **Metric:** `exact_match,strict-match`
- **Repetitions:** 3

### MMLU-Pro

- **Harness:** lm-eval (neuralmagic fork)
- **Task:** `mmlu_pro_chat`
- **Shots:** 0
- **Metric:** `exact_match,custom-extract`
- **Repetitions:** 3

### IFEval

- **Harness:** lm-eval
- **Task:** `ifeval`
- **Shots:** 0
- **Metrics:** `prompt_level_strict_acc` and `inst_level_strict_acc`
- **Repetitions:** 3

### MATH-500

- **Harness:** lighteval
- **Task:** `math_500`
- **Shots:** 0
- **Metric:** `pass@k:k=1&n=1`
- **Repetitions:** 3

### AIME 2025

- **Harness:** lighteval
- **Task:** `aime25`
- **Shots:** 0
- **Metric:** `pass@k:k=1&n=1`
- **Repetitions:** 8

### GPQA Diamond

- **Harness:** lighteval
- **Task:** `gpqa:diamond`
- **Shots:** 0
- **Metric:** `gpqa_pass@k:k=1`
- **Repetitions:** 3

---

## Coding

### LiveCodeBench v6

- **Harness:** lighteval
- **Task:** `lcb:codegeneration_v6`
- **Shots:** 0
- **Metric:** `codegen_pass@1:16`
- **Repetitions:** 3

---

## Tool Calling

### BFCLv4

- **Tool:** Berkeley Function Call Leaderboard CLI (`bfcl`)
- **When to include:** For models with tool-calling support, when BFCL results are available.
- **Test category:** `all`
- **Reported sub-scores:**
  - **Overall** (weighted composite)
  - **Single Turn** (Non-Live AST Accuracy)
  - **Multi-Turn**
  - **Agentic**
- **Scoring weights:** Non-Live=10, Live=10, Irrelevance=10, Multi-Turn=30, Agentic=40
  - Agentic = unweighted average of web_search and memory categories
  - Overall = weighted sum of all categories divided by 100
- **Repetitions:** 1 (BFCL is deterministic for a fixed model)

### Where to find results

BFCL scores may be in several locations depending on how the evaluation was run:

- `score/data_overall.csv` — unquantized model scores
- `score/quantized/data_overall.csv` — quantized model scores (may not exist)
- Per-category JSON files under `score/` — needed to compute sub-scores when
  the CSV does not contain them

If results cannot be located automatically, prompt the user for the path to the
BFCL score directory.

The Overall score is typically pre-computed in the CSV. Sub-scores may require
reading per-category JSON files and applying the scoring weights above.

### Registration (for running new evaluations)

1. Add a `ModelConfig` entry to `bfcl_eval/constants/model_config.py` under `api_inference_model_map`.
2. Add the model key to the `SUPPORTED_MODELS` list in `bfcl_eval/constants/supported_models.py`.
3. The model slug must match `--served-model-name` passed to vLLM (or the HuggingFace repo path by default).

### Commands

```shell
bfcl generate --model <MODEL_SLUG> --test-category all
bfcl evaluate --model <MODEL_SLUG> --test-category all
```
