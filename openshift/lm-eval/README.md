# lm-eval on OpenShift

Accuracy benchmarks using
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Uses the [neuralmagic fork](https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant)
which adds the `mmlu_pro_chat` task variant.

> **Note:** Replace all occurrences of `<YOUR_NAME>` in the YAML with
> your identifier before applying.

## Example model

`lm-eval.yml` ships configured for **Qwen3-8B** on a single GPU as a lightweight
example. To use a different model, update the lines marked with `✎` in the YAML:

| Field | What to change |
|-------|----------------|
| Job name suffix | Match your model/quantization |
| `MODEL_ID` / `MODEL` | HuggingFace model ID |
| `QUANTIZATION` | `fp8`, `bf16`, etc. |
| `TP` / `DP` | Scale to model size |
| GPU requests/limits | Scale to model size |

## Tasks

The sweep runs 4 tasks across 3 seeds (0, 1, 2) for statistical robustness:

| Task | Few-shot | Description |
|------|----------|-------------|
| `gsm8k_platinum_cot_llama` | 0-shot | Grade school math with chain-of-thought |
| `mmlu_pro_chat` | 0-shot | MMLU-Pro in chat format |
| `ifeval` | 0-shot | Instruction following evaluation |
| `mmlu_cot_llama` | 5-shot | MMLU with chain-of-thought, few-shot as multi-turn |

All tasks use `temperature=0.6`, `top_p=0.95`, `max_gen_toks=32768`.

## Usage

```bash
oc apply -f lm-eval.yml
```

### Quick validation run

Set `LIMIT=N` on the benchmark container to cap samples per task:

```bash
oc set env job/mlr-vllm-<YOUR_NAME>-lm-eval LIMIT=10 -c benchmark
```

Or edit `lm-eval.yml` to add the env var before applying.

### Monitor progress

```bash
oc logs -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-lm-eval -c benchmark -f
```

### Clean up

```bash
oc delete job mlr-vllm-<YOUR_NAME>-lm-eval -n machine-learning
```
