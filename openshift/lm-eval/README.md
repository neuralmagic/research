# lm-eval on OpenShift

Accuracy benchmarks using
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
for evaluating FP8 quantization recovery on Command A+.

Uses the [neuralmagic fork](https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant)
which adds the `mmlu_pro_chat` task variant.

> **Note:** Replace all occurrences of `<YOUR_NAME>` in the YAML with
> your identifier before applying.

## Presets

`lm-eval.yml` ships configured for FP8. To switch presets, update the lines
marked with `✎` in the YAML:

| Field | FP8 (default) | BF16 baseline |
|-------|---------------|---------------|
| Job name suffix | `-lm-eval-fp8` | `-lm-eval-bf16` |
| `MODEL_ID` / `MODEL` | `RedHatAI/command-a-plus-05-2026-fp8` | `CohereLabs/command-a-plus-05-2026-bf16` |
| `QUANTIZATION` | `fp8` | `bf16` |
| `TP` / `DP` | `4` / `2` | `8` / `1` |

## Tasks

The sweep runs 4 tasks across 3 seeds (0, 1, 2) for statistical robustness:

| Task | Few-shot | Description |
|------|----------|-------------|
| `gsm8k_platinum_cot_llama` | 0-shot | Grade school math with chain-of-thought |
| `mmlu_pro_chat` | 0-shot | MMLU-Pro in chat format |
| `ifeval` | 0-shot | Instruction following evaluation |
| `mmlu_cot_llama` | 5-shot | MMLU with chain-of-thought, few-shot as multi-turn |

All tasks use `temperature=0.6`, `top_p=0.95`, `max_gen_toks=65536`.

## Usage

```bash
oc apply -f lm-eval.yml
```

### Monitor progress

```bash
oc logs -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-lm-eval-fp8 -c benchmark -f
```

### Clean up

```bash
oc delete job mlr-vllm-<YOUR_NAME>-lm-eval-fp8 -n machine-learning
```
