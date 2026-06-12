# BFCL v4 on OpenShift

Tool calling benchmarks using the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla).

Each benchmark runs as a Kubernetes **Job** with two containers on an H100 node:
- **vllm-server** — serves the model on port 8000 with `--served-model-name`
  matching the BFCL registry (image: `vllm/vllm-openai:v0.22.1`)
- **benchmark** — clones the gorilla fork, registers the model, then runs
  `bfcl generate` + `bfcl evaluate` for each test category

> **Note:** Replace all occurrences of `<YOUR_NAME>` in the YAML with
> your identifier before applying. This affects the Job name, secret references,
> and PVC claim names.

## Example model

`bfcl.yml` ships configured for **Qwen3-8B** on a single GPU as a lightweight
example. To use a different model, update the lines marked with `✎` in the YAML:

| Field | What to change |
|-------|----------------|
| Job name suffix | Match your model/quantization |
| `MODEL_ID` / `MODEL` | HuggingFace model ID |
| `QUANTIZATION` | `fp8`, `bf16`, etc. |
| `NUM_THREADS` | Scale to model throughput |
| GPU requests/limits | Scale to model size |

For larger models, also increase the tier1 ephemeral storage and adjust
`--tensor-parallel-size` / `--data-parallel-size` in the vllm-server script.

For models that require a specific tool-call parser, add
`--tool-call-parser <parser>` to the `vllm serve` command in the vllm-server
script (e.g. `--tool-call-parser hermes`).

## Prerequisites

### 1. Create the HF token secret (one-time)

```bash
oc create secret generic mlr-<YOUR_NAME>-hf-token-read-only \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n machine-learning
```

### 2. Ensure the tier2 PVC exists

The YAML mounts `mlr-tier2-<YOUR_NAME>` (ReadWriteMany) for:
- HF model cache (`/tier2/hf-hub`)
- Benchmark results (`/tier2/benchmark_results/<timestamp>/`)

## Usage

### Run a benchmark

```bash
oc apply -f bfcl.yml
```

### Monitor progress

```bash
# Find the pod created by the Job
oc get pods -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-bfcl

# Server logs
oc logs -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-bfcl -c vllm-server -f

# Benchmark logs
oc logs -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-bfcl -c benchmark -f
```

### Clean up

```bash
# By name
oc delete job mlr-vllm-<YOUR_NAME>-bfcl -n machine-learning

# Or by label
oc delete all -l app.kubernetes.io/instance=mlr-vllm-<YOUR_NAME>-bfcl -n machine-learning
```

## What it benchmarks

The sweep runs 2 test categories using native tool calling (`is_fc_model=True`):

| Category | Description | Reproducibility |
|----------|-------------|-----------------|
| `non_live` | Single-turn tool use (simple, parallel, multiple, java, javascript, relevance) | Static dataset, fully reproducible |
| `multi_turn` | Multi-turn dialog with tool use, deterministic simulated backends | Deterministic, fully reproducible |

The sidecar uses the
[neuralmagic gorilla fork](https://github.com/neuralmagic/gorilla/tree/shubhra/bfcl-vllm-patches)
which includes the OpenAI completions fix and optional DuckDuckGo web search.

### How it works

1. The sidecar clones the fork, installs BFCL, and auto-registers the model
   in `model_config.py` and `supported_models.py`
2. A `.env` file is written pointing `OPENAI_BASE_URL` at the local vLLM server
3. For each category: `bfcl generate` sends prompts and collects responses,
   then `bfcl evaluate` scores them
4. Both `result/` and `score/` directories are copied to the timestamped
   result directory on tier2

Each run produces a timestamped directory under `/tier2/benchmark_results/` containing:
- `config.json` — full reproducibility metadata
- `bfcl_result/` — raw generation output per test category
- `bfcl_score/` — evaluation scores and CSVs
- `run-metadata.json` — written only on successful completion

## Configuration

### Model

To use a different model, update the `MODEL_ID` / `MODEL` variables
(marked `✎`) in both container scripts. The `--served-model-name` must match
the model name registered in BFCL.

### Test categories

To add or change categories, edit the `BFCL_CATEGORIES` variable in the
benchmark container. Available categories:

```bash
# After installing BFCL:
bfcl test-categories
```

### GPU count

The YAML requests 1 GPU by default. Update `nvidia.com/gpu` in the vllm-server
container's resources to match your model's requirements.
