# OpenShift Benchmarks

Job and Pod definitions for running ML benchmarks and experiments on the
OpenShift H100 cluster.

## Directory structure

Each task lives in its own subdirectory with a README and one or more YAML files:

```
openshift/
├── README.md              ← this file (conventions & architecture)
├── bfcl/
│   ├── README.md
│   └── bfcl.yml
└── utility-pods/
    ├── README.md
    ├── cpu.yml
    └── h100.yml
```

## Shared prerequisites

### HuggingFace token

Most tasks need a HF token for model downloads. Create the secret once:

```bash
oc create secret generic mlr-<YOUR_NAME>-hf-token-read-only \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n machine-learning
```

Then reference it in container env:

```yaml
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: mlr-<YOUR_NAME>-hf-token-read-only
      key: HF_TOKEN
```

### Tier2 PVC

All tasks share a user-specific PVC (`mlr-tier2-<YOUR_NAME>`, CephFS,
ReadWriteMany) mounted at `/tier2`.

### Cleanup label

Every resource should carry an `app.kubernetes.io/instance` label so the
entire task can be torn down with one command:

```bash
oc delete all -l app.kubernetes.io/instance=<instance-name> -n machine-learning
```

## Creating a new task

1. Create a subdirectory: `openshift/<task-name>/`
2. Add your YAML file(s) and a `README.md` with usage instructions.
3. Follow the conventions below.

### Job conventions

- Use `kind: Job` with `backoffLimit: 0` for run-to-completion workloads.
  Use `kind: Pod` only for interactive utility pods (e.g. `sleep infinity`).
- Set `activeDeadlineSeconds` at the Job spec level.
- Set `restartPolicy: Never` in the pod template spec.
- Use `nodeSelector: node-role.kubernetes.io/up-h100mcp: ""` for GPU work.
- Use `serviceAccountName: ml-workload`.
- Reference the HF token from the shared secret — never hardcode tokens.
- Add the `app.kubernetes.io/instance` label to both Job metadata and the
  pod template metadata for cleanup and Service selectors.

### Server + sidecar architecture

For benchmarks that pair a vLLM server with an eval harness, use the
**native sidecar** pattern (requires Kubernetes 1.28+ / OpenShift 4.15+).
Place the server in `initContainers` with `restartPolicy: Always` so that:

1. The server starts before the benchmark container.
2. When the benchmark container exits (success or failure), Kubernetes
   automatically sends SIGTERM to the server.
3. The pod completes without burning GPU time on idle servers.

```yaml
apiVersion: batch/v1
kind: Job
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 86400
  template:
    spec:
      restartPolicy: Never
      initContainers:
      - name: vllm-server
        restartPolicy: Always    # native sidecar — runs alongside main container
        image: vllm/vllm-openai:v0.22.1
        ...
      containers:
      - name: benchmark          # main container — its exit terminates the pod
        image: python:3.12-slim
        ...
```

### Benchmark results

All benchmark tasks must write results to:

```
/tier2/benchmark_results/<timestamp>/
```

where `<timestamp>` is UTC formatted as `YYYYMMDD-HHMMSS` (e.g., `20260610-143052`).

Each result directory must contain:
- `config.json` — written at the start with the fields below, so partial runs
  are still identifiable.
- `run-metadata.json` — written **only** when the entire sweep completes
  successfully. Its presence is the definitive indicator that the run finished.
  A directory with `config.json` but no `run-metadata.json` is an
  incomplete or failed run.

### run-metadata.json spec

Written at the very end of the benchmark script. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"completed"` (file only exists on success). |
| `start_timestamp` | `string` | UTC start time (`YYYYMMDD-HHMMSS`). |
| `end_timestamp` | `string` | UTC end time (`YYYYMMDD-HHMMSS`). |
| `duration_seconds` | `int` | Wall-clock duration of the benchmark sweep. |
| `result_files_written` | `int` | Number of result JSON files produced. |

### config.json spec

#### Required top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_type` | `string` | Identifies the benchmark (e.g., `"bfcl"`, `"speed-bench"`). Used as the primary filter key. |
| `target_model` | `string` | The model being served (e.g., `"Qwen/Qwen3-8B"`). |
| `run_timestamp` | `string` | UTC timestamp matching the directory name (`YYYYMMDD-HHMMSS`). |

#### Required sections

**`server`** — everything needed to reproduce the server configuration:

| Field | Type | Description |
|-------|------|-------------|
| `server.vllm_version` | `string` | Version string from the `/version` endpoint. |
| `server.model` | `string` | Model identifier passed to `vllm serve`. |
| `server.max_model_len` | `int \| null` | The `--max-model-len` value, or `null` if using the model default. |
| `server.image` | `string` | Container image (omit if built from source). |

Include any other server flags that affect results (e.g. `speculative_config`,
`tensor_parallel_size`, `tool_call_parser`).

**`client`** — the benchmark runner:

| Field | Type | Description |
|-------|------|-------------|
| `client.harness` | `string` | Harness name (e.g., `"bfcl"`, `"lm-eval"`). |
| `client.image` | `string` | Container image used for the benchmark client. |

**`benchmark`** — dataset and parameters:

Add benchmark-specific fields (tasks, categories, num_threads, etc.)
under this section.

#### Example

```json
{
  "benchmark_type": "bfcl",
  "target_model": "RedHatAI/command-a-plus-05-2026-fp8",
  "run_timestamp": "20260610-143052",
  "server": {
    "vllm_version": "0.22.1",
    "model": "RedHatAI/command-a-plus-05-2026-fp8",
    "image": "vllm/vllm-openai:v0.22.1",
    "tensor_parallel_size": 8,
    "tool_call_parser": "cohere_command4"
  },
  "client": {
    "harness": "bfcl",
    "image": "python:3.12-slim"
  },
  "benchmark": {
    "test_categories": ["non_live", "multi_turn"],
    "num_threads": 12
  }
}
```

### Searching results

```bash
# All benchmark runs
ls /tier2/benchmark_results/

# All runs of a specific benchmark type
grep -rl '"benchmark_type": "bfcl"' /tier2/benchmark_results/*/config.json

# All runs for a specific model
grep -rl '"target_model": "Qwen/Qwen3-8B"' /tier2/benchmark_results/*/config.json

# Only completed runs
ls /tier2/benchmark_results/*/run-metadata.json

# Incomplete/failed runs
for d in /tier2/benchmark_results/*/; do
  [ -f "$d/config.json" ] && [ ! -f "$d/run-metadata.json" ] && echo "$d"
done
```
