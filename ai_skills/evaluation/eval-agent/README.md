# eval-agent

Unified LLM evaluation orchestrator for lm-eval and lighteval benchmarks, served via vLLM with GPU reservation via canhazgpu.

## What it evaluates

| Category | Tasks | Harness |
|---|---|---|
| instruct | GSM8k, MMLU, MMLU-Pro, IFEval, Math-500 | lm-eval + lighteval |
| reasoning | GSM8k, MMLU-Pro, IFEval, Math-500, AIME25, GPQA Diamond | lm-eval + lighteval |
| coding | LiveCodeBench v6 | lighteval |
| long_context | MRCR | lm-eval |

## Installation

The skill is installed in the agent's workspace at runtime. No global installation needed.

The agent (SKILL.md) handles all three venv setups during Step 0. See SKILL.md for the full setup script.

**Requires in the workspace:**
- `uv` — for creating venvs
- `canhazgpu` (`chg`) — for GPU reservation
- `hf` (Hugging Face CLI) — for downloading model configs

**Python 3.12** is required for all three venvs.

## Three-venv architecture

| venv path | Contents | Use |
|---|---|---|
| `.venv/` | `eval-agent` package + `vllm` | CLI entry point + vLLM binary |
| `.venvs/lm-eval/` | neuralmagic fork of lm-evaluation-harness | lm-eval tasks |
| `.venvs/lighteval/` | neuralmagic eldar-fix-litellm branch of lighteval | lighteval tasks |

Harnesses run via their absolute binary paths — venv activation is not needed.

## CLI

```
eval-agent run     --model MODEL --server-cmd CMD --gen-params JSON --category CATEGORY \
                   --max-length N --run-dir DIR \
                   [--port N] [--num-concurrent N] [--timeout N] \
                   [--lm-eval-venv PATH] [--lighteval-venv PATH] \
                   [--health-timeout N] [--smoke-only]

eval-agent resume  --run-dir DIR [--timeout N] [--num-concurrent N]
eval-agent status  --run-dir DIR [--follow]
eval-agent cleanup --run-dir DIR
```

## Run directory structure

```
runs/<run_name>/
├── manifest.json          # Run config snapshot (immutable after creation)
├── events.jsonl           # Chronological audit trail
├── summary_data.json      # Machine-readable results (generated on completion)
├── summary.md             # Human-readable summary (written by agent)
├── commands.jsonl         # All harness commands actually executed
├── logs/
│   ├── vllm_server.log    # vLLM server output (KV-cache utilization here)
│   ├── gsm8k_seed1234.log
│   ├── math_500_seed1234.log
│   └── ...
├── configs/
│   ├── litellm_math_500_seed1234.yaml   # Per (task, seed) lighteval config
│   ├── litellm_aime25_seed1234.yaml
│   └── ...
└── results/
    ├── gsm8k_seed1234.json              # lm-eval result (single JSON file)
    ├── math_500_seed1234/               # lighteval result (directory)
    │   └── details/results_*.json
    └── ...
```

## Registry

Task definitions are in `eval_agent/benchmarks/registry.yaml`. The registry is **immutable** — task names, harness, max_gen_tokens, n_repetitions, and metric fields must not be changed.

Concurrency (`--num-concurrent`) is not in the registry. It is hardware-dependent and determined by the agent via smoke test KV-cache monitoring.

## Monitoring KV-cache utilization

vLLM logs GPU KV cache usage periodically:
```
GPU KV cache usage: 73.5%, CPU KV cache usage: 0.0%
```

Watch with:
```bash
tail -f runs/<run_name>/logs/vllm_server.log | grep "GPU KV cache"
```

If peak utilization exceeds 85%, halve `--num-concurrent` and resume. Values above 85% risk preemption, which collapses throughput.

## Resuming interrupted runs

```bash
eval-agent resume --run-dir runs/<run_name>
```

The runner reads `events.jsonl` to identify completed `(task, seed)` pairs and skips them. A fresh vLLM server is started. Use `--num-concurrent` and `--timeout` to override manifest values.
