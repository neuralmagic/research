# eval-agent

Unified LLM evaluation orchestrator for lm-eval and lighteval benchmarks, served via vLLM with GPU reservation via canhazgpu. Also supports an optional additional `inspect_ai_core` category run through the Inspect AI / `inspect_evals` harness (see [What it evaluates](#what-it-evaluates)).

## Adding to Claude Code or Cursor

**1. Copy the skill into your agent's skills directory:**

```bash
# Claude Code
cp -r eval-agent ~/.claude/skills/eval-agent

# Cursor
cp -r eval-agent ~/.cursor/skills/eval-agent
```

Or clone the full repo and symlink:

```bash
git clone git@github.com:neuralmagic/research.git
ln -s "$(pwd)/research/ai_skills/evaluation/eval-agent" ~/.claude/skills/eval-agent
```

**2. Restart Claude Code / Cursor** — the skill is picked up automatically on startup.

**3. Invoke it** by typing `/eval-agent` in the chat. The agent will:
- Ask for the model ID and any GPU constraints
- Set up the three evaluation venvs (first run only, ~5 min)
- Run the full benchmark suite and write a `summary.md` with results

**Workspace requirements** (must be available on the machine running the benchmarks):
- `uv` — Python venv manager
- `canhazgpu` (`chg`) — GPU reservation
- `hf` (Hugging Face CLI) — model config download
- CUDA-capable GPUs with enough VRAM for the target model

## What it evaluates

| Category | Tasks | Harness |
|---|---|---|
| instruct | GSM8k, MMLU, MMLU-Pro, IFEval, Math-500 | lm-eval + lighteval |
| reasoning | GSM8k, MMLU-Pro, IFEval, Math-500, AIME25, GPQA Diamond | lm-eval + lighteval |
| coding | LiveCodeBench v6 | lighteval |
| long_context | MRCR | lm-eval |
| inspect_ai_core (optional) | GSM8k (0-shot, 5-shot), MMLU (0-shot), MMLU (5-shot CoT), MMLU-Pro (0-shot, 5-shot), IFEval, GPQA Diamond | inspect-ai |

`inspect_ai_core` is a separate, additive category run through the
[Inspect AI](https://inspect.aisi.org.uk/) / [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals)
harness — it doesn't change or replace the four categories above, and its
scores aren't directly comparable to them for the "same" benchmark name
(different prompt templates/scorers). AIME is deliberately not included yet:
`inspect_evals`' AIME scorer has a bug fixed in an open, unmerged PR
([UKGovernmentBEIS/inspect_evals#2025](https://github.com/UKGovernmentBEIS/inspect_evals/pull/2025)).
HumanEval, MBPP, and BigCodeBench are also deliberately excluded: they score
by executing generated code, which requires a real Docker daemon reachable
from the host (not just a `docker`-aliased Podman, which is the RHEL/canhazgpu
node default) — add them once real Docker is provisioned on the target infra.
See `SKILL.md` → "Core Evals via Inspect AI" for usage.

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

A fourth, optional venv is only needed for the `inspect_ai_core` category:

| venv path | Contents | Use |
|---|---|---|
| `.venvs/inspect-ai/` | `inspect_ai` + `inspect_evals` (PyPI) | inspect-ai tasks (`inspect_ai_core` category only) |

## CLI

```
eval-agent run     --model MODEL --server-cmd CMD --gen-params JSON --category CATEGORY \
                   --max-length N --run-dir DIR \
                   [--tasks NAME1,NAME2,...] \
                   [--port N] [--num-concurrent N] [--timeout N] \
                   [--lm-eval-venv PATH] [--lighteval-venv PATH] [--inspect-ai-venv PATH] \
                   [--health-timeout N] [--smoke-only]

eval-agent resume  --run-dir DIR [--timeout N] [--num-concurrent N]
eval-agent status  --run-dir DIR [--follow]
eval-agent cleanup --run-dir DIR
```

`--tasks` runs only the named subset of `CATEGORY`'s tasks (e.g.
`--tasks ifeval,gpqa_diamond`) instead of all of them; `resume` picks up the
same subset automatically from the run's manifest.

`CATEGORY` is one of `instruct`, `reasoning`, `coding`, `long_context`,
`inspect_ai_core`.

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

Task definitions are in `eval_agent/benchmarks/registry.yaml`. The registry is **immutable** — task names, harness, max_gen_tokens, n_repetitions, and metric fields must not be changed. This applies per-category: the `inspect_ai_core` category (see [What it evaluates](#what-it-evaluates)) is additive and does not alter the task definitions of `instruct`, `reasoning`, `coding`, or `long_context`.

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
