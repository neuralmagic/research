---

## name: eval-agent

description: >-
  Run the standard benchmark suite (lm-eval + lighteval tasks) against a model
  served by vLLM, with GPU reservation via canhazgpu. Use when the user asks to
  evaluate a model, run standard benchmarks, or measure model accuracy across the
  full suite (GSM8k, MMLU, IFEval, Math-500, AIME25, GPQA Diamond, LiveCodeBench),
  or when they ask for SWE-bench Verified using an agentic coding harness.
  Handles GPU reservation, vLLM serving, benchmark execution, monitoring, retry,
  and summary. Supports instruct, reasoning, coding, and long_context categories;
  SWE-bench Verified uses a separate mini-SWE-agent + SWE-bench workflow.

# LLM Evaluation Agent

Run the unified lm-eval + lighteval standard benchmark suite against a model served
by vLLM. GPU reservation via canhazgpu. The CLI (`eval-agent`) handles deterministic
execution; you handle planning, monitoring, retry, and summary.

## Evaluation spec

Tasks and their benchmark settings are defined in:
`eval_agent/benchmarks/registry.yaml` (within the installed package, IMMUTABLE)

**You may NOT change:** task names, harness, max_gen_tokens, n_repetitions, num_fewshot,
apply_chat_template, fewshot_as_multiturn, task_str, metric.

**You control:** vllm serve command, gen_params, num_concurrent, timeout, port.

## ⚠️ MANDATORY FIRST STEP: Three-Venv Setup

**STOP. Before doing anything else, set up all three venvs.**

### Step 0: Setup and Validate Environment

```bash
# --- Base venv: router CLI + vllm ---
if [ ! -d ./.venv ]; then
    uv venv ./.venv --python 3.12
    source ./.venv/bin/activate
    SKILL_DIR=$(find ~/.claude/skills ~/.cursor/skills -name "eval-agent" -type d 2>/dev/null | head -1)
    uv pip install -e "$SKILL_DIR"
    uv pip install vllm
else
    source ./.venv/bin/activate
fi

# --- lm-eval venv ---
if [ ! -d ./.venvs/lm-eval ]; then
    uv venv ./.venvs/lm-eval --python 3.12
    ./.venvs/lm-eval/bin/pip install \
      "lm_eval[api,ifeval,multilingual] @ git+https://github.com/neuralmagic/lm-evaluation-harness.git@main"
fi

# --- lighteval venv ---
if [ ! -d ./.venvs/lighteval ]; then
    uv venv ./.venvs/lighteval --python 3.12
    ./.venvs/lighteval/bin/pip install \
      pillow "lighteval[extended] @ git+https://github.com/neuralmagic/lighteval.git@eldar-fix-litellm"
fi

# --- Validate ---
eval-agent --help
.venvs/lm-eval/bin/lm_eval --help
.venvs/lighteval/bin/lighteval --help
vllm --version
```

**If any validation command fails, STOP and fix the installation before proceeding.**

### Step 0d (optional): inspect-ai venv

Only needed if the user also wants to run `--category inspect_ai_core` (see
[Core Evals via Inspect AI](#core-evals-via-inspect-ai-optional) below). Not
part of the mandatory three-venv setup above — skip this unless asked.

```bash
if [ ! -d ./.venvs/inspect-ai ]; then
    uv venv ./.venvs/inspect-ai --python 3.12
    # `openai` is required by inspect_ai's `vllm` model provider even in pure
    # remote-client mode (talking to an already-running vLLM server via
    # --model-base-url) -- it is the underlying HTTP client. The heavy `vllm`
    # pip package itself is NOT required for this: it's only imported by
    # inspect_ai if you omit --model-base-url and ask it to spawn its own
    # local vLLM server instead, which this skill never does.
    ./.venvs/inspect-ai/bin/pip install inspect_ai "inspect_evals[ifeval]" openai
fi

# --- Validate ---
.venvs/inspect-ai/bin/inspect --help
```

## Workflow

### Step 1: Gather information

1. Verify all three venvs are set up (Step 0). If not, do Step 0 first.
2. Get model ID from user. Optionally get `--max-gpus` limit.
3. Download model configuration files:
  ```bash
   hf download <model_id> README.md config.json tokenizer_config.json generation_config.json \
     --local-dir ./model-configs/<model_name>
  ```
   Read `README.md`, `config.json`, `tokenizer_config.json`, `generation_config.json`.
4. Check for a vLLM recipe at `https://github.com/vllm-project/recipes`:
  ```bash
   curl -sL https://raw.githubusercontent.com/vllm-project/recipes/main/<Provider>/<ModelFamily>.md
  ```
   Use the Full-Featured server command from the recipe. Download any referenced chat template
   files into the workspace.
5. Determine model category:
  - Chat template has thinking/reasoning tokens (`<think>`, `<|think|>`, etc.): **reasoning**
  - Otherwise: **instruct**
  - User requests long-context eval: **long_context**
  - User requests coding eval: **coding**
6. Run `chg status` to see free GPUs. Wait every 120s if not enough are available.

### Step 2: Plan vLLM serving

1. Check if model is quantized (look for `quantization_config` in config.json or filename like `-FP8`, `-GPTQ`).
2. Estimate TP: use recipe value, or `ceil(1.5 × params × bytes_per_param / GPU_VRAM)`.
3. Compute DP: `floor(available_gpus / TP)`.
4. Determine `--max-model-len`:
  - **instruct/reasoning/coding**: `max(task.max_gen_tokens for tasks in category) + 4096`
    - instruct → 20480, reasoning → 69632, coding → 36864
  - **long_context**: depends on whether YaRN scaling is applied:
    - Ask the user: "Is YaRN (RoPE context extension) applied to this model?"
    - **Without YaRN**: use `max_position_embeddings` from `config.json`
    - **With YaRN**: read `rope_scaling` from `config.json`:
      ```json
      {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
      ```
      Effective max = `original_max_position_embeddings × factor`
      (e.g. 32768 × 4.0 = 131072)
    - If the model card or recipe specifies a different effective context, prefer that value.
5. Build the full `vllm serve` command (all recipe flags preserved).
6. Wrap with canhazgpu: `chg run --gpus <TP×DP> -- vllm serve ...`

### Step 3: Build generation params

Extract from `generation_config.json` or model card as a **JSON dict**:

```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
```

**Do NOT include** `max_new_tokens`, `max_gen_toks`, or `seed` — the CLI injects these per task and per seed automatically.

For greedy decoding (temperature=0 or not specified), pass `{}`.

### Step 4a: Smoke test (MANDATORY — doubles as concurrency calibration)

Choose an initial `--num-concurrent`:

- instruct / coding: start at **64**
- reasoning (long outputs): start at **32**

```bash
RUN_DIR="./runs/$(date +%Y%m%d_%H%M%S)_<model_short_name>_smoke"

eval-agent run \
  --model <model_name_as_served> \
  --server-cmd "<full chg run ... -- vllm serve ... command>" \
  --gen-params '{"temperature": 0.6, "top_p": 0.95}' \
  --category <instruct|reasoning|coding|long_context> \
  --port 8000 \
  --max-length <max_model_len_from_step2> \
  --num-concurrent <initial_value> \
  --timeout 1200 \
  --run-dir "$RUN_DIR" \
  --smoke-only
```

The smoke test runs each task once (first seed, 100 samples each).

**While the smoke test runs — monitor KV-cache utilization:**

```bash
# In a second terminal, tail the vLLM server log:
tail -f $RUN_DIR/logs/vllm_server.log | grep "GPU KV cache"
```

vLLM logs lines like:

```
GPU KV cache usage: 73.5%, CPU KV cache usage: 0.0%
```

- **Peak > 85%**: KV-cache pressure — halve `--num-concurrent` and re-run smoke test.
- **Peak ≤ 85%**: Proceed to full run with this concurrency.
- Also watch for `Preempted` or `Aborted request` as secondary signals.

**Do NOT proceed to the full run until all smoke tasks pass and concurrency is calibrated.**

Common smoke failures:

- Missing dependencies → reinstall the affected venv
- Bad gen_params → check format, re-run smoke
- Port conflict → change `--port`
- OOM at server startup → increase TP or reduce `--max-model-len`

### Step 4b: Full evaluation (only after smoke test passes)

```bash
RUN_DIR="./runs/$(date +%Y%m%d_%H%M%S)_<model_short_name>"

eval-agent run \
  --model <model_name_as_served> \
  --server-cmd "<full chg run ... -- vllm serve ... command>" \
  --gen-params '{"temperature": 0.6, "top_p": 0.95}' \
  --category <instruct|reasoning|coding|long_context> \
  --port 8000 \
  --max-length <max_model_len_from_step2> \
  --num-concurrent <calibrated_value> \
  --timeout 1200 \
  --run-dir "$RUN_DIR"
```

For reasoning models, consider `--timeout 3600`.

### Step 5: Monitor (CRITICAL — active supervision required)

**Do NOT wait passively.** Check every 30–60 seconds.

```bash
# In a second terminal, follow events:
eval-agent status --follow --run-dir $RUN_DIR

# In a third terminal, monitor KV-cache utilization:
tail -f $RUN_DIR/logs/vllm_server.log | grep "GPU KV cache"
```

**KV-cache threshold during full run**: >85% peak utilization → preemption risk. Halve
`--num-concurrent` and resume:

```bash
eval-agent resume --run-dir $RUN_DIR --num-concurrent <halved_value>
```

< 50% persistent utilization (not increasing over time) is a sign of poor hardware utilization.
Double `--num-concurrent` and resume:

```bash
eval-agent resume --run-dir $RUN_DIR --num-concurrent <doubled_value>
```

Also check `$RUN_DIR/events.jsonl` for `eval_failed` events. On any failure:

1. Read the task log: `tail -50 $RUN_DIR/logs/<task>_seed<seed>.log`
2. Identify root cause
3. Fix and resume

### Step 6: Handle failures

Maximum 3 retry attempts per failure type.


| Failure                   | Detection                                                         | Action                                                          |
| ------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- |
| **OOM at server startup** | "CUDA out of memory" in `vllm_server.log`                         | Increase TP, reduce DP. Run `eval-agent cleanup` first.         |
| **KV-cache preemption**   | GPU KV cache >85%, or `Preempted`/`Aborted` in vLLM log, HTTP 507 | Halve `--num-concurrent`. Resume.                               |
| **Timeout**               | Requests timing out well under `--timeout`                        | Double `--timeout`. If still failing, halve `--num-concurrent`. |
| **Bad gen params**        | SamplingParams mismatch in vLLM log                               | Fix `--gen-params`, re-run smoke.                               |
| **Port conflict**         | "Address already in use"                                          | Change `--port`.                                                |


**Never change task names, n_repetitions, or harness. Only adjust environment knobs.**

### Step 7: Cleanup

1. CLI kills vLLM automatically when run completes. `chg run` exits, freeing GPUs.
2. Run `chg status` to confirm GPUs are free.
3. If GPUs still reserved: `eval-agent cleanup --run-dir $RUN_DIR`, then check again.
4. If still stuck: `ps aux | grep vllm` and kill orphan processes manually.

### Step 8: Write summary

1. Read `$RUN_DIR/summary_data.json` (generated by CLI on completion).
2. Write `$RUN_DIR/summary.md` in plain English:
  - Model name, category, key configuration (TP, DP, gen_params, num_concurrent)
  - For each benchmark: mean score across seeds with context
    - lm-eval tasks: averaged across 3 seeds
    - lighteval tasks: averaged across 3 or 8 seeds (AIME25 uses 8)
  - Timing stats, anomalies (e.g. first seed slower due to compilation)
  - Failures with root cause analysis
  - Cleanup confirmation
3. Present key results to the user with brief interpretation.

## Core Evals via Inspect AI (optional)

A separate, additive category — `inspect_ai_core` — runs a handful of
standard benchmarks through the [Inspect AI](https://inspect.aisi.org.uk/) /
[`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) harness
instead of lm-eval or lighteval. It does not change or replace the instruct
/ reasoning / coding / long_context workflow above — use it only if the user
explicitly asks for Inspect AI evals, or wants a third, independent harness
on the same benchmarks.

**Included tasks** (confirmed to run with no `inspect_evals` code changes):
GSM8k (0-shot, 5-shot), MMLU (0-shot), MMLU (5-shot CoT), MMLU-Pro (0-shot,
5-shot), IFEval, GPQA Diamond. IFEval requires
`model_args: {responses_api: false}` (set in the registry already) to avoid
leaking `<think>` reasoning into the graded completion — without it, scores
are ~40 points lower than correct.

**Deliberately excluded: AIME.** `aime2024`/`aime2025`/`aime2026` in
`inspect_evals` share a scorer (`utils/aime_common.py`) with a last-line- and
state-mutation bug; the fix is an open, unmerged PR
([UKGovernmentBEIS/inspect_evals#2025](https://github.com/UKGovernmentBEIS/inspect_evals/pull/2025)).
Do not add AIME to this category until that PR merges.

**Deliberately excluded: HumanEval, MBPP, BigCodeBench.** These also run
against `inspect_evals` with no code changes, but they score by executing
generated code, which requires a real Docker daemon reachable from the host
running eval-agent — `inspect_evals` calls `docker`/`docker compose` directly
via subprocess, so a shell alias routing `docker` to Podman (the RHEL
default, and what our canhazgpu nodes have) is not sufficient. Do not add
these back until a real Docker Engine is provisioned on the target infra.

**Scores are not directly comparable** to the instruct/reasoning results
above for the "same" benchmark name (e.g. `gsm8k`) — `inspect_evals` uses
its own prompt templates and scorers, which differ from lm-eval/lighteval's.
Treat this as an independent additional read, not a cross-check.

### Running it

Reuses the same vLLM server, port, and `--gen-params` as the main workflow —
either point it at an already-running server from Step 4b, or start one the
same way as Step 2/3 and run this category standalone:

```bash
RUN_DIR="./runs/$(date +%Y%m%d_%H%M%S)_<model_short_name>_inspect_ai_core"

eval-agent run \
  --model <model_name_as_served> \
  --server-cmd "<full chg run ... -- vllm serve ... command>" \
  --gen-params '{"temperature": 0.6, "top_p": 0.95}' \
  --category inspect_ai_core \
  --port 8000 \
  --max-length <max_model_len> \
  --num-concurrent <calibrated_value> \
  --timeout 1200 \
  --inspect-ai-venv .venvs/inspect-ai \
  --run-dir "$RUN_DIR"
```

Same `--smoke-only`, `resume`, `status --follow`, and KV-cache monitoring
guidance from Steps 4-6 applies unchanged — this category just adds more
`(task, seed)` work items dispatched through a third harness.

## SWE-bench Verified (optional, agentic coding)

SWE-bench Verified is an agent benchmark, not a direct-completion task. The
model must inspect and edit a repository through tools; the SWE-bench harness
then applies the patch in an isolated environment and runs the task tests. Do
not add it to `registry.yaml` as a regular lighteval/lm-eval task, and do not
compare its score with LiveCodeBench as if they measured the same behavior.

Use the separate [SWE-bench Verified guide](SWE_BENCH_VERIFIED.md)
for setup, local vLLM settings, smoke runs, prediction evaluation, and storage
requirements. It includes a Qwen3.6 configuration based on the model card's
published SWE-bench settings. Keep patch generation and patch testing as
separate recorded stages, and use the same agent config and instance set when
comparing BF16 and quantized checkpoints.

## Resuming Interrupted Runs

```bash
eval-agent resume --run-dir <path-to-interrupted-run>
```

Reads `events.jsonl` to find completed (task, seed) pairs and skips them.
Starts a fresh vLLM server. Use `--timeout` and `--num-concurrent` to override.

**Do not use resume if:** smoke test failed (fix and re-run smoke), wrong model/category
(start fresh), or registry changed (start fresh).

## CLI Reference

```
eval-agent run     --model M --server-cmd CMD --gen-params JSON --category C \
                   --max-length L --run-dir D \
                   [--tasks NAME1,NAME2,...] \
                   [--port P] [--num-concurrent N] [--timeout T] \
                   [--lm-eval-venv PATH] [--lighteval-venv PATH] [--inspect-ai-venv PATH] \
                   [--health-timeout H] [--smoke-only]

eval-agent resume  --run-dir D [--timeout T] [--num-concurrent N] [--health-timeout H]
eval-agent status  --run-dir D [--follow]
eval-agent cleanup --run-dir D
```

`C` (`--category`) is one of `instruct`, `reasoning`, `coding`, `long_context`,
`inspect_ai_core`. `--inspect-ai-venv` (default `.venvs/inspect-ai`) is only
read when `C` is `inspect_ai_core`; see
[Core Evals via Inspect AI](#core-evals-via-inspect-ai-optional).

`--tasks` scopes a run to a comma-separated subset of task names within `C`
(e.g. `--tasks ifeval,gpqa_diamond`), instead of every task in the category.
Omit it to run the full category (the default, unchanged behavior). The
selection is recorded in the run's manifest, so `eval-agent resume` on that
run directory automatically resumes the same subset — no need to pass
`--tasks` again on resume.

## Verifying Sampling Parameters Reach vLLM

Start vLLM with logging enabled and inspect the SamplingParams lines:

```bash
VLLM_LOGGING_LEVEL=INFO vllm serve <model> \
    --enable-log-requests \
    --uvicorn-log-level info
```

Look for lines like:

```
SamplingParams(n=1, presence_penalty=1.5, temperature=0.6, top_p=0.95, top_k=20, seed=1234, ...)
```

Confirm seed, temperature, and all sampling params are correct before the full run.
