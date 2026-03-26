---
name: create-model-card
description: >-
  Generate standardized HuggingFace model cards for RedHatAI quantized LLMs.
  Use when the user wants to create a model card, generate a README for a
  quantized model, prepare a model for HuggingFace upload, or mentions
  model cards, quantized model documentation, or RedHatAI model publishing.
---

# Create Model Card for Quantized LLMs

Interactive, conversational workflow to produce a HuggingFace model card for a
RedHatAI quantized model. Walk the user through each phase, confirm auto-detected
values, and let them skip or override any field.

## Platform Compatibility

This skill is designed to be portable across coding assistants and IDEs.

- Prefer capability-based behavior instead of platform-specific assumptions.
- If a platform has an authenticated Hugging Face integration (MCP/tool/plugin), use it.
- If not, fall back to `huggingface_hub` (Python), `hf` CLI, or manual user-provided files.
- Keep the model card generation flow identical regardless of integration method.

## Published card conventions (Red Hat AI)

These are the **defaults for the final README** unless the user asks otherwise:

### Model Overview

- Keep **weight / activation quantization** lines **minimal** (e.g. **FP8** / **FP4** / **INT8** only — no channel strategy, token scaling, or file references).
- Keep overview bullets short and simple. Leave details to other sections. For instance, quantization scheme details in **Model Optimizations**.
- **Release date** defaults to **yyyy-mm-dd** (e.g. `2026-03-25`) unless the user requests otherwise.
- One short intro sentence (quantized variant of base model + pointer to Evaluation / Reproduction as needed). Avoid duplicating long technical lists here.


### Model Optimizations (body section)

- Use `recipe.yaml`, `config.json` (`quantization_config`) to infer quantization information.
- Focus on **what optimizations were applied** and **user-visible benefits** (memory, disk, inference layout).
- Be descriptive about memory, disk usage and compute benefits, but keep it short and simple.
- Do not mention layers ignored in quantization.
- Focus on high-level description. No need to link to specific library versions or cite of `recipe.yaml`, `config.json`, or `quantization_config`


### Creation

- Short, factual: quantization tooling (**LLM Compressor**), scheme name (e.g. FP8 dynamic / W8A8), **compressed-tensors** export.
- If not provided, prompt the user to provide a script for quantization. The user may ignore or refuse.
- If the user does not provide a script, create a script yourself by reading the examples in https://github.com/vllm-project/llm-compressor/tree/main/examples and the model's recipe.yaml. The quantization scheme must follow what is dictated in recipe.yaml. The examples should be used only to understand data processing and llm-compressor syntax. Note that some quantization schemes do not require data.
- If the user does provide a Python script, embed it into the model card hidden under `<details>`.
- Simplify the Python script and remove any hard-coded local paths to make it general.


### Deployment (vLLM)

- **Simple** primary example: `vllm serve RedHatAI/<model>…` with **model-specific flags taken from the base model’s HuggingFace README** (e.g. `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`, `--speculative-config` for MTP, `--language-model-only`, `--max-model-len`) **merged into the command** when that matches upstream guidance.
- Do **not** prescribe **tensor parallel size** or long hardware tuning prose; a single line that users can adapt is enough.

### Evaluation (intro paragraph)

- **Brief**: name the **benchmarks** and the libraries — **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**, **[lighteval](https://github.com/huggingface/lighteval)**, and serving with **vLLM** (OpenAI-compatible API).
- Do **not** mention **forks** of lm-eval or lighteval unless the user explicitly asks; treat upstream as the reference.

### Accuracy table

- When **baseline (e.g. BF16) JSON** exists: include baseline and quantized columns plus **Recovery** (quantized score ÷ baseline score, as a percentage). Omit baseline/Recovery if no baseline.
- Prefer **HTML `<table>`** with **`rowspan`** on **Category** so each category label appears once per group.
- **IFEval**: default to **two benchmark rows** (e.g. prompt-level strict and instruction-level strict) with **Instruction following** `rowspan` on the category column — unless the user prefers a grouped layout.
- Omit a separate **metric** column if the user wants a cleaner table (benchmark names can carry the distinction).

### Reproduction

- Show the **simplest** representative **lm-eval** and **lighteval** commands (no seed loops in the snippet unless the user wants them).
- Summarize **repetitions per benchmark in prose** (e.g. 3 runs for most tasks, 8 for AIME): state that each task is run **N times** with seeds.
- Include the full **`litellm_config.yaml`** content used for lighteval (`provider`, `hosted_vllm` model id, `base_url`, timeouts, `concurrent_requests`, `max_model_length`, `generation_parameters` aligned with the model card / base model sampling guidance).
- Keep **sampling / adapter details** out of the Evaluation intro; they belong here under Reproduction.

## Phase 1: Gather Model Information

1. **Check Hugging Face integration availability early**:
   - Verify whether an authenticated Hugging Face integration is available before collecting model details.
   - Integration can be MCP, a native platform connector, CLI auth, or local token-based API access.
   - If integration auth is required, authenticate before any Hub operations.
   - If integration is unavailable, tell the user early that automated Hub fetch/upload will be limited and provide setup guidance for their platform.
   - If setup is not possible, continue with local/manual fallbacks (user-provided files and local README generation).

2. Ask the user for the **model path** (local folder or HuggingFace model ID).

3. **Read `config.json`** from the model path:
   - Extract `model_type`, `_name_or_path`, and architecture details.
   - For a HuggingFace model ID, fetch via the Hub API or `huggingface_hub` (raw `huggingface.co/.../raw/main/config.json` may return **401** for gated or token-only repos — use authenticated download or a workspace cache directory if needed).

4. **Read `recipe.yaml`** from the model path to determine the quantization
   format. Follow the parsing guide in `recipe-parsing.md` located in the **same folder as this `SKILL.md`**.

5. **Determine the original (base) model ID**:
   - Check `config.json` for `_name_or_path` or similar fields.
   - Infer from the model folder name (strip quantization suffix).
   - Present the inferred ID to the user and ask them to confirm or correct it.

6. **Fetch the original model's README.md** to extract its YAML frontmatter and **vLLM deployment section** (for flags to reuse on the quantized card):
   Use the best available integration (platform connector/MCP, `huggingface_hub`, or `hf` CLI).
   Parse the YAML between the `---` delimiters. This will be used to build the
   quantized model's YAML header and deployment hints.

Present a summary of everything detected so far and ask the user to confirm
before continuing. Example:

> Here is what I found:
> - **Original model**: meta-llama/Llama-3-8B-Instruct
> - **Quantization format**: NVFP4 (W4A4) — FP4 weights, FP4 activations
> - **Architecture**: LlamaForCausalLM
>
> Does this look correct?

## Phase 2: Collect Evaluation Results

7. **Search for evaluation result JSON files** in the model directory (and
   subdirectories). Look for:
   - **lm-eval output**: JSON files containing a top-level `"results"` key with
     task names mapping to metric dictionaries.
   - **lighteval output**: JSON files inside output directories with task metrics.
   - Common patterns: `results*.json`, `**/results.json`
   - Baseline runs may use parallel naming (e.g. `*_bf16_*` vs `*_fp8_*`); parse and match the same metrics.
   - Normalize benchmark/task names before matching (lowercase, trim whitespace, collapse repeated separators, and map common aliases such as `ifeval`/`IFEval`).

8. **If results are found**:
   - Parse each file, extracting benchmark name and metric scores.
   - Group by benchmark name.
   - If the same benchmark appears multiple times (different seeds/runs),
     **average the scores** across repetitions and report the average.
   - Present the parsed results to the user for confirmation.

9. **If no results are found**: tell the user and ask:
   - Provide a path to evaluation results
   - Paste results directly
   - Skip the evaluation section entirely

10. **Ask for unquantized (baseline) results**: request a path to the baseline
   model's evaluation results. If provided:
   - Load and match benchmarks by normalized benchmark and metric names (case-insensitive).
   - Compute Recovery = (quantized_score / baseline_score) * 100 for each metric.
   - If baseline score is `0`, missing, or non-numeric, set Recovery to `N/A` for that row and continue.
   - If not available, omit the Recovery column (and baseline column) from the table.

## Phase 3: Prompt for Missing Information

11. Present what has been auto-detected and ask for anything still missing.
    Keep it conversational — do NOT dump a long form. Ask in natural groups:

    **Group A — Model metadata** (with defaults):
    - Release date (default: today in **yyyy-mm-dd**)
    - Model input/output modalities (default: Text / Text; VL as needed)

    **Group B — Deployment** (often auto-filled from base README):
    - Confirm or adjust **model-specific vLLM flags** inferred from the base model card (not a generic tensor-parallel questionnaire unless the user wants it).

    **Group C — Optional**:
    - Full **quantization Python script** if the user has a path; otherwise skip.

    The user may say "skip" for any field or group.

## Phase 4: Generate Draft

12. **Read the model card template** from `template.md` located in the **same folder as this `SKILL.md`**. This template contains the full structure with placeholders and guidance comments. **Adapt** the template to the **Published card conventions** above when they conflict (overview brevity, deployment simplicity, no fork language, etc.). Try to keep to the template as closely as possible.

13. **Read the evaluation protocol** from
    `evaluations.md` located in the **same folder as this `SKILL.md`** (if present). If the file is missing, continue with best-practice defaults from this skill. Use it to choose harnesses, task names,
    shots, and repetition counts — but present **Reproduction** commands in the
    **simplified** style (yaml + short CLI, repetitions in prose).

14. **Fill the template** with all collected information:

    **YAML frontmatter**:
    - Start from the original model's YAML.
    - Remove: `gated`, `extra_gated_*`, `widget`, `inference` fields.
    - Add tags: quantization data type tag(s), `vllm`, `llm-compressor`,
      `compressed-tensors`.
    - Set `base_model` to the original model ID (the post-trained parent the
      quantization was applied to, as confirmed with the user).

    **All other sections**: follow the template and **Published card conventions**.

15. **Present the complete draft** to the user. Show it as a markdown code block
    so they can review the full content.

## Phase 5: Iterate

16. Ask the user for feedback. They may request:
    - Wording changes
    - Adding or removing sections
    - Fixing scores or metadata
    - Adjusting the YAML header

17. Incorporate changes and show the updated draft.

18. Repeat until the user says the draft is final.

## Phase 6: Save and Upload

19. **Save the model card**:
    - If the model path is a **local folder**: write it as `README.md` inside
      that folder.
    - If the model path is a **HuggingFace model ID**: write it to a local file
      (e.g. `model_cards/<model_name>_README.md`) and tell the user where it is.

20. **HuggingFace upload** (only if Hub upload access is available):
    - Re-use the availability/authentication result from Step 1.
    - If authentication is still needed, authenticate before upload actions.
    - If the model path is **local**: ask the user if they want to upload the
      entire model folder (including the new README.md) to
      `RedHatAI/<model_name>` on HuggingFace.
    - If the model path is a **HuggingFace model ID**: ask the user if they want
      to upload just the README.md to the existing repo.
    - **Only proceed after the user explicitly confirms.**
    - If no automated upload path is available, explicitly tell the user the card was generated locally and provide manual upload commands/options.

## Integration Notes (Optional, Platform-Specific)

Use these only when relevant for the current platform:

- **Cursor with Hugging Face MCP**:
  - Typical server name: `plugin-huggingface-skills-huggingface-skills`.
  - If available, authenticate early and use it for Hub reads/writes.
- **CLI fallback**:
  - Authenticate: `hf auth login`
  - Upload README-only: `hf upload <repo_id> README.md README.md`
- **Python fallback**:
  - Use `huggingface_hub` authenticated API calls for `config.json` and `README.md`.

## Style Guidelines

- Be conversational and helpful throughout.
- Always confirm auto-detected values before using them.
- Never block on missing information — offer to skip any field.
- When presenting the draft, show the full markdown so the user can review it.
- When asking questions, group related items together to avoid a long back-and-forth.
