---
name: compress_models
description: >
  Quantize / compress a HuggingFace model using llm-compressor.
  Routes to the correct scheme-specific skill (FP8, INT, NVFP4, MXFP4) based on the user's request.
  Triggers on: "compress", "quantize", "quantization", "fp8", "FP8_DYNAMIC", "FP8_BLOCK", "MXFP8",
  "int8", "int4", "W8A8", "W4A16", "GPTQ", "SmoothQuant", "integer quantization",
  "nvfp4", "NVFP4", "fp4 nvidia", "fp4 hopper", "fp4 blackwell", "h100 fp4",
  "mxfp4", "MXFP4", "fp4 amd", "mi300x quantization",
  "compress model", "quantize model", "model compression".
allowed-tools: [Read, Write, Glob, Shell, AskQuestion, WebFetch]
---

# Model Compression — Unified Skill Router

This skill determines the appropriate quantization scheme and delegates to the scheme-specific sub-skill.

## Step 0 — Detect the user's intent

Before routing to a scheme, determine **what the user is asking to do**. The user's request falls into one of these categories:

### A) Upload-only (skip quantization)

The user already has a quantized model on disk and wants to **generate a README and/or upload to Hugging Face**.

**Trigger phrases:** "upload", "push to hub", "create readme", "generate readme", "publish model", "upload to huggingface", "push to hf", combined with a local model directory path.

**Required information — gather from the user (or infer from context):**

1. **SAVE_DIR** — path to the local quantized model directory
2. **HF_REPO** — target Hugging Face repo name (e.g. `RedHatAI/Qwen3-8B-FP8-dynamic`)
3. **BASE_MODEL_ID** — the original unquantized model ID (e.g. `Qwen/Qwen3-8B`). If not provided, try to infer from the model directory name or `config.json` inside `SAVE_DIR`.
4. **Scheme** — the quantization scheme used (FP8_DYNAMIC, FP8_BLOCK, MXFP8, W4A16, W8A8, NVFP4, MXFP4). If not provided, try to infer from the directory name (e.g. `-FP8-dynamic` → FP8_DYNAMIC, `-NVFP4` → NVFP4) or from `config.json` / `quantization_config` inside `SAVE_DIR`.
5. **Quantization script (optional)** — the Python code used for quantization, to embed in the README's "Creation" section. Ask: "Do you have the quantization script you used? If so, provide the path or paste it. Otherwise this section will be omitted from the README."

**Action:** Once the scheme is identified, read the corresponding sub-skill (see Step 2 table) and jump directly to its **Step 10** (Upload to Hugging Face). Execute Step 10 using the information gathered above.

### B) Verify-only (skip quantization, just run vLLM test)

The user has a quantized model and wants to **verify it loads and generates correctly with vLLM**.

**Trigger phrases:** "test", "verify", "run vllm", "check model", "vllm test", combined with a local model directory path.

**Required information:**
1. **SAVE_DIR** — path to the local quantized model directory
2. **TENSOR_PARALLEL_SIZE** — number of GPUs (infer from model size or ask)

**Action:** Read any sub-skill (they all share the same Steps 8–9) and jump directly to **Step 8** (Write and run vLLM verification test), then **Step 9** (Validate results).

### C) Full quantization workflow (default)

The user wants to quantize a model from scratch. Proceed to **Step 1** below.

## Step 1 — Determine the quantization scheme

Analyze the user's request for explicit or implicit scheme indicators. Use the table below to classify:

| Scheme   | Trigger keywords / phrases | Hardware target |
|----------|---------------------------|-----------------|
| **FP8**  | `fp8`, `FP8_DYNAMIC`, `FP8_BLOCK`, `MXFP8`, `w8a8 fp8`, `float8` | NVIDIA Ampere+ (FP8_DYNAMIC), Hopper/Blackwell (FP8_BLOCK), AMD MI300X (MXFP8) |
| **INT**  | `int8`, `int4`, `W8A8` (integer context), `W4A16`, `GPTQ`, `SmoothQuant`, `integer quantization` | NVIDIA Turing+ (W8A8), Ampere+ (W4A16) |
| **NVFP4** | `nvfp4`, `NVFP4`, `nv fp4`, `fp4 nvidia`, `fp4 hopper`, `fp4 blackwell`, `h100 fp4` | NVIDIA H100 / Blackwell (sm90+) |
| **MXFP4** | `mxfp4`, `MXFP4`, `mx fp4`, `fp4 amd`, `mi300x quantization` | AMD MI300X |

### If one scheme is clear from the request:

Proceed directly to **Step 2** with the identified scheme.

### If multiple schemes are requested:

The user may request several quantization schemes at once (e.g., "quantize to fp8 and nvfp4", "compress this model in int8, fp8, and mxfp4"). This is fully supported.

1. Identify **all** requested schemes from the user's message.
2. Collect shared information **once** in Step 1 of the first sub-skill (MODEL_ID, model type, virtual environment, additional dependencies, HF upload preferences). Reuse these answers for all subsequent schemes — do not re-ask.
3. Execute each scheme **sequentially** by following Step 2 for each one. The environment setup (venv creation, dependency installation) only needs to happen once for the first scheme; subsequent schemes reuse the same environment.
4. Each scheme produces its own quantization script, its own output directory (with a scheme-specific suffix), and its own vLLM verification run.
5. If the user wants to upload to HF, each scheme gets its own repo (e.g., `RedHatAI/Model-FP8-dynamic`, `RedHatAI/Model-NVFP4`).

### If the scheme is ambiguous or not specified:

Ask the user to choose using `AskQuestion` (with `allow_multiple: true`) with the following options:

- **FP8** — 8-bit floating point. No calibration required (FP8_DYNAMIC / FP8_BLOCK) or calibration-free MX format (MXFP8). ~50% memory reduction. Broadest hardware support.
- **INT (W4A16)** — 4-bit integer weight-only (GPTQ). ~75% memory reduction. Requires calibration. Best compression-to-accuracy ratio.
- **INT (W8A8)** — 8-bit integer weights and activations (SmoothQuant + GPTQ). ~50% memory reduction. Requires calibration. Best INT8 quality.
- **NVFP4** — 4-bit floating point for NVIDIA H100/Blackwell. ~75% memory reduction. Requires calibration.
- **MXFP4** — 4-bit MX floating point for AMD MI300X. ~75% memory reduction. Calibration optional.

### Disambiguation hints

- If the user says "fp4" without specifying vendor, ask whether they target NVIDIA (→ NVFP4) or AMD (→ MXFP4).
- If the user says "W8A8" without specifying int vs fp8, ask whether they want integer (→ INT W8A8) or floating point (→ FP8 FP8_DYNAMIC).
- If the user simply says "compress" or "quantize" without a scheme, present all options (with multi-select enabled).

## Step 2 — Delegate to the scheme-specific skill

Once the scheme(s) are determined, read the corresponding sub-skill(s) and follow each from its Step 1 onward. The sub-skill files are located relative to this file:

| Scheme   | Sub-skill path |
|----------|---------------|
| FP8      | `fp8/SKILL.md` |
| INT      | `int/SKILL.md` |
| NVFP4    | `nvfp4/SKILL.md` |
| MXFP4    | `mxfp4/SKILL.md` |

**Action:** Resolve the sub-skill's absolute path by replacing the filename of this skill file's own path with the relative sub-skill path. For example, if this file was read from `/any/where/compress_models/SKILL.md`, then the FP8 sub-skill is at `/any/where/compress_models/fp8/SKILL.md`. Use the `Read` tool to read that file, then execute every step in that sub-skill exactly as written.

### When multiple schemes are requested:

1. Read **all** selected sub-skill files.
2. Run the **first** sub-skill end-to-end (Steps 1–10). This handles all shared setup (venv, dependencies, GPU check) and the first quantization + verification.
3. For each **subsequent** sub-skill, skip Steps 1–3 (information gathering, GPU check, environment setup — already done) and begin from Step 4 (check for existing examples) onward. Reuse the same virtual environment, MODEL_ID, model type, and GPU count.
4. After all schemes complete, report a summary of all results to the user (scheme, output directory, vLLM verification status, upload status).

## Quick Reference — Scheme Comparison

| Property | FP8_DYNAMIC | FP8_BLOCK | MXFP8 | W4A16 (INT) | W8A8 (INT) | NVFP4 | MXFP4 |
|----------|------------|-----------|-------|-------------|------------|-------|-------|
| Weight bits | 8 | 8 | 8 | 4 | 8 | 4 | 4 |
| Activation bits | 8 | 8 | 8 | 16 (fp16) | 8 | 4 | 4 |
| Calibration required | No | No | No | Yes | Yes | Yes | Optional |
| Memory reduction | ~50% | ~50% | ~50% | ~75% | ~50% | ~75% | ~75% |
| Min GPU | Ampere | Hopper | MI300X | Ampere | Turing | H100 (sm90+) | MI300X |
| Algorithm | QuantizationModifier | QuantizationModifier | QuantizationModifier | GPTQModifier | SmoothQuant + GPTQ | QuantizationModifier or GPTQModifier | QuantizationModifier or GPTQModifier |
