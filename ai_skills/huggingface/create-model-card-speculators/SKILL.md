---
name: create-model-card-speculators
description: >-
  Generate standardized HuggingFace model cards for RedHatAI speculator models
  (speculative-decoding draft models: EAGLE-3, DFlash, DFlash2, DSpark,
  P-EAGLE). Use when the user wants to create a model card, generate a README
  for a speculator or draft model, prepare a speculative-decoding model for
  HuggingFace upload, or mentions speculators, speculative decoding,
  speculative-config, or RedHatAI speculator publishing.
---

# Create Model Card for Speculator Models

Interactive, conversational workflow to produce a HuggingFace model card for a
RedHatAI speculator model (a small draft model that accelerates a larger
"verifier" model via speculative decoding in vLLM). Walk the user through each
phase, confirm auto-detected values, and let them skip or override any field.

## What a speculator model is

A speculator model is a small draft model (typically 1–5 transformer layers)
trained to predict the next tokens of a target "verifier" LLM. At inference
time, vLLM runs the draft model first and verifies its tokens in parallel with
the full model, trading a small extra compute cost for a large decode-throughput
speedup. The card documents four things:

1. **Model basics** — verifier, speculative decoding algorithm, draft model
   layer count + backbone.
2. **Creation** — training library, datasets, and the training scripts/commands.
3. **Deployment** — the vLLM serving command (verifier + speculative config).
4. **Evaluations** — token acceptance rates / acceptance lengths and performance
   benchmarks, plus optimization details.

## Platform Compatibility

This skill is designed to be portable across coding assistants and IDEs.

- Prefer capability-based behavior instead of platform-specific assumptions.
- If a platform has an authenticated Hugging Face integration (MCP/tool/plugin), use it.
- If not, fall back to `huggingface_hub` (Python), `hf` CLI, or manual user-provided files.
- Keep the model card generation flow identical regardless of integration method.

## Algorithm families

The speculative decoding algorithm is encoded in the model name
(`<verifier>-speculator.<algorithm>`). Known families:

| Suffix    | Display  | Draft architecture (config.json) |
|-----------|----------|----------------------------------|
| `eagle3`  | EAGLE-3  | `Eagle3Speculator`               |
| `eagle2`  | EAGLE-2  | `Eagle2Speculator`               |
| `eagle`   | EAGLE    | `EagleSpeculator`                |
| `dflash`  | DFlash   | `DFlashDraftModel`               |
| `dflash2` | DFlash2  | `DFlash2DraftModel`              |
| `dspark`  | DSpark   | `DSparkDraftModel`               |
| `peagle`  | P-EAGLE  | `PEagleDraftModel`               |

See `algorithm-reference.md` (same folder as this `SKILL.md`) for detection
rules, config fields, and per-family notes.

## Ground rule: never guess — prompt the user

The four `collect_*.py` scripts extract **only** what is already present in the
model's files (README + config.json). They report what is missing in a
`missing` list. When anything is missing:

- **Do not invent** evaluation results, training command code, dataset
  details, or serving flags.
- **Prompt the user** for each missing category before drafting. The prompt
  must cover, at minimum:
  - **Evaluation results**: acceptance rates, acceptance lengths, and
    performance benchmarking data (throughput, speedup vs baseline, ITL/TTFT)
  - **Evaluation tooling** (optional): which evaluation scripts were used and
    which vLLM version or branch the evaluation was run on
  - **Training command code** (prepare data / launch vLLM / launch training)
  - **Training code** (optional): which training code/scripts/repo were used
  - **Speculators version / branch / commit** (optional)
  - **Dataset details** (dataset ids, splits, regeneration notes)
  - **Serving command**, if not in the card
  For each item the user can: provide a file/path, paste the content directly,
  or skip it.
- If the user cannot provide something, omit the section (or the specific row)
  rather than fabricating content.

## Published card conventions (Red Hat AI)

These are the **defaults for the final README** unless the user asks otherwise.

### YAML frontmatter

- `library_name`: `speculators` (use `transformers` for EAGLE-3 models that
  ship a custom `eagle3.py`).
- `base_model`: list containing the verifier model id.
- `license`: from the verifier model (confirm with the user).
- `tags`: `speculative-decoding`, the algorithm tag (e.g. `dflash`, `eagle3`),
  `speculators`.

### Model Overview (top of the card)

Always start the body with a `## Model Overview` block in this exact shape:

```
## Model Overview
- **Model Architecture:** <VERIFIER_ARCHITECTURE>
  - **Input:** <MODEL_INPUT>
  - **Output:** <MODEL_OUTPUT>
- **Model Optimizations:**
  - **Speculative Decoding Algorithm:** <ALGORITHM_DISPLAY>
  - **Draft Model:** <NUM_DRAFT_LAYERS>-layer <BACKBONE> backbone (<DRAFT_ARCHITECTURE>)
- **Release Date:** <yyyy-mm-dd>
- **Version:** 1.0
- **Model Developers:** RedHatAI
```

- **Model Architecture** is the **verifier** model's architecture class
  (e.g. `Qwen3ForCausalLM`, `Gemma4ForConditionalGeneration`), resolved as:
  1. user-provided,
  2. `speculators_config.verifier.architectures[0]` in the speculator's
     `config.json`,
  3. the verifier's own `config.json` (fetch from the Hub),
  4. ask the user.
- **Input / Output** default to `Text` / `Text`; use `Text / Image` / `Text`
  for multimodal verifiers.
- Do **not** add weight/activation quantization lines — the optimization line
  for speculator cards is the **Speculative Decoding Algorithm** (plus the
  draft model line).
- **Draft Model** line: only the number of draft layers and the backbone
  family (e.g. `llama`, `qwen3`) from `transformer_layer_config`. Do **not**
  estimate or quote parameter counts.
- **Release date** is the date the model was **first published**, in **yyyy-mm-dd**.
  Resolve it from the HuggingFace API `createdAt` field of the model repo
  (reported by `collect_basics.py` as `created_at`); if the model is local or
  the field is unavailable, ask the user. Confirm the date with the user before
  using it.

### Verifier

- The verifier (target) model is either **user-defined** or fetched from the
  speculator's `config.json` (`speculators_config.verifier.name_or_path`).
- A user-provided verifier always wins (pass `--verifier` to
  `collect_basics.py`).
- Always present the detected verifier to the user and ask them to confirm or
  correct it.

### Training Details (Creation)

- One short paragraph: trained with the **Speculators** library on
  <datasets>; mention response regeneration when present; mention warm-start
  (e.g. "warm-started from a DFlash checkpoint"); mention compute sponsors when
  present.
- Put the full training commands (prepare data / launch vLLM / launch training)
  in a `<details><summary>Commands</summary>` block. Keep them as in the
  upstream card; remove hard-coded local paths only when obvious.
- State the **Speculators version / branch / commit** when known (from
  `config.json` `speculators_version`, version pins in the card, or the user),
  and which training code/scripts were used.
- If the training command code, training code, speculators version/branch, or
  dataset details are **not** in the model card, prompt the user for them
  (provide a script/path, paste the content, or skip) — exactly as is done for
  missing evaluation results. Never invent commands, versions, or dataset
  names. If the user has nothing, omit the part in question rather than
  fabricating content.

### Model Specifications

A two-column table: Base Model, Chat Template (verifier id + "use
`/chat/completions` endpoint"), Format (Safetensors), License (display form),
Validation Hardware.

### Deployment (vLLM)

- The serve command targets the **verifier** model with the speculator attached
  via `--speculative-config`:
  `vllm serve <verifier> --speculative-config '{"model": "<speculator>", "num_speculative_tokens": N, "method": "<algorithm>"}'`.
- Include the `pip install` line when the card requires a specific vLLM build
  (e.g. install from a PR branch). Keep it verbatim.
- Legacy flag style (`--spec-model ... --spec-tokens N --spec-method <algo>`)
  is acceptable when the upstream card uses it.
- Do not add `--tensor-parallel-size` / `-tp` unless the upstream card or the
  verifier's card prescribes it.
- Chat template note: use the verifier's chat template via the
  `/chat/completions` endpoint.

### Evaluation

- Always use the section heading **`## Evaluation`** — never
  "Acceptance Rates" or "Evaluations" — regardless of the table style.
- State the **evaluation scripts** used and the **vLLM version or branch** the
  model was served/benchmarked with, when known (from the card, e.g. a
  "vLLM version" bullet or a `pip install` line, or from the user).
- **Acceptance rates** (per-position acceptance % + average accepted length)
  and/or **acceptance lengths** (`k=1..N` per use case) are the primary
  metric — one table, one row per dataset.
- **Performance benchmarking**: include any benchmarking data that shows
  performance — throughput (tokens/s), speedup vs the baseline, ITL/TTFT,
  etc. — as tables and/or plots (images under `assets/`), with a `<details>`
  block holding the sampling config, hardware, library versions, and the
  benchmark commands (e.g. `guidellm benchmark`).
- For EAGLE-3-style cards this means: use-case table (dataset + sample count),
  acceptance lengths, and the performance plots + `<details>` block.
- Do **not** report quantization-style accuracy/recovery tables — speculator
  quality is measured by acceptance rates and throughput, not task accuracy.
- If evaluation results are not in the card, do not guess them: ask the user
  for the results (path or paste) or skip the section.

### Optimization details

The "optimization" of a speculator card is speculative decoding itself. Keep
the description short and factual: a small <N>-layer <backbone> draft model
proposes up to <K> tokens per step (from the proposal config), and the average
accepted length from the evaluation table quantifies the expected speedup.

### References

Paper links for the algorithm (e.g. EAGLE-3: arXiv:2503.01840, DFlash:
arXiv:2602.06036, P-EAGLE: arXiv:2602.01469). See `algorithm-reference.md`.

## Phase 1: Gather Model Information

1. **Check Hugging Face integration availability early**:
   - Verify whether an authenticated Hugging Face integration is available
     before collecting model details (MCP/connector, CLI auth, or token-based
     API access).
   - If setup is not possible, continue with local/manual fallbacks
     (user-provided files, local README generation).

2. Ask the user for the **model path** (local folder or HuggingFace model id).

3. **Run `collect_basics.py`** (same folder as this `SKILL.md`):
   ```
   python collect_basics.py <model-path-or-id>
   ```
   - It reports the draft/verifier architectures, algorithm, verifier (from
     `config.json`), draft model layer count + backbone, speculators version,
     and proposal settings.
   - If the user specifies the verifier explicitly, pass
     `--verifier <id>` — this overrides the config value.
   - **Never estimate draft parameters** — the output intentionally reports
     only layer count and backbone family.

4. **Fetch the verifier model's README/config** (best available integration)
   to confirm the verifier's architecture class and deployment hints (vLLM
   flags, chat template, modalities).

5. Present a summary and ask the user to confirm:

   > Here is what I found:
   > - **Verifier**: Qwen/Qwen3-8B (Qwen3ForCausalLM)
   > - **Algorithm**: DFlash
   > - **Draft model**: 5-layer qwen3 backbone (DFlashDraftModel)
   > - **Speculative tokens**: 7
   >
   > Does this look correct?

## Phase 2: Collect Creation Details

6. **Run `collect_creation.py`**:
   ```
   python collect_creation.py <model-path-or-id>
   ```
   - Reports the training library, datasets (with splits), data-preparation
     notes, warm-start, sponsors, and the training commands — all from the
     card.
   - Present the datasets and commands; confirm with the user.

7. **Prompt for missing creation details** (same rule as missing evaluation
   results in Phase 4):
   - Check the `missing` list from the script output.
   - If the **training command code** (prepare data / launch vLLM / launch
     training) is not in the card, prompt the user for it: provide a script
     path, paste the commands, or skip.
   - If the **training code** (scripts/repo used) or the **Speculators
     version / branch / commit** is not in the card, ask the user (optional —
     they may skip): which training code and which speculators version/branch
     they used. When provided, mention them in the Training Details paragraph.
   - If the **dataset details** (dataset ids, splits, regeneration notes) are
     not in the card, prompt the user for them: provide the dataset id(s) and
     splits, paste the details, or skip.
   - Do not proceed to the draft with these silently omitted — every missing
     item must have been explicitly provided or explicitly skipped by the user.

## Phase 3: Deployment

8. **Run `collect_deployment.py`**:
   ```
   python collect_deployment.py <model-path-or-id>
   ```
   - Reports the vLLM install line (if any), the `vllm serve` command, the
     parsed speculative config, and the model specifications table.
   - If the serve command or speculative config is missing, prompt the user
     for the serving command; they may skip.
   - Confirm the serve command, especially `num_speculative_tokens` and
     `method`.

## Phase 4: Evaluations

9. **Run `collect_evaluations.py`**:
   ```
   python collect_evaluations.py <model-path-or-id>
   ```
   - Reports acceptance rate tables, acceptance length tables, use cases,
     performance benchmarking data (throughput/speedup tables, plot images,
     hardware, sampling config, versions, commands), and references — all
     from the card.
   - Present the parsed tables for confirmation.

10. **If the user has evaluation results not in the card** (e.g. acceptance
    rates from a fresh run): ask for the path or have them paste the tables,
    and use those values. Never hand-fill numbers that have no source.
    - If the card has **no performance benchmarking data** (throughput /
      speedup tables or plots), also ask the user for it: provide a file/path,
      paste the tables, or skip.

11. **Ask about evaluation tooling** (optional — the user may skip): which
    **evaluation scripts** were used, and which **vLLM version or branch** the
    model was served/benchmarked with. When provided, state them in the
    Evaluation section (e.g. a short line under the heading, or inside the
    `<details>` block for GuideLLM-style performance sections).

## Phase 5: Generate Draft

12. **Read the model card template** from `template.md` (same folder as this
    `SKILL.md`). Fill every placeholder from the collected data. When in doubt,
    mirror the structure of a published card in the
    [RedHatAI speculator collection](https://huggingface.co/collections/RedHatAI/speculator-models).

13. **Present the complete draft** as a markdown code block for review.

## Phase 6: Iterate

14. Ask the user for feedback (wording, sections, scores, YAML header); apply
    changes; re-show the draft. Repeat until the user says it is final.

## Phase 7: Save and Upload

15. **Save the model card**:
    - Local folder: write `README.md` into that folder.
    - HuggingFace id: write `model_cards/<model_name>_README.md` locally and
      tell the user where it is.

16. **HuggingFace upload** (only if Hub access is available):
    - Ask the user before uploading; upload the README, and any benchmark
      images under `assets/` that the card references.
    - If no automated upload path is available, provide manual commands:
      `hf upload <repo_id> README.md README.md`.

## Integration Notes (Optional, Platform-Specific)

- **CLI fallback**: authenticate with `hf auth login`; upload a single file
  with `hf upload <repo_id> README.md README.md`.
- **Python fallback**: use `huggingface_hub` for `config.json` / `README.md`
  reads and uploads.

## Style Guidelines

- Be conversational and helpful throughout.
- Always confirm auto-detected values (verifier, algorithm, tokens, tables)
  before using them.
- Never block on missing information — offer to skip any field.
- Never guess evaluation results or creation details — prompt the user.
- When presenting the draft, show the full markdown so the user can review it.
- When asking questions, group related items together to avoid a long
  back-and-forth.
