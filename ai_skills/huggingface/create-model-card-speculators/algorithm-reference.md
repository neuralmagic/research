# Speculator Algorithm Reference

Reference for identifying the speculative decoding algorithm and model details
from a speculator model's `config.json` / README. Mirrors the role of
`recipe-parsing.md` in the compression skill.

## What a speculator model is

A small draft model (typically 1–5 transformer layers) trained to predict the
next tokens of a larger target model (the **verifier**). vLLM runs the draft
model first and verifies its tokens in parallel with the full model, trading
small extra compute for large decode-throughput speedup. Quality is measured
by **token acceptance rates / accepted length** and throughput benchmarks —
not by task accuracy.

## Algorithm detection

Order of precedence:

1. `config.json` → `speculators_config.algorithm`
2. `config.json` → `speculators_model_type`
3. Model name suffix after `-speculator.` (e.g. `Qwen3-8B-speculator.dflash` → `dflash`)
4. README tags / "Speculative Decoding Algorithm" line

## Known algorithm families

| Suffix   | Display  | Draft architecture (config.json) | Config markers                                                        | Paper                    |
|----------|----------|----------------------------------|-----------------------------------------------------------------------|--------------------------|
| `eagle3` | EAGLE-3  | `Eagle3Speculator`               | single draft layer, custom `eagle3.py`, `library_name: transformers` | [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) |
| `eagle2` | EAGLE-2  | `Eagle2Speculator`               | —                                                                     | [arXiv:2408.04488](https://arxiv.org/abs/2408.04488) |
| `eagle`  | EAGLE    | `EagleSpeculator`                | —                                                                     | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| `dflash` | DFlash   | `DFlashDraftModel`               | `block_size`, `mask_token_id`, `aux_hidden_state_layer_ids`          | [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) |
| `dflash2`| DFlash2  | `DFlash2DraftModel`              | `block_size`, `conv_kernel_size`, `conv_group_size`, `selector_rank` | —                        |
| `dspark` | DSpark   | `DSparkDraftModel`               | `markov_rank`, `markov_head_type`, `enable_confidence_head`, `sliding_window` | —                    |
| `peagle` | P-EAGLE  | `PEagleDraftModel`               | `num_depths`, `down_sample_ratio`, `eagle_aux_hidden_state_layer_ids`| [arXiv:2602.01469](https://arxiv.org/abs/2602.01469) |

Notes:

- **DFlash** — parallel block drafting: the draft produces a block of tokens
  per step (diffusion-style with `mask_token_id`).
- **DFlash2** — extends DFlash with local dynamic convolutions and a
  candidate selector.
- **DSpark** — typically warm-started from a DFlash checkpoint; adds a Markov
  head and confidence head, sliding-window attention.
- **EAGLE-3** — autoregressive draft conditioned on hidden states from
  multiple verifier layers; ships its own `eagle3.py` and uses
  `library_name: transformers`.
- **P-EAGLE** — parallel-drafting EAGLE variant with `num_depths` depth and
  downsampling.

## Config structure (speculators format)

```json
{
  "architectures": ["DFlashDraftModel"],
  "speculators_config": {
    "algorithm": "dflash",
    "default_proposal_method": "greedy",
    "proposal_methods": [
      {"accept_tolerance": 0.0, "proposal_type": "greedy",
       "speculative_tokens": 7, "verifier_accept_k": 1}
    ],
    "verifier": {
      "architectures": ["Qwen3ForCausalLM"],
      "name_or_path": "Qwen/Qwen3-8B"
    }
  },
  "speculators_model_type": "dflash",
  "speculators_version": "0.5.0",
  "transformer_layer_config": {
    "model_type": "qwen3",
    "num_hidden_layers": 5,
    "hidden_size": 4096,
    "vocab_size": 151936
  }
}
```

Field guide:

| Field | Meaning |
|-------|---------|
| `speculators_config.verifier.name_or_path` | Verifier (target) model — the card's primary "base" model |
| `speculators_config.verifier.architectures` | Verifier architecture class (may be empty — then fetch the verifier's own `config.json`) |
| `speculators_config.proposal_methods[*].speculative_tokens` | Default `num_speculative_tokens` for serving |
| `speculators_model_type` / `speculators_config.algorithm` | Algorithm id |
| `speculators_version` | Version of the speculators library that produced the model |
| `transformer_layer_config.num_hidden_layers` | Number of layers in the draft model |
| `transformer_layer_config.model_type` | Draft backbone family (`llama`, `qwen3`, ...) |
| `draft_vocab_size` | Draft vocabulary (may be smaller than the full `vocab_size`) |
| `architectures[0]` | Draft model architecture class |

For the Model Overview section, report only the draft **layer count** and
**backbone family** — do not estimate or quote parameter counts.

## Verifier resolution

1. User-provided (always wins).
2. `speculators_config.verifier.name_or_path` in the speculator's `config.json`.
3. Frontmatter `base_model` in the README.
4. Ask the user.

Verifier architecture (for the **Model Architecture** line):

1. `speculators_config.verifier.architectures[0]` when non-empty.
2. Fetch the verifier's own `config.json` → `architectures[0]`.
3. Ask the user.

## vLLM serving

Modern style (JSON `--speculative-config`, `method` = algorithm id):

```bash
vllm serve <verifier> \
  --speculative-config '{
    "model": "RedHatAI/<verifier>-speculator.<algorithm>",
    "num_speculative_tokens": 7,
    "method": "dflash"
  }'
```

Legacy flag style (older cards):

```bash
vllm serve <verifier> -tp 4 \
  --spec-model RedHatAI/<verifier>-speculator.<algorithm> \
  --spec-tokens 7 \
  --spec-method dspark
```

Some algorithms require a specific vLLM build — the card may include a
`pip install git+https://github.com/vllm-project/vllm.git@refs/pull/<N>/head`
line. Keep it verbatim when present.

## Evaluation styles

- **Per-position acceptance rates** (DFlash family): table of acceptance %
  per draft position (Pos 0/1..k) plus average accepted length, one row per
  dataset (HumanEval, math_reasoning, qa, ...).
- **Acceptance lengths** (EAGLE-3 family): accepted length at k=1..k=N per use
  case, plus use-case table (dataset + sample count) and GuideLLM performance
  plots with a `<details>` block (sampling config, hardware, vLLM/GuideLLM
  versions, `guidellm benchmark` command).
