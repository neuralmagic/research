# Parsing recipe.yaml for Quantization Format Detection

Reference for interpreting `recipe.yaml` files produced by
[LLM Compressor](https://github.com/vllm-project/llm-compressor).

## File Structure

The recipe lives at `recipe.yaml` in the model directory. Structure:

```yaml
default_stage:
  default_modifiers:
    SmoothQuantModifier:       # Optional — present in NVFP4
      smoothing_strength: ...
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            num_bits: <int>
            type: <"int" | "float">
            strategy: <str>
            group_size: <int | null>
            block_structure: <null | list>
            dynamic: <bool>
          input_activations:   # null or absent for weight-only
            num_bits: <int>
            type: <"int" | "float">
            dynamic: <"local" | bool>
          output_activations: null
      ignore: [...]
```

## Format Detection Table

Read `QuantizationModifier.config_groups`. For the primary group (usually `group_0`):

| weights.num_bits | weights.type | input_activations          | Extra signals                          | Format             | Tag    | Suffix               | Bits | Reduction |
|------------------|--------------|----------------------------|----------------------------------------|--------------------|--------|----------------------|------|-----------|
| 8                | `int`        | `.num_bits=8, .type=int`   | —                                      | **INT W8A8**       | `int8` | `-quantized.w8a8`    | 8    | ~50%      |
| 4                | `int`        | `null` or absent           | —                                      | **INT W4A16**      | `int4` | `-quantized.w4a16`   | 4    | ~75%      |
| 8                | `float`      | `.num_bits=8, .type=float` | `block_structure` is `null`            | **FP8 W8A8**       | `fp8`  | `-FP8-dynamic`       | 8    | ~50%      |
| 8                | `float`      | `.num_bits=8, .type=float` | `block_structure` is set (e.g. `[128]`)| **FP8 W8A8 block** | `fp8`  | `-FP8-block`         | 8    | ~50%      |
| 4                | `float`      | `.num_bits=4, .type=float` | `SmoothQuantModifier` present          | **NVFP4 (W4A4)**   | `fp4`  | `-NVFP4`             | 4    | ~75%      |
| varies           | varies       | varies                     | Multiple `config_groups` with different bit widths | **Mixed precision** | varies | adapted to mix | avg  | varies    |

## Detection Steps

1. Look under `default_stage.default_modifiers.QuantizationModifier.config_groups`.
2. If there is only **one group** (e.g. `group_0`):
   - Read `weights.num_bits` and `weights.type`.
   - Read `input_activations` — if it is `null` or the key is absent, this is
     weight-only quantization.
   - Check `weights.block_structure` and `input_activations.block_structure` for
     block-scaling variants.
   - Check if `SmoothQuantModifier` is present among the modifiers.
   - Match against the table above.
3. If there are **multiple groups** with different `num_bits` values, this is
   **mixed precision**. Note the range of bit widths and compute the average if
   possible.

## Weight and Activation Descriptions

Use these for the Model Overview section of the model card:

| Format             | Weight quantization | Activation quantization |
|--------------------|--------------------|-----------------------|
| **INT W8A8**       | INT8               | INT8                  |
| **INT W4A16**      | INT4               | None                  |
| **FP8 W8A8**       | FP8                | FP8                   |
| **FP8 W8A8 block** | FP8                | FP8                   |
| **NVFP4 (W4A4)**   | FP4                | FP4                   |
| **Mixed precision** | Mixed (describe per-layer breakdown) | varies |

## Model Name Suffix

Apply the suffix from the detection table to the base model name:

```
<OriginalModelName><Suffix>
```

Examples:
- `Llama-3-8B-Instruct` + NVFP4 = `Llama-3-8B-Instruct-NVFP4`
- `Mistral-7B-v0.3` + FP8 W8A8 = `Mistral-7B-v0.3-FP8-dynamic`
- `Qwen2-72B-Instruct` + INT W4A16 = `Qwen2-72B-Instruct-quantized.w4a16`

## Example: NVFP4 Recipe

```yaml
default_stage:
  default_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.9
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            num_bits: 4
            type: float
            strategy: tensor_group
            group_size: 16
          input_activations:
            num_bits: 4
            type: float
            dynamic: local
      ignore: ['re:.*lm_head.*']
```

Detection:
- `weights`: num_bits=4, type=float
- `input_activations`: num_bits=4, type=float
- `SmoothQuantModifier` is present
- Result: **NVFP4 (W4A4)**, tag=`fp4`, suffix=`-NVFP4`, 4 bits, ~75% reduction
