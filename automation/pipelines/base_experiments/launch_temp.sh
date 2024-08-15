python llmcompressor_oneshot_base.py \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --recipe "/network/alexandre/quantization/recipe_w8a16_sequential.yaml" \
  --tags "llm-compressor" "oneshot" \
  --max-memory-per-gpu 20

