
source ~/environments/clearml/bin/activate
python /network/alexandre/automation/queue_llmcompressor_oneshot.py \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --recipe "/network/alexandre/quantization/recipe_w8a16_rtn.yaml" \
  --tags "llm-compressor" "oneshot"

