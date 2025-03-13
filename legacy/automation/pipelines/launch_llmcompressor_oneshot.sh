python pipeline_llmcompressor_oneshot.py \
  --project-name "LLM quantization - W8" \
  --task-prefix "Llama-3_1-8B-Instruct/RTN/calibration/256/8196" \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 256 \
  --max-seq-len 8196 \
  --tags "Llama 3.1" "8B" "RTN" "calibration" \
  --recipe /network/alexandre/quantization/recipe_w8a16_rtn.yaml \
  --benchmark-tasks "winogrande" "arc_challenge_llama_3.1_instruct"