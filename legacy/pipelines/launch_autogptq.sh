python pipeline_autogptq.py \
  --project-name "LLM quantization - W4A16" \
  --task-prefix "Llama-3_1-8B-Instruct/calibration/256/8196" \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 256 \
  --max-seq-len 8196 \
  --disable-shuffle \
  --tags "Llama 3.1" "8B" "W4A16" "calibration"