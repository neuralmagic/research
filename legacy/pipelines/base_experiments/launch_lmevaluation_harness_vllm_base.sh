python lm_evaluation_harness_vllm_base.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --lm_eval \
  --tasks gsm8k \
  --model_args "dtype=auto,max_model_len=8192" \
  --batch_size "auto"