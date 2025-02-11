python queue_alpacaeval_vllm.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --queue-name oneshot-a100x4 \
  --project-name alexandre-debug \
  --task-name alpaca-eval \
  --max-instances 10 \
  --v2