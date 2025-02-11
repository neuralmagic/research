recipe=$(cat <<'EOF'
quant_stage:
  quant_modifiers:
    GPTQModifier:
      sequential_update: true
      dampening_frac: 0.0
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: "int"
            symmetric: true
            strategy: "weight"
            group_size: 128
            actorder: "weight"
            observer: "mse"
          targets: ["Linear"]
EOF
)

python pipeline_llmcompressor_base.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --llmcompressor-queue "oneshot-a5000x1" \
  --num-samples 256 \
  --max-seq-len 8192 \
  --recipe "${recipe}" \
  --evaluation_kwargs \
  --lm_eval \
  --tasks "gsm8k" \
  --num_fewshot 5 \
  --batch_size "auto" \
  --model_args "dtype=auto,max_model_len=8192" \
  --monitor_metrics "Summary" "gsm8k/5shot/exact_match,strict-match" \
  --lm_eval \
  --tasks "winogrande" \
  --batch_size "auto" \
  --num_fewshot 5 \
  --model_args "dtype=auto,max_model_len=8192" \
  --monitor_metrics "Summary" "winogrande/5shot/acc,none"
