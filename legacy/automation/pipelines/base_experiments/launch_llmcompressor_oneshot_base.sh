recipe=$(cat <<'EOF'
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      scheme: "W8A16"
      targets: "Linear"
EOF
)

python llmcompressor_oneshot_base.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --recipe "${recipe}" \
  --tags "llm-compressor" "oneshot"

