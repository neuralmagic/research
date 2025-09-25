
MAPPINGS_PER_MODEL_CONFIG = {
    "llama": [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
        [["re:.*down_proj"], "re:.*up_proj"],
    ],
    "phi3": [
        [["re:.*qkv_proj"], "re:.*input_layernorm"],
        [["re:.*gate_up_proj"], "re:.*post_attention_layernorm"],
    ],
    "apertus": [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*attention_layernorm"],
        [["re:.*up_proj"], "re:.*feedforward_layernorm"],
    ],
}
