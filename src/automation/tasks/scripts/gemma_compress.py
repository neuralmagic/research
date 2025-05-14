import sys
import os
from automation.tasks import LLMCompressorTask
from transformers import Gemma3ForConditionalGeneration
from automation.datasets.calibration import load_calibration_dataset
from automation.datasets.utils import gemma_data_collator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# W8A8 Quantization
# recipe = """
# quant_stage:
#   quant_modifiers:
#     SmoothQuantModifier:
#       smoothing_strength: ${smoothing_strength}
#       mappings:
#         - [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"]
#         - [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
#         - [["re:.*down_proj"], "re:.*up_proj"]
#     GPTQModifier:
#       sequential_update: true
#       dampening_frac: ${dampening_frac}
#       ignore: ["re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"]
#       config_groups:
#         group_0:
#           targets: ["Linear"]
#           weights:
#             num_bits: 8
#             type: "int"
#             symmetric: true
#             strategy: "channel"
#             observer: "mse"
#           input_activations:
#             num_bits: 8
#             type: "int"
#             symmetric: true
#             strategy: "token"
#             dynamic: true
#             observer: "memoryless"
# """

# W4A16 Quantization 
recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"]
      sequential_targets: ["Gemma3DecoderLayer"]
      dampening_frac: 0.01
      config_groups:
        group0:
          targets: ["Linear"]
          weights:
            num_bits: 4
            type: "int"
            strategy: "group"
            group_size: 128
            symmetric: true
            actorder: "weight"
            observer: "mse"
"""

# Create and launch ClearML task
task = LLMCompressorTask(
    project_name="Gemma Quantization",
    max_memory_per_gpu="auto", 
    task_name="gemma-3-4b-it_w4a16_quant/vs_512/actorder_weight/ob_mse",
    model_id="google/gemma-3-4b-it",
    model_class="TraceableGemma3ForConditionalGeneration",
    recipe=recipe,
    recipe_args={"observer": "mse"},
    text_samples=512,
    vision_samples=0,
    max_seq_len=8192,
    save_directory="/network/shubhra/gemma_quantized/gemma-3-4b-it-w4a16_ts_512_actorder_weight_ob_mse",
    trust_remote_code=True,
    data_collator=gemma_data_collator,
    dataset_loader=load_calibration_dataset
)

task.execute_locally()
#task.execute_remotely("oneshot-a100x1")
