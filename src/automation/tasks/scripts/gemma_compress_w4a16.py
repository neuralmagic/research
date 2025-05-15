import sys
import os
from automation.tasks import LLMCompressorTask
from automation.datasets.calibration import load_calibration_dataset
from automation.datasets.utils import gemma_data_collator

recipe_template = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ["re:.*lm_head.*", "re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"]
      sequential_targets: ["Gemma3DecoderLayer"]
      dampening_frac: {dampening_frac}
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

# Sweep variables
num_samples = [512]
damp_frac = [0.05, 0.07]

models = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
)

for model in models:
    model_slug = model.split("/")[-1]
    for text_samples in num_samples:
        for dampening_frac in damp_frac:
            df_str = f"df_{dampening_frac}"
            ts_str = f"ts_{text_samples}"

            task_name = f"{model_slug}_w4a16_quant/{ts_str}/{df_str}/actorder_weight_ob_mse"
            save_directory = f"/network/shubhra/gemma_quantized/{model_slug}_w4a16_{ts_str}_{df_str}_actorder_weight_ob_mse"

            # Format recipe with current dampening value
            recipe = recipe_template.format(dampening_frac=dampening_frac)

            task = LLMCompressorTask(
                project_name="Gemma Quantization",
                max_memory_per_gpu="auto",
                task_name=task_name,
                model_id=model,
                model_class="TraceableGemma3ForConditionalGeneration",
                recipe=recipe,
                recipe_args=None,
                text_samples=text_samples,
                vision_samples=0,
                max_seq_len=8192,
                save_directory=save_directory,
                trust_remote_code=True,
                data_collator=gemma_data_collator,
                dataset_loader=load_calibration_dataset,
            )

            # task.execute_locally()
            task.execute_remotely("oneshot-a100x1")
