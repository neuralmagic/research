import sys
import os
from automation.tasks import LLMCompressorTask
from transformers import Gemma3ForConditionalGeneration
from automation.datasets.calibration import load_calibration_dataset
from automation.datasets.utils import gemma_data_collator

# W8A8 Quantization with SmoothQuant + GPTQ
recipe_template = """
quant_stage:
  quant_modifiers:
    SmoothQuantModifier:
      smoothing_strength: {smoothing_strength}
      mappings:
        - [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"]
        - [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
        - [["re:.*down_proj"], "re:.*up_proj"]
    GPTQModifier:
      sequential_update: true
      dampening_frac: {dampening_frac}
      ignore: ["re:.*lm_head.*", "re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"]
      config_groups:
        group_0:
          targets: ["Linear"]
          weights:
            num_bits: 8
            type: "int"
            symmetric: true
            strategy: "channel"
            observer: "mse"
          input_activations:
            num_bits: 8
            type: "int"
            symmetric: true
            strategy: "token"
            dynamic: true
            observer: "memoryless"
"""

# Sweep values
damp_frac = [0.01, 0.05]
smoothing_values = [0.7, 0.9]
text_sample_options = [512, 1024]

# Models to run
models = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
)

project = "Gemma Quantization"
max_seq_len = 8192
vision_samples = 0

for model_id in models:
    model_slug = model_id.split("/")[-1]
    for dampening_frac in damp_frac:
        for smoothing_strength in smoothing_values:
            for text_samples in text_sample_options:
                # Format recipe for current run
                recipe = recipe_template.format(
                    dampening_frac=dampening_frac,
                    smoothing_strength=smoothing_strength,
                )

                # Build name components
                df_str = f"df_{dampening_frac}"
                smoothing_str = f"smooth_{str(smoothing_strength).replace('.', '')}"
                ts_str = f"ts_{text_samples}"

                # Task name and output directory
                task_name = f"{model_slug}_w8a8_smoothquant/{ts_str}/{smoothing_str}/{df_str}"
                save_directory = f"/network/shubhra/gemma_quantized/{model_slug}_w8a8_sq_{ts_str}_{smoothing_str}_{df_str}"

                # Launch compression task
                task = LLMCompressorTask(
                    project_name=project,
                    max_memory_per_gpu="auto",
                    task_name=task_name,
                    model_id=model_id,
                    model_class="TraceableGemma3ForConditionalGeneration",
                    recipe=recipe,
                    recipe_args=None,
                    text_samples=text_samples,
                    vision_samples=vision_samples,
                    max_seq_len=max_seq_len,
                    save_directory=save_directory,
                    trust_remote_code=True,
                    data_collator=gemma_data_collator,
                    dataset_loader=load_calibration_dataset,
                )

                # task.execute_locally()
                task.execute_remotely("oneshot-a100x1")
