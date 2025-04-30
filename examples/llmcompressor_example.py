from automation.tasks import LLMCompressorTask

recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      scheme: "W8A8"
      targets: "Linear"
      observer: $observer
"""

task = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="test_llmcompressor_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    recipe=recipe,
    text_samples=512,
    recipe_args={"observer": "mse"},
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()