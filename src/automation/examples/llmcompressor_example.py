from automation.tasks import LLMCompressorTask

recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      scheme: "W8A16"
      targets: "Linear"
"""

task = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="test_llmcompressor_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    recipe=recipe,
)

task.execute_remotely("oneshot-a6000x1")
#task.execute_locally()