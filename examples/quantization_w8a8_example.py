from automation.tasks import LLMCompressorTask

task = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="test_w8a8_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    config="quantization_w8a8",
    recipe_args={"dampening_frac": "0.1"},
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()