from automation.tasks import LLMCompressorTask

task = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="test_w4a16_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    config="quantization_w4a16",
    recipe_args={"dampening_frac": "0.1"},
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()