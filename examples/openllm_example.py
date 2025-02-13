from automation.standards import OpenLLMTask

task = OpenLLMTask(
    project_name="alexandre_debug",
    task_name="test_openllm_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
)

task.execute_remotely("oneshot-a6000x1")
#task.execute_locally()