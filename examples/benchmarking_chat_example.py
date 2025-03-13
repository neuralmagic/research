from automation.tasks import GuideLLMTask

task = GuideLLMTask(
    project_name="alexandre_debug",
    task_name="test_benchmarking_chat_task",
    model="meta-llama/Llama-3.2-1B-Instruct",
    rate_type="throughput",
    max_seconds=30,
    config="benchmarking_chat"
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()