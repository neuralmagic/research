from automation.tasks import LMEvalTask

task = LMEvalTask(
    project_name="alexandre_debug",
    task_name="test_leaderboard_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    limit=10,
    model_args="gpu_memory_utilization=0.4,enable_chunked_prefill=True",
    config="leaderboard",
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()