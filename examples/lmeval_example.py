from automation.tasks import LMEvalTask

task = LMEvalTask(
    project_name="alexandre_debug",
    task_name="test_lmeval_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    tasks="gsm8k",
    model_args="dtype=auto,max_model_len=8192",
    batch_size="auto",    
)

#task.execute_remotely("oneshot-a100x1")
task.execute_remotely("remote-upgrade-default")
#task.execute_locally()
