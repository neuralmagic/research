from automation.tasks import DebugTask

task = DebugTask(
    project_name="alexandre-debug", 
    task_name="debug-task",
    time_in_sec=1200,
    docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-cuda12_5:latest",
    packages=["vllm", "git+https://github.com/EleutherAI/lm-evaluation-harness.git", "numpy==2.1"],
)

task.execute_remotely("oneshot-a100x1")