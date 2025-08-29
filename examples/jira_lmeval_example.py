from automation.tasks import LMEvalTask
from itertools import product

"""
model_list = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8",
]

model_list = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8",
]
"""
"""
not intruct models
model_list = [
    "RedHatAI/Llama-3.2-1B-quantized.w8a8",
    "RedHatAI/Llama-3.2-3B-quantized.w8a8"
]
"""

versions = ["v1", "v2"]

def run_task(version, model):
    config = "openllm" if version == "v1" else "leaderboard"
    task = LMEvalTask(
        project_name="model_evals_jirai_1653",
        task_name=f"lmeval_task_{config}_{model.lower()}",
        model_id=f"{model}",
        config=f"{config}",
        model_args="gpu_memory_utilization=0.4,enable_chunked_prefill=True",
        batch_size="auto",
        branch="lmeval_update",
    )
    
    task.execute_remotely("oneshot-a100x1")

for version,model in product(versions, model_list):
    run_task(version,model)
