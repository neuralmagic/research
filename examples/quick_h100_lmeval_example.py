from automation.tasks import LMEvalTask
from itertools import product

model_queue_list_dict = [
    {"RedHatAI/Qwen3-0.6B-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Qwen3-1.7B-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Qwen3-4B-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Qwen3-8B-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Qwen3-14B-FP8-dynamic":"oneshot-h100x1"},


    {"RedHatAI/Qwen3-32B-FP8-dynamic":"oneshot-h100x2"},

    {"RedHatAI/Qwen3-30B-A3B-FP8-dynamic":"oneshot-h100x2"},

    {"RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic":"oneshot-h100x1"},
     
    {"RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic":"oneshot-h100x1"},

    {"RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": "oneshot-h100x4"},

    {"RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": "oneshot-h100x8"},

    {"RedHatAI/phi-4-FP8-dynamic" :"oneshot-h100x1"},

    {"RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic" :"oneshot-h100x2"},

    {"RedHatAI/gemma-2-2b-it-FP8" :"oneshot-h100x1"},

]

"""
model_queue_list_dict = [
    {"RedHatAI/Qwen3-0.6B-FP8-dynamic":"oneshot-h100x1"},
]
"""

versions = ["v1"]
#versions = ["v1", "v2"]

def run_task(version, model_queue_dict):
    model, queue =  model_queue_dict.popitem()
    config = "openllm" if version == "v1" else "leaderboard"
    task = LMEvalTask(
        project_name="h100_debugging",
        #packages = ["huggingface-hub==0.34.3"],
        task_name=f"1_config_withhub_ablation_test_lmeval_task_{config}_{model.lower()}",
        model_id=f"{model}",
        config=f"{config}",
        model_args="gpu_memory_utilization=0.4,enable_chunked_prefill=True",
        #tasks=["gsm8k"],
        #tasks = ["arc_challenge","hellaswag","winogrande","gsm8k","truthfulqa"],
        #model_args="add_bos_token=True,gpu_memory_utilization=0.4,enable_chunked_prefill=True,max_model_len=4096",
        batch_size="auto",
        branch="lmeval_update",
        limit=10,
    )
    
    task.execute_remotely(queue)
    import time
    time.sleep(300)

for version,model_queue_dict in product(versions, model_queue_list_dict):
    run_task(version,model_queue_dict)
