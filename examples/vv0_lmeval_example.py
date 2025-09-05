from automation.tasks import LMEvalTask
from itertools import product

model_queue_list_dict = [
    {"Qwen/Qwen3-0.6B" :"oneshot-a100x1"},
    {"RedHatAI/Qwen3-0.6B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-1.7B":"oneshot-a100x1"},
    {"RedHatAI/Qwen3-1.7B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-4B":"oneshot-a100x1"},
    {"RedHatAI/Qwen3-4B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-8B":"oneshot-a100x1"},
    {"RedHatAI/Qwen3-8B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-14B":"oneshot-a100x1"},
    {"RedHatAI/Qwen3-14B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-32B":"oneshot-a100x2"},
    {"RedHatAI/Qwen3-32B-quantized.w4a16":"oneshot-a100x2"},

    {"Qwen/Qwen3-30B-A3B":"oneshot-a100x2"},
    {"RedHatAI/Qwen3-30B-A3B-quantized.w4a16":"oneshot-a100x2"},

    {"meta-llama/Llama-3.1-8B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16":"oneshot-a100x1"},
    {"RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8":"oneshot-a100x1"},
     
    {"meta-llama/Llama-3.2-1B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8":"oneshot-a100x1"},

    {"meta-llama/Llama-3.2-3B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8":"oneshot-a100x1"},

    {"meta-llama/Llama-3.3-70B-Instruct": "oneshot-a100x4"},
    {"RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16": "oneshot-a100x4"},
    {"RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8": "oneshot-a100x4"},

    {"meta-llama/Llama-4-Scout-17B-16E": "oneshot-a100x8"},
    {"RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16": "oneshot-a100x8"},

    {"microsoft/phi-4" :"oneshot-a100x1"},
    {"RedHatAI/phi-4-quantized.w4a16" :"oneshot-a100x1"},
    {"RedHatAI/phi-4-quantized.w8a8" :"oneshot-a100x1"},

    {"mistralai/Mistral-Small-3.1-24B-Instruct-2503" :"oneshot-a100x2"},
    {"RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8" :"oneshot-a100x2"},
    {"RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16" :"oneshot-a100x2"},

    {"google/gemma-2b-it" :"oneshot-a100x1"},
    {"RedHatAI/gemma-2-2b-it-quantized.w8a8" :"oneshot-a100x1"},
    {"RedHatAI/gemma-2-2b-it-quantized.w8a16" :"oneshot-a100x1"},

]

versions = ["v1"]
#versions = ["v1", "v2"]

def run_task(version, model_queue_dict):
    model, queue =  model_queue_dict.popitem()
    config = "openllm" if version == "v1" else "leaderboard"
    task = LMEvalTask(
        project_name="simple_debug",
        task_name=f"1_ablation_test_lmeval_task_{config}_{model.lower()}",
        model_id=f"{model}",
        config=f"{config}",
        model_args="gpu_memory_utilization=0.6,enable_chunked_prefill=True",
        #tasks=["gsm8k","mmlu"],
        #model_args="add_bos_token=True,gpu_memory_utilization=0.4,enable_chunked_prefill=True,max_model_len=4096",
        batch_size="auto",
        branch="lmeval_update",
        limit=10,
    )
    
    task.execute_remotely(queue)

for version,model_queue_dict in product(versions, model_queue_list_dict):
    run_task(version,model_queue_dict)
