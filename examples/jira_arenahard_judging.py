from automation.tasks import ArenaHardJudgeTask

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
]






# TODO:
model_queue_list_dict = [
    {"RedHatAI/Qwen3-1.7B-quantized.w4a16":"oneshot-a100x1"},
    {"Qwen/Qwen3-8B":"oneshot-a100x1"},
    {"Qwen/Qwen3-14B":"oneshot-a100x1"},
    {"Qwen/Qwen3-32B":"oneshot-a100x2"},
    {"RedHatAI/Qwen3-32B-quantized.w4a16":"oneshot-a100x2"},
    {"meta-llama/Llama-3.3-70B-Instruct": "oneshot-a100x4"},
    {"RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16": "oneshot-a100x4"},
    {"meta-llama/Llama-4-Scout-17B-16E": "oneshot-a100x8"},
    {"RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16": "oneshot-a100x8"},
]

model_queue_list_dict = [
    #{"Qwen/Qwen3-0.6B" :"oneshot-a100x1"},
    #{"RedHatAI/Qwen3-0.6B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-1.7B":"oneshot-a100x1"},

    {"Qwen/Qwen3-4B":"oneshot-a100x1"},
    {"RedHatAI/Qwen3-4B-quantized.w4a16":"oneshot-a100x1"},

    {"RedHatAI/Qwen3-8B-quantized.w4a16":"oneshot-a100x1"},

    {"RedHatAI/Qwen3-14B-quantized.w4a16":"oneshot-a100x1"},

    {"Qwen/Qwen3-30B-A3B":"oneshot-a100x2"},
    {"RedHatAI/Qwen3-30B-A3B-quantized.w4a16":"oneshot-a100x2"},

    {"meta-llama/Llama-3.1-8B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16":"oneshot-a100x1"},
    {"RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8":"oneshot-a100x1"},
     
    {"meta-llama/Llama-3.2-1B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8":"oneshot-a100x1"},

    {"meta-llama/Llama-3.2-3B-Instruct":"oneshot-a100x1"},
    {"RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8":"oneshot-a100x1"},

    {"RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8": "oneshot-a100x4"},

    {"microsoft/phi-4" :"oneshot-a100x1"},
    {"RedHatAI/phi-4-quantized.w4a16" :"oneshot-a100x1"},
    {"RedHatAI/phi-4-quantized.w8a8" :"oneshot-a100x1"},

    {"mistralai/Mistral-Small-3.1-24B-Instruct-2503" :"oneshot-a100x2"},
    {"RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8" :"oneshot-a100x2"},
    {"RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16" :"oneshot-a100x2"},
]
model_queue_list_dict = [
    {"Qwen/Qwen3-0.6B" :"oneshot-a100x1"},
    {"RedHatAI/Qwen3-0.6B-quantized.w4a16":"oneshot-a100x1"},
]

judgement_model_dict = {"model": "openai/gpt-oss-120b", "queue": "oneshot-a100x4" }

def run_task(model_queue_dict):
    answer_model, _ =  model_queue_dict.popitem()
    judgement_model = judgement_model_dict["model"]
    queue = judgement_model_dict["queue"]

    task = ArenaHardJudgeTask(
        project_name="jira_arenahard_judging",
        task_name = f"judge_{answer_model.lower()}_task",
        packages = ["huggingface-hub==0.34.3", "triton==3.3.1", "vllm==0.10.1.1"],
        answer_project_name = "jira_arenahard_generation",
        answer_task_name = f"generate_task_{answer_model.lower()}",
        judgement_model = judgement_model,
        #question_size = "small",
        rate_type="throughput",
        backend="aiohttp_server",
        target="http://localhost:8000/v1",
        bench_name = "arena-hard-v2.0",
        branch = "arena_upgrade",
        vllm_kwargs={"enable-chunked-prefill": True},
        max_tokens = 16000, 
    )
    
    task.execute_remotely(queue)

for model_queue_dict in model_queue_list_dict:
    run_task(model_queue_dict)

#task.execute_locally()
