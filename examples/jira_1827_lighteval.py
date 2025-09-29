from automation.tasks import LightEvalTask
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


model_args="""
model_parameters:
  max_model_length: 40960
  generation_parameters:
    max_new_tokens: 32000
    temperature: 0.6
    top_k: 20
    min_p: 0.0
    top_p: 0.95
"""

def run_task(model, queue, config, model_args ):
    print(model, queue, config, model_args)
    print("\n")

    model_name = model.lower().split("/")[1]

    task = LightEvalTask(
        project_name=f"jira_1827_lighteval/{model_name}",
        task_name=f"{model_name}_{config}_task",
        model_id=f"{model}",
        config=f"{config}",
        model_args=model_args,
        branch="ablation-lighteval_standards",
    )
    task.execute_remotely(queue)

config_list =["aime2024", "aime2025", "math500", "gpqa_diamond"]
for model_queue_dict in model_queue_list_dict:
    model_name, queue =  model_queue_dict.popitem()
    for model, config  in product([model_name], config_list):
        print(model, queue, config )
        print("\n")
        run_task(model, queue, config, model_args )
