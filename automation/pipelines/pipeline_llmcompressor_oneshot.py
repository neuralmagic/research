from clearml.automation import PipelineController
from clearml import Task
import argparse

parser = argparse.ArgumentParser(description = "Apply recipe in one-shot")

parser.add_argument("--project-name", type=str)
parser.add_argument("--task-prefix", type=str)
parser.add_argument("--pipeline-name", type=str, default="llm-compressor_oneshot")
parser.add_argument("--model-id", type=str)
parser.add_argument("--recipe", type=str)
parser.add_argument("--oneshot-queue", type=str, default="oneshot-a100x1")
parser.add_argument("--evaluation-queue", type=str, default="oneshot-a100x1")
parser.add_argument("--save-dir", type=str, default="output")
parser.add_argument("--dataset", type=str, default="neuralmagic/LLM_compression_calibration")
parser.add_argument("--random-fraction", type=float, default=0.)
parser.add_argument("--disable-shuffle", action="store_true", default=False)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--tags", type=str, nargs="+", default=None)
parser.add_argument("--oneshot-packages", type=str, nargs="+", default=None)
parser.add_argument("--evaluation-packages", type=str, nargs="+", default=None)
parser.add_argument("--benchmark-tasks", type=str, nargs="+", default=["openllm"])
parser.add_argument("--engine", type=str, choices=["vllm", "hf", "sparseml"], default="vllm")
parser.add_argument("--build-vllm", action="store_true", default=False)
parser.add_argument("--batch-size", type=str, default="auto")

args = parser.parse_args()
args = vars(args)

project_name = args.pop("project_name")
task_prefix = args.pop("task_prefix")
pipeline_name = args.pop("pipeline_name")
args["packages"] = args.pop("oneshot_packages")
evaluation_packages = args.pop("evaluation_packages")

Task.force_store_standalone_script()

pipe = PipelineController(
    name=pipeline_name, project=project_name, version="0.0.1",target_project=project_name,
)

oneshot_task_id = Task.get_task(project_name="Automation",task_name="llmcompressor_oneshot", task_filter={'order_by': ["-last_update"]}).id

oneshot_step_name = f"{task_prefix}_llm_compressor"

if "single" in args["evaluation_queue"] or "x1" in args["evaluation_queue"]:
    num_gpus_evaluation = 1
elif "double" in args["evaluation_queue"] or "x2" in args["evaluation_queue"]:
    num_gpus_evaluation = 2
elif "quad" in args["evaluation_queue"] or "x4" in args["evaluation_queue"]:
    num_gpus_evaluation = 4
elif "octo" in args["evaluation_queue"] or "x8" in args["evaluation_queue"]:
    num_gpus_evaluation = 8

oneshot_override = {f"Args/{k}": v for k, v in args.items()}
pipe.add_step(
    name=oneshot_step_name,
    base_task_id=oneshot_task_id,
    execution_queue=args["oneshot_queue"],
    parameter_override=oneshot_override,
    monitor_models=["*"]
)

oneshot_model_id = f"${{{oneshot_step_name}.models.output.-1.id}}"

batch_size = args["batch_size"]
if batch_size == "auto":
    evalplus_batch_size = 1
else:
    batch_size = int(batch_size)
    evalplus_batch_size = batch_size

engine = args["engine"]
if engine == "vllm":
    lm_evaluation_harness_task_id = Task.get_task(project_name="Automation",task_name="lm_evaluation_harness_vllm", task_filter={'order_by': ["-last_update"]}).id
    lm_evaluation_override_engine = {
        "Args/build_vllm": args["build_vllm"],
        "Args/gpu_memory_utilization": 0.7,
        "Args/num_gpus": num_gpus_evaluation,
    }

    evalplus_task_id = Task.get_task(project_name="Automation",task_name="evalplus_vllm", task_filter={'order_by': ["-last_update"]}).id
    evalplus_override_engine = lm_evaluation_override_engine
else:
    if engine == "hf":
        lm_evaluation_harness_task_id = Task.get_task(project_name="Automation",task_name="lm_evaluation_harness_hf", task_filter={'order_by': ["-last_update"]}).id
        evalplus_task_id = Task.get_task(project_name="Automation",task_name="evalplus_hf", task_filter={'order_by': ["-last_update"]}).id
    elif engine == "sparseml":
        lm_evaluation_harness_task_id = Task.get_task(project_name="Automation",task_name="lm_evaluation_harness_sparseml", task_filter={'order_by': ["-last_update"]}).id
        evalplus_task_id = Task.get_task(project_name="Automation",task_name="evalplus_sparseml", task_filter={'order_by': ["-last_update"]}).id

    if num_gpus_evaluation > 1:
        lm_evaluation_override_engine = {"Args/parallelize": True}
    else:
        lm_evaluation_override_engine = {}

    evalplus_override_engine = lm_evaluation_override_engine

lm_evaluation_override = {
    "Args/model_id": oneshot_model_id,
    "Args/clearml_model": True,
    "Args/packages": evaluation_packages,
    "Args/batch_size": batch_size,
}
lm_evaluation_override.update(lm_evaluation_override_engine)

evalplus_override = {
    "Args/model_id": oneshot_model_id,
    "Args/clearml_model": True,
    "Args/packages": evaluation_packages,
    "Args/batch_size": evalplus_batch_size,
}
evalplus_override.update(evalplus_override_engine)


if "mmlu_llama_3.1_instruct" in args["benchmark_tasks"]:
    mmlu_instruct_step_name = f"{oneshot_step_name}_mmlu_llama_3.1_{engine}"
    mmlu_instruct_override = {
        "Args/benchmark_tasks": "mmlu_llama_3.1_instruct",
        "Args/num_fewshot": 5,
        "Args/fewshot_as_multiturn": True,
        "Args/apply_chat_template": True,
        "Args/add_bos_token": True,
        "Args/max_gen_toks": 10,
    }
    mmlu_instruct_override.update(lm_evaluation_override)
    mmlu_instruct_max_model_len = min(3850, args["max_seq_len"])
    if engine == "vllm":
        mmlu_instruct_override["Args/max_model_len"] = mmlu_instruct_max_model_len
    else:
        mmlu_instruct_override["Args/max_length"] = mmlu_instruct_max_model_len

    pipe.add_step(
        name=mmlu_instruct_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=mmlu_instruct_override,
    )

if "mmlu_cot_0shot_llama_3.1_instruct" in args["benchmark_tasks"]:
    mmlu_cot_step_name = f"{oneshot_step_name}_mmlu_cot_llama_3.1_{engine}"
    mmlu_cot_override = {
        "Args/benchmark_tasks": "mmlu_cot_0shot_llama_3.1_instruct",
        "Args/num_fewshot": 0,
        "Args/apply_chat_template": True,
        "Args/add_bos_token": True,
        "Args/max_gen_toks": 1024,
    }
    mmlu_cot_override.update(lm_evaluation_override)
    mmlu_cot_max_model_len = min(4064, args["max_seq_len"])
    if engine == "vllm":
        mmlu_cot_override["Args/max_model_len"] = mmlu_cot_max_model_len
    else:
        mmlu_cot_override["Args/max_length"] = mmlu_cot_max_model_len
    pipe.add_step(
        name=mmlu_cot_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=mmlu_cot_override,
    )

if "arc_challenge_llama_3.1_instruct" in args["benchmark_tasks"]:
    arc_instruct_step_name = f"{oneshot_step_name}_arc_challenge_llama_3.1_instruct_{engine}"
    arc_instruct_override = {
        "Args/benchmark_tasks": "arc_challenge_llama_3.1_instruct",
        "Args/num_fewshot": 0,
        "Args/apply_chat_template": True,
        "Args/add_bos_token": True,
        "Args/max_gen_toks": 100,
    }
    arc_instruct_override.update(lm_evaluation_override)
    arc_instruct_max_model_len = min(3940, args["max_seq_len"])
    if engine == "vllm":
        arc_instruct_override["Args/max_model_len"] = arc_instruct_max_model_len
    else:
        arc_instruct_override["Args/max_length"] = arc_instruct_max_model_len
    pipe.add_step(
        name=arc_instruct_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=arc_instruct_override,
    )

if "gsm8k_cot_llama_3.1_instruct" in args["benchmark_tasks"]:
    gsm8k_instruct_step_name = f"{oneshot_step_name}_gsm8k_cot_llama_3.1_instruct_{engine}"
    gsm8k_instruct_override = {
        "Args/benchmark_tasks": "gsm8k_cot_llama_3.1_instruct",
        "Args/num_fewshot": 8,
        "Args/fewshot_as_multiturn": True,
        "Args/apply_chat_template": True,
        "Args/add_bos_token": True,
        "Args/max_gen_toks": 1024,
    }
    gsm8k_instruct_override.update(lm_evaluation_override)
    gsm8k_instruct_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        gsm8k_instruct_override["Args/max_model_len"] = gsm8k_instruct_max_model_len
    else:
        gsm8k_instruct_override["Args/max_length"] = gsm8k_instruct_max_model_len
    pipe.add_step(
        name=gsm8k_instruct_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=gsm8k_instruct_override,
    )

if "openllm" in args["benchmark_tasks"]:
    openllm_step_name = f"{oneshot_step_name}_openllm_{engine}"
    openllm_override = {
        "Args/benchmark_tasks": "openllm",
        "Args/add_bos_token": True,
    }
    openllm_override.update(lm_evaluation_override)
    openllm_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        openllm_override["Args/max_model_len"] = openllm_max_model_len
    else:
        openllm_override["Args/max_length"] = openllm_max_model_len
    pipe.add_step(
        name=openllm_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=openllm_override,
    )

if "mmlu" in args["benchmark_tasks"]:
    mmlu_step_name = f"{oneshot_step_name}_mmlu_{engine}"
    mmlu_override = {
        "Args/benchmark_tasks": "mmlu",
        "Args/num_fewshot": 5,
        "Args/add_bos_token": True,
    }
    mmlu_override.update(lm_evaluation_override)

    mmlu_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        mmlu_override["Args/max_model_len"] = mmlu_max_model_len
    else:
        mmlu_override["Args/max_length"] = mmlu_max_model_len
    pipe.add_step(
        name=mmlu_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=mmlu_override,
    )

if "hellaswag" in args["benchmark_tasks"]:
    hellaswag_step_name = f"{oneshot_step_name}_hellaswag_{engine}"
    hellaswag_override = {
        "Args/benchmark_tasks": "hellaswag",
        "Args/num_fewshot": 10,
        "Args/add_bos_token": True,
    }
    hellaswag_override.update(lm_evaluation_override)
    hellaswag_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        hellaswag_override["Args/max_model_len"] = hellaswag_max_model_len
    else:
        hellaswag_override["Args/max_length"] = hellaswag_max_model_len
    pipe.add_step(
        name=hellaswag_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=hellaswag_override,
    )

if "winogrande" in args["benchmark_tasks"]:
    winogrande_step_name = f"{oneshot_step_name}_winogrande_{engine}"
    winogrande_override = {
        "Args/benchmark_tasks": "winogrande",
        "Args/num_fewshot": 5,
        "Args/add_bos_token": True,
    }
    winogrande_override.update(lm_evaluation_override)
    winogrande_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        winogrande_override["Args/max_model_len"] = winogrande_max_model_len
    else:
        winogrande_override["Args/max_length"] = winogrande_max_model_len
    pipe.add_step(
        name=winogrande_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=winogrande_override,
    )

if "arc_challenge" in args["benchmark_tasks"]:
    arc_challenge_step_name = f"{oneshot_step_name}_arc_challenge_{engine}"
    arc_challenge_override = {
        "Args/benchmark_tasks": "arc_challenge",
        "Args/num_fewshot": 25,
        "Args/add_bos_token": True,
    }
    arc_challenge_override.update(lm_evaluation_override)
    arc_challenge_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        arc_challenge_override["Args/max_model_len"] = arc_challenge_max_model_len
    else:
        arc_challenge_override["Args/max_length"] = arc_challenge_max_model_len
    pipe.add_step(
        name=arc_challenge_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=arc_challenge_override,
    )

if "gsm8k" in args["benchmark_tasks"]:
    gsm8k_step_name = f"{oneshot_step_name}_gsm8k_{engine}"
    gsm8k_override = {
        "Args/benchmark_tasks": "gsm8k",
        "Args/num_fewshot": 5,
        "Args/add_bos_token": True,
    }
    gsm8k_override.update(lm_evaluation_override)
    gsm8k_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        gsm8k_override["Args/max_model_len"] = gsm8k_max_model_len
    else:
        gsm8k_override["Args/max_length"] = gsm8k_max_model_len
    pipe.add_step(
        name=gsm8k_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=gsm8k_override,
    )

if "truthfulqa" in args["benchmark_tasks"]:
    truthfulqa_step_name = f"{oneshot_step_name}_truthfulqa_{engine}"
    truthfulqa_override = {
        "Args/benchmark_tasks": "truthfulqa",
        "Args/num_fewshot": 0,
        "Args/add_bos_token": True,
    }
    truthfulqa_override.update(lm_evaluation_override)
    truthfulqa_max_model_len = min(4096, args["max_seq_len"])
    if engine == "vllm":
        truthfulqa_override["Args/max_model_len"] = truthfulqa_max_model_len
    else:
        truthfulqa_override["Args/max_length"] = truthfulqa_max_model_len
    pipe.add_step(
        name=truthfulqa_step_name,
        parents=[oneshot_step_name],
        base_task_id=lm_evaluation_harness_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=truthfulqa_override,
    )

if "humaneval" in args["benchmark_tasks"]:
    humaneval_step_name = f"{oneshot_step_name}_humaneval_{engine}"
    humaneval_override = {
        "Args/num_fewshot": 0,
        "Args/add_bos_token": True,
    }
    humaneval_override.update(evalplus_override)
    pipe.add_step(
        name=humaneval_step_name,
        parents=[oneshot_step_name],
        base_task_id=evalplus_task_id,
        execution_queue=args["evaluation_queue"],
        parameter_override=humaneval_override,
    )

pipe.start()
