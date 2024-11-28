from clearml import Task
import argparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ evalplus harness")

parser.add_argument("--model-id", type=str)
parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--mode", type=str, choices=["generate", "evaluate", "complete"], default="complete")
parser.add_argument("--v2", action="store_true", default=False)
parser.add_argument("--max-instances", type=int, default=None)
parser.add_argument("--chunksize", type=int, default=64)
parser.add_argument("--generation-task", type=str, default=None)
parser.add_argument("--annotator", type=str, default="Llama-3.1-70B-Instruct")
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--max-new-tokens", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--cpu-offload-gb", type=int, default=None)
parser.add_argument("--enable-chunked-prefill", action="store_true", default=False)
parser.add_argument("--max-model-len", type=int, default=None)
parser.add_argument("--max-num-batched-tokens", type=int, default=512)
parser.add_argument("--max-num-seqs", type=int, default=128)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--build-vllm", action="store_true", default=False)

args = parser.parse_args()

packages = [
    "git+https://github.com/neuralmagic/alpaca_eval.git@main", 
    "git+https://github.com/neuralmagic/compressed-tensors.git@main", 
    "sentencepiece",
]

if args.build_vllm:
    packages.append("git+https://github.com/vllm-project/vllm.git@main")
else:
    packages.append("https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl")
    #packages.append("vllm")


if args.packages is not None and len(args.packages) > 0:
    packages.extend(args.packages)

Task.force_store_standalone_script()

task = Task.init(project_name=args.project_name, task_name=args.task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
task.set_script(repository="https://github.com/neuralmagic/alpaca_eval.git", branch="main")
task.set_packages(packages)

task.execute_remotely(args.queue_name)

#
# REMOTE
#

from clearml import InputModel
import yaml
import os
import subprocess
import sys
from transformers import AutoTokenizer
import alpaca_eval

if "single" in args.queue_name or "x1" in args.queue_name:
    num_gpus = 1
elif "double" in args.queue_name or "x2" in args.queue_name:
    num_gpus = 2
elif "quad" in args.queue_name or "x4" in args.queue_name:
    num_gpus = 4
elif "octo" in args.queue_name or "x8" in args.queue_name:
    num_gpus = 8

if args.clearml_model:
    input_model = InputModel(model_id=args.model_id)
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args.model_id

generate = args.mode in ["generate", "complete"]
evaluate = args.mode in ["evaluate", "complete"]

alpaca_eval_repo_path = os.path.split(alpaca_eval.__file__)[0]
executable_path = os.path.dirname(sys.executable)
alpaca_eval_exec_path = os.path.join(executable_path, "alpaca_eval")

if generate:

    # Create model directory
    model_name = model_id.replace("/", "__")
    model_path = os.path.join(alpaca_eval_repo_path, "models_configs", f"{model_name}")
    os.mkdir(model_path)

    # Create prompt file
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dummy_message = [{"role": "user", "content": "{instruction}"}]
    prompt = tokenizer.apply_chat_template(dummy_message, add_generation_prompt=True, tokenize=False)

    prompt_file_path = os.path.join(model_path, "prompt.txt")
    prompt_file = open(prompt_file_path, "w")
    prompt_file.write(prompt)
    prompt_file.close()

    task.upload_artifact(name="generation prompt template", artifact_object=prompt_file_path)

    # Create config file
    model_kwargs = {
        "dtype": args.dtype,
        "tensor_parallel_size": num_gpus,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "seed": args.seed,
    }
    if args.cpu_offload_gb:
        model_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.max_model_len:
        model_kwargs["max_model_len"] = args.max_model_len

    completions_kwargs = {
        "model_name": model_id,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "model_kwargs": model_kwargs,
    }

    config = {
        model_name: {
            "prompt_template": prompt_file_path,
            "fn_completions": "vllm_local_completions",
            "completions_kwargs": completions_kwargs,
        }
    }

    config_file_path = os.path.join(model_path, "configs.yaml")
    config_file = open(config_file_path, "w")
    yaml.dump(config, config_file)
    config_file.close()

    task.upload_artifact(name="model configs", artifact_object=config_file_path)

    inputs = [
        alpaca_eval_exec_path, "generate",
        "--model_configs", model_name,
        "--chunksize", str(args.chunksize),
        "--output_path", "results",
    ]

    if args.max_instances is not None:
        inputs.extend(["--max_instances", str(args.max_instances)])

    print("\n\nStarting generation...\n\n")

    subprocess.run(inputs)

    generation_outputs = os.path.join("results", "model_outputs.json")
    reference_outputs = os.path.join("results", "reference_outputs.json")

    task.upload_artifact(name="generation outputs", artifact_object=generation_outputs)
    task.upload_artifact(name="reference outputs", artifact_object=reference_outputs)


if evaluate:
    if not generate:
        generation_task = Task.get_task(task_id=args.generation_task)
        generation_outputs = generation_task.artifacts["generation outputs"].get_local_copy()
        reference_outputs = generation_task.artifacts["reference outputs"].get_local_copy()

    os.environ['IS_ALPACA_EVAL_2'] = str(args.v2)
    
    annotator_config_path = os.path.join(alpaca_eval_repo_path, "evaluators_configs", args.annotator, "configs.yaml")
    annotator_config = yaml.safe_load(open(annotator_config_path, "r"))
    annotator_config[args.annotator]["completions_kwargs"]["model_kwargs"]["tensor_parallel_size"] = num_gpus

    annotator_config_file = open(annotator_config_path, "w")
    yaml.dump(annotator_config, annotator_config_file)
    annotator_config_file.close()

    task.upload_artifact(name="annotator configs", artifact_object=annotator_config_path)

    inputs = [
        alpaca_eval_exec_path, "evaluate",
        "--model_outputs", generation_outputs,
        "--annotators_config", args.annotator,
        "--output_path", "results",
    ]

    if args.max_instances is not None:
        inputs.extend(["--max_instances", str(args.max_instances)])

    print("\n\nStarting evaluation...\n\n")

    subprocess.run(inputs)

    annotation_outputs = os.path.join("results", args.annotator, "annotations.json")
    leaderboad = os.path.join("results", args.annotator, "leaderboard.csv")

    task.upload_artifact(name="annotations", artifact_object=annotation_outputs)

    task.get_logger().report_table(
        title="AlpacaEval Leaderboard",
        series=f"Annotator: {args.annotator}",
        csv=leaderboad,
    )