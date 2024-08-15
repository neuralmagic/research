from clearml import Task
import argparse

project_name = "Automation"
task_name = "lm_evaluation_harness_vllm"
queue_name = "oneshot-a100x1"

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ lm-evaluation-harness using vllm backend")

parser.add_argument("--model-id", type=str)
parser.add_argument("--num-gpus", type=int)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--benchmark-tasks", type=str, default="openllm")
parser.add_argument("--num-fewshot", type=int, default=None)
parser.add_argument("--add-bos-token", action="store_true", default=False)
parser.add_argument("--max-gen-toks", type=int, default=256)
parser.add_argument("--batch-size", type=str, default="auto")
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
parser.add_argument("--max-model-len", type=int, default=4096)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--apply-chat-template", action="store_true", default=False)
parser.add_argument("--fewshot-as-multiturn", action="store_true", default=False)
parser.add_argument("--build-vllm", action="store_true", default=False)

args = parser.parse_args()

args = vars(args)
additional_packages = args.pop("packages")
build_vllm = args.pop("build_vllm")

packages = [
    "git+https://github.com/neuralmagic/lm-evaluation-harness.git@llama_3.1_instruct", 
    "git+https://github.com/neuralmagic/compressed-tensors.git@main",
    "sentencepiece",
]

if build_vllm:
    packages.append("git+https://github.com/vllm-project/vllm.git@main")
else:
    packages.append("vllm")

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
task.set_script(repository="https://github.com/neuralmagic/research.git", branch="main",working_dir="clearml_evaluation_parsing")
task.set_packages(packages)

task.execute_remotely(queue_name)

#
# REMOTE
#

from clearml import InputModel
from glob import glob
import subprocess
import os
from lm_evaluation_harness import push_to_clearml

if args["clearml_model"]:
    input_model = InputModel(model_id=args["model_id"])
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args["model_id"]

max_model_len = args["max_model_len"]
max_gen_toks = args["max_gen_toks"]
gpu_memory_utilization = args["gpu_memory_utilization"]
num_gpus = args["num_gpus"]
model_args = f"pretrained={model_id},dtype=auto,max_model_len={max_model_len},max_gen_toks={max_gen_toks},gpu_memory_utilization={gpu_memory_utilization},tensor_parallel_size={num_gpus}"
if args["add_bos_token"]:
    model_args += ",add_bos_token=True"
if args["trust_remote_code"]:
    model_args += ",trust_remote_code=True"

inputs = [
    "python3", "-m", "lm_eval", 
    "--model", "vllm", 
    "--tasks", args["benchmark_tasks"], 
    "--model_args", model_args,
    "--write_out", 
    "--show_config", 
    "--output_path", ".",
    "--batch_size", args["batch_size"],
]

if args["num_fewshot"] is not None:
    inputs.extend(["--num_fewshot", str(args["num_fewshot"])])

if args["apply_chat_template"]:
    inputs.append("--apply_chat_template")

if args["fewshot_as_multiturn"]:
    inputs.append("--fewshot_as_multiturn")

subprocess.run(inputs)
model_suffix = os.path.split(model_id)[-1]
results_dir = glob(f"*{model_suffix}")[-1]
json_file = glob(os.path.join(results_dir, "results*.json"))[0]
if args["clearml_model"]:
    push_to_clearml(task, json_file, input_model)
else:
    push_to_clearml(task, json_file)
