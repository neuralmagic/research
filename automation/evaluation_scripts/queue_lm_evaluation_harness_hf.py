from clearml import Task
import argparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ lm-evaluation-harness using HF backend")

parser.add_argument("--model-id", type=str, nargs="+")
parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--benchmark-tasks", type=str, default="openllm")
parser.add_argument("--num-fewshot", type=int, default=None)
parser.add_argument("--add-bos-token", action="store_true", default=False)
parser.add_argument("--max-length", type=int, default=4096)
parser.add_argument("--batch-size", type=str, default="auto")
parser.add_argument("--parallelize", action="store_true", default=False)
parser.add_argument("--apply-chat-template", action="store_true", default=False)
parser.add_argument("--fewshot-as-multiturn", action="store_true", default=False)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--packages", type=str, nargs="+", default=None)

args = parser.parse_args()

args = vars(args)
project_name = args.pop("project_name")
task_name = args.pop("task_name")
queue_name = args.pop("queue_name")
additional_packages = args.pop("packages")

packages = [
    "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main", 
    "sentencepiece",
    "git+https://github.com/neuralmagic/compressed-tensors.git@main"
]

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-torch:latest")
task.set_script(repository="https://github.com/neuralmagic/research.git", branch="main",working_dir="clearml")
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

max_length = args["max_length"]
model_args = f"pretrained={model_id},dtype=auto,max_length={max_length}"
if args["add_bos_token"]:
    model_args += ",add_bos_token=True"
if args["trust_remote_code"]:
    model_args += ",trust_remote_code=True"
if args["parallelize"]:
    model_args += ",parallelize=True"

inputs = [
    "python3", "-m", "lm_eval", 
    "--model", "hf", 
    "--tasks", args["benchmark_tasks"], 
    "--model_args", model_args,
    "--write_out", 
    "--show_config", 
    "--output_path", ".",
]

if args["apply_chat_template"]:
    inputs.append("--apply_chat_template")
if args["fewshot_as_multiturn"]:
    inputs.append("--fewshot_as_multiturn")

if args["num_fewshot"] is not None:
    inputs.extend(["--num_fewshot", str(args["num_fewshot"])])

subprocess.run(inputs)

model_suffix = os.path.split(model_id)[-1]
results_dir = glob(f"*{model_suffix}")[-1]
json_file = glob(os.path.join(results_dir, "results*.json"))[0]
push_to_clearml(task, json_file)
