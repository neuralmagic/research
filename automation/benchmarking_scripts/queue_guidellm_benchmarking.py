from clearml import Task
import argparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Benchmark LLM performance with GuideLLM")

parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--build-vllm", action="store_true", default=False)
parser.add_argument("--max-model-len", type=int, default=None)

args, unparsed_args = parser.parse_known_args()

guidellm_args = {}
for id, entry in enumerate(unparsed_args):
    if entry.startswith("--") or entry.startswith("-"):
        if entry.startswith("--"):
            key = entry[2:]
        else:
            key = entry[1:]
        if len(unparsed_args) > id+1:
            if unparsed_args[id+1].startswith("-"):
                value = True
            else:
                value = unparsed_args[id+1]
        guidellm_args[key] = value
    
args = vars(args)
additional_packages = args["packages"]

packages = [
    "git+https://github.com/neuralmagic/guidellm.git@fd04739254d8eeb5ebff21ac2dc4c79c47645703",
    "sentencepiece",
]

if args["build_vllm"]:
    packages.append("git+https://github.com/vllm-project/vllm.git@main")
else:
    packages.append("vllm")

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

task = Task.init(project_name=args["project_name"], task_name=args["task_name"])
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
task.set_packages(packages)
task.connect(guidellm_args, name="GuideLLM")

task.execute_remotely(args["queue_name"])

#
# REMOTE
#

from clearml import InputModel
import subprocess
import requests
import time
import os
import sys
from urllib.parse import urlparse

if "single" in args["queue_name"] or "x1" in args["queue_name"]:
    num_gpus = 1
elif "double" in args["queue_name"] or "x2" in args["queue_name"]:
    num_gpus = 2
elif "quad" in args["queue_name"] or "x4" in args["queue_name"]:
    num_gpus = 4
elif "octo" in args["queue_name"] or "x8" in args["queue_name"]:
    num_gpus = 8

guidellm_args = task.get_parameters_as_dict()["GuideLLM"]

if "output-path" not in guidellm_args:
    guidellm_args["output-path"] = "guidellm_output.json"

if args["clearml_model"]:
    input_model = InputModel(model_id=guidellm_args["model"])
    guidellm_args["model"] = input_model.get_local_copy()
    task.connect(input_model)

executable_path = os.path.dirname(sys.executable)
vllm_path = os.path.join(executable_path, "vllm")
guidellm_path = os.path.join(executable_path, "guidellm")

parsed_target = urlparse(guidellm_args["target"])

server_command = [f"{vllm_path}", "serve", guidellm_args["model"], "--host", parsed_target.hostname, "--port", str(parsed_target.port)]
if num_gpus > 1:
    server_command.extend(["--tensor-parallel-size", str(num_gpus)])
if args["max_model_len"] is not None:
    server_command.extend(["--max-model-len", str(args["max_model_len"])])

server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

delay = 5
server_initialized = False
for _ in range(800):
    try:
        response = requests.get(guidellm_args["target"] + "/models")
        if response.status_code == 200:
            print("Server initialized")
            server_initialized = True
            break  # Exit the loop if the request is successful
    except requests.exceptions.RequestException as e:
        pass

    time.sleep(delay)

if server_initialized:
    inputs = [f"{guidellm_path}"]
    for k, v in guidellm_args.items():
        argument_name = k.replace("_","-")
        inputs.append(f"--{argument_name}")
        inputs.append(v)

    print("Starting benchmarking...")
    subprocess.run(inputs)

    task.upload_artifact(name="guidellm benchmarking output", artifact_object=guidellm_args["output-path"])
else:
    print("Server failed to intialize")
    stdout = server_process.communicate()[0]
    print(stdout)
