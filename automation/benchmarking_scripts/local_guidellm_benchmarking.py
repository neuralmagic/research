from clearml import Task
import argparse

parser = argparse.ArgumentParser(description = "Benchmark LLM performance with GuideLLM")

parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--dtype", type=str, default=None)
parser.add_argument("--max-model-len", type=int, default=None)
parser.add_argument("--num-gpus", type=int, default=None)
parser.add_argument("--server-wait-time", type=int, default=600)

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
    

Task.force_store_standalone_script()

task = Task.init(project_name=args["project_name"], task_name=args["task_name"])
task.connect(guidellm_args, name="GuideLLM")

from clearml import InputModel
import subprocess
import requests
import time
import os
import sys
from urllib.parse import urlparse
import torch

if args["num_gpus"] is None:
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = args["num_gpus"]

guidellm_args = task.get_parameters_as_dict()["GuideLLM"]

if "output-path" not in guidellm_args:
    guidellm_args["output-path"] = "guidellm_output.json"

if args["clearml_model"]:
    input_model = InputModel(model_id=guidellm_args["model"])
    guidellm_args["model"] = input_model.get_local_copy()
    task.connect(input_model)

parsed_target = urlparse(guidellm_args["target"])

server_command = ["vllm", "serve", guidellm_args["model"], "--host", parsed_target.hostname, "--port", str(parsed_target.port)]
if num_gpus > 1:
    server_command.extend(["--tensor-parallel-size", str(num_gpus)])
if args["max_model_len"] is not None:
    server_command.extend(["--max-model-len", str(args["max_model_len"])])
if args["dtype"] is not None:
    server_command.extend(["--dtype", args["dtype"]])

server_log_file = open("vllm_server_log.txt", "w")
server_process = subprocess.Popen(" ".join(server_command), stdout=server_log_file, stderr=server_log_file, shell=True)

delay = 5
server_initialized = False
for _ in range(args["server_wait_time"] // delay):
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
    inputs = ["guidellm"]
    for k, v in guidellm_args.items():
        argument_name = k.replace("_","-")
        inputs.append(f"--{argument_name}")
        inputs.append(str(v))

    print("Starting benchmarking...")
    subprocess.run(" ".join(inputs), shell=True)

    server_process.kill()

    task.upload_artifact(name="guidellm benchmarking output", artifact_object=guidellm_args["output-path"])
    task.upload_artifact(name="vLLM server log", artifact_object="vllm_server_log.txt")
else:
    server_process.kill()
    task.upload_artifact(name="vLLM server log", artifact_object="vllm_server_log.txt")
    raise AssertionError("Server failed to intialize")
