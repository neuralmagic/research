from clearml import Task
import argparse
import os
#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ lm-evaluation-harness using vllm backend")

parser.add_argument("--model-id", type=str)
parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--config-format", type=str)
parser.add_argument("--server-wait-time", type=int, default=4000)
parser.add_argument("--tokenizer-mode", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--eval-name", type=str)
parser.add_argument("--max-model-len", type=int, default=None)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--limit-mm-per-prompt", type=str)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--build-vllm", action="store_true", default=False)

args = parser.parse_args()

args = vars(args)
project_name = args.pop("project_name")
task_name = args.pop("task_name")
queue_name = args.pop("queue_name")
additional_packages = args.pop("packages")
#os.environ["VLLM_COMMIT"] = "098f94de42859f8251fe920f87adb88336129c53"

packages = [
    "datasets==3.0.0",
    "fire==0.6.0",
    "numpy==1.26.4",
    "openai==1.60.0",
    "pillow==10.4.0",
    "tqdm==4.66.5",
    "mistral_common[opencv]",
    #f"https://vllm-wheels.s3.us-west-2.amazonaws.com/{os.environ['VLLM_COMMIT']}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl",
    "vllm",
]

# # After installing vllm, uninstall any existing torch version
# post_install_commands = [
#     "pip uninstall -y torch",
#     "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
# ]

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

# # Create a symbolic link from python3 to python
# if not os.path.exists("/usr/bin/python"):
#     subprocess.run(["ln", "-s", "/usr/bin/python3.10", "/usr/bin/python"], check=True)

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
# Add the repository URL as a requirement
task._update_requirements(
    {"pip": ["git+https://github.com/mistralai/mistral-evals.git@main"]}
)
task.set_script(repository="https://github.com/mistralai/mistral-evals.git", branch="main",working_dir=".")
task.set_packages(packages)
task.execute_remotely(queue_name)

# # Run post-install commands to enforce the correct PyTorch version
# for command in post_install_commands:
#     os.system(command)

#
# REMOTE
#

from clearml import InputModel
from glob import glob
import subprocess
import requests
import time
import os
import sys
from urllib.parse import urlparse

if "single" in queue_name or "x1" in queue_name:
    num_gpus = 1
elif "double" in queue_name or "x2" in queue_name:
    num_gpus = 2
elif "quad" in queue_name or "x4" in queue_name:
    num_gpus = 4
elif "octo" in queue_name or "x8" in queue_name:
    num_gpus = 8

if args["clearml_model"]:
    input_model = InputModel(model_id=args["model_id"])
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args["model_id"]

#VLLM Serve

# Determine the paths for the executable and VLLM
executable_path = os.path.dirname(sys.executable)
vllm_path = os.path.join(executable_path, "vllm")

# Build the initial server command with the vllm executable and action
server_command = [vllm_path, "serve", model_id]

# Add optional arguments if provided
if args["config_format"] is not None:
    server_command.extend(["--config_format", args["config_format"]])

if args["tokenizer_mode"] is not None:
    server_command.extend(["--tokenizer_mode", args["tokenizer_mode"]])

# Add optional GPU configuration if applicable
if num_gpus > 1:
    server_command.extend(["--tensor_parallel_size", str(num_gpus)])

# If max_model_len is provided, ensure it's added to the command
if args["max_model_len"] is not None:
    server_command.extend(["--max_model_len", str(args["max_model_len"])])

if args["limit_mm_per_prompt"] is not None:
    server_command.extend(["--limit_mm_per_prompt", args["limit_mm_per_prompt"]])

server_command.extend(["--max_num_seqs", "8"])

server_command.extend(["--gpu_memory_utilization", "0.9"])
# server_command.extend(["--dtype", "float16"])

if "Llama" in args["model_id"]:
    server_command.extend(["--disable_frontend_multiprocessing"])
    server_command.extend(["--enforce_eager"])

import random

# Generate a random 4-digit port number
random.seed(time.time())
random_port = random.randint(1000, 9999)

server_command.extend(["--port", f"{random_port}"])

# Set up the log file for the server
server_log_file = open("vllm_server_log.txt", "w")

print("server_command", server_command)

# Start vllm server
server_process = subprocess.Popen(" ".join(server_command), stdout=server_log_file, stderr=server_log_file, shell=True)

# Wait for server to spin up
delay = 5
server_initialized = False

# Construct the server URL with the random port
server_url = f"http://localhost:{random_port}/v1"

for _ in range(args["server_wait_time"] // delay):
    try:
        response = requests.get(server_url + "/models")
        if response.status_code == 200:
            print("Server initialized")
            server_initialized = True
            break  # Exit the loop if the request is successful
    except requests.exceptions.RequestException as e:
        pass

    time.sleep(delay)

if server_initialized:
    inputs = [
        "python3", "-m", "eval.run", "eval_vllm", 
        "--model_name", args["model_id"], 
        "--url", f"http://0.0.0.0:{random_port}",
        "--output_dir", args["output_dir"],
        "--eval_name", args["eval_name"],
    ]
    print("Starting evaluation...")
    subprocess.run(" ".join(inputs), shell=True)

    server_process.kill()

    task.upload_artifact(name="evaluation output", artifact_object=args["output_dir"])
    task.upload_artifact(name="vLLM server log", artifact_object="vllm_server_log.txt")
else:
    server_process.kill()
    task.upload_artifact(name="vLLM server log", artifact_object="vllm_server_log.txt")
    print("Server failed to intialize")
