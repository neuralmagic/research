import subprocess
import requests
import time
import os
import sys
from urllib.parse import urlparse
from clearml import Task
import torch
from automation.utils import resolve_model_id, cast_args
import psutil
from pyhocon import ConfigFactory


SERVER_LOG_PREFIX = "vllm_server_log"


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()  # or child.kill()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass


def start_vllm_server(
    vllm_args, 
    model_id, 
    target, 
    server_wait_time, 
    suffix,
):
    executable_path = os.path.dirname(sys.executable)
    vllm_path = os.path.join(executable_path, "vllm")

    num_gpus = torch.cuda.device_count()

    parsed_target = urlparse(target)

    server_command = [
        f"{vllm_path}", "serve", 
        model_id,
        "--host", parsed_target.hostname, 
        "--port", str(parsed_target.port),
        "--tensor-parallel-size", str(num_gpus)
    ]

    subprocess_env = os.environ.copy()

    for k, v in vllm_args.items():
        if k.startswith("VLLM_"):
            subprocess_env[k] = str(v)
        else:
            if v == True or v == "True":
                v = "true"
            server_command.extend([f"--{k}", str(v)])

    server_log_file = open(f"{SERVER_LOG_PREFIX}_{suffix}.txt", "w")
    server_process = subprocess.Popen(server_command, stdout=server_log_file, stderr=server_log_file, shell=False, env=subprocess_env)

    delay = 5
    server_initialized = False
    for _ in range(server_wait_time // delay):
        try:
            response = requests.get(target + "/models")
            if response.status_code == 200:
                print("Server initialized")
                server_initialized = True
                break  # Exit the loop if the request is successful
        except requests.exceptions.RequestException as e:
            pass

        time.sleep(delay)

    if server_initialized:
        return server_process, True
    else:
        return server_process, False

def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    guidellm_args = ConfigFactory.parse_string(task.get_configuration_object("GuideLLM"))
    
    environment_args = task.get_configuration_object("environment")
    if environment_args is None:
        environment_args = {}
    else:
        environment_args = ConfigFactory.parse_string(environment_args)
    
    vllm_args = task.get_configuration_object("vLLM")
    if vllm_args is None:
        vllm_args = {}
    else:
        vllm_args = ConfigFactory.parse_string(vllm_args)

    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(args["Args"]["model_id"], bool(args["Args"]["clearml_model"]), task)

    # Start vLLM server
    server_process, server_initialized = start_vllm_server(
        vllm_args,
        model_id,
        guidellm_args["target"],
        args["Args"]["server_wait_time"],
        task.id,
    )

    if not server_initialized:
        kill_process_tree(server_process.pid)
        task.upload_artifact(name="vLLM server log", artifact_object=f"{SERVER_LOG_PREFIX}_{task.id}.txt")
        raise AssertionError("Server failed to intialize")

    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    guidellm_args["model"] = model_id

    from guidellm import generate_benchmark_report
    guidellm_args = cast_args(guidellm_args, generate_benchmark_report)
    report = generate_benchmark_report(**guidellm_args)
    kill_process_tree(server_process.pid)

    task.upload_artifact(name="guidellm guidance report", artifact_object=report.to_json())
    task.upload_artifact(name="vLLM server log", artifact_object=f"{SERVER_LOG_PREFIX}_{task.id}.txt")

if __name__ == '__main__':
    main()