import subprocess
import requests
import time
import sys
import os
import torch
from urllib.parse import urlparse
from clearml import Task

SERVER_LOG_PREFIX = "generation_server_log"


def start_generation(
    generation_args, 
    model_id, 
    target, 
    server_wait_time,
    gpu_count,
):
    task = Task.current_task()

    print("Inside start generation server")

    executable_path = os.path.dirname(sys.executable)
    generation_path = os.path.join(executable_path, "generation")

    available_gpus = list(range(torch.cuda.device_count()))
    selected_gpus = available_gpus[:gpu_count]

    subprocess_env = os.environ.copy()
    subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected_gpus)

    parsed_target = urlparse(target)
    print(f"generation path is: {generation_path}")

    server_command = [
        f"{generation_path}", "--help", 
    ]

    print(server_command)
    subprocess_env = os.environ.copy()

    for k, v in generation_args.items():
        if k.startswith("GENERATION_"):
            subprocess_env[k] = str(v)
        else:
            if v == True or v == "True":
                server_command.append(f"--{k}")
            else:
                server_command.extend([f"--{k}", str(v)])


    server_log_file_name = f"{SERVER_LOG_PREFIX}_{task.id}.txt"
    server_log_file = open(server_log_file_name, "w")
    print("Server command:", " ".join(server_command))
    print(f"GENERATION logs are located at: {server_log_file} in {os.getcwd()}")
    server_process = subprocess.Popen(server_command, stdout=server_log_file, stderr=server_log_file, shell=False, env=subprocess_env)

    delay = 5
    server_initialized = False
    for _ in range(server_wait_time // delay):
        try:
            response = requests.get(target + "/models")
            print(f"response: {response}")
            if response.status_code == 200:
                print("Server initialized")
                server_initialized = True
                break  # Exit the loop if the request is successful
        except requests.exceptions.RequestException as e:
            pass

        time.sleep(delay)

    if server_initialized:
        return server_process, True, server_log_file_name
    else:
        return server_process, False, server_log_file_name
