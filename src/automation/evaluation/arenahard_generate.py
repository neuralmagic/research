import subprocess
import requests
import time
import sys
import os
#import arenahard
from urllib.parse import urlparse
from clearml import Task
from pathlib import Path

SERVER_LOG_PREFIX = "generation_server_log"


def start_generation(
    #module_path
    #generation_args, 
):
    task = Task.current_task()

    print("Inside start generation server")

    executable_path = os.path.dirname(sys.executable)
    python_path = os.path.join(executable_path, "python3")
    print(f"python path is: {python_path}")
    base_path = Path(executable_path)
    sitepackages_path = os.path.join(base_path.parents[0], "lib", "python3.10", "site-packages")
    sys.path.append(sitepackages_path)
    generation_path = os.path.join(sitepackages_path, "arenahard", "gen_answer.py")
    assert os.path.exists(generation_path), f"{generation_path} does not exist"
    config_path = os.path.join(os.getcwd(), "src", "automation", "standards", "arenahard")
    api_config_path = os.path.join(config_path, "api_config.yaml")
    assert os.path.exists(api_config_path), f"{api_config_path} does not exist"
    gen_answer_config_path = os.path.join(config_path, "gen_answer_config.yaml")
    assert os.path.exists(gen_answer_config_path), f"{gen_answer_config_path} does not exist"

    subprocess_env = os.environ.copy()
    #subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected_gpus)
    # src/automation/standards/arenahard/api_config.yaml
    # src/automation/standards/arenahard/gen_answer_config.yaml
    # python3 gen_answer.py --config-file config/gen_answer_config.yaml  --endpoint-file config/api_config.yaml


    server_command = [
        python_path,
        f"{generation_path}",
        "--config-file",
        "api_config.yaml",
        "--endpoint-file",
        "gen_answer_config.yaml",
        "--config-path",
        config_path,
        "--question-path",
        config_path
    ]

    print(server_command)
    subprocess_env = os.environ.copy()

    """
    for k, v in generation_args.items():
        if k.startswith("GENERATION_"):
            subprocess_env[k] = str(v)
        else:
            if v == True or v == "True":
                server_command.append(f"--{k}")
            else:
                server_command.extend([f"--{k}", str(v)])
    """


    server_log_file_name = f"{SERVER_LOG_PREFIX}_{task.id}.txt"
    server_log_file = open(server_log_file_name, "w")
    print("Server command:", " ".join(server_command))
    print(f"GENERATION logs are located at: {server_log_file} in {os.getcwd()}")
    server_process = subprocess.Popen(server_command)
    #server_process = subprocess.Popen(server_command, stdout=server_log_file, stderr=server_log_file, shell=False, env=subprocess_env)
    time.sleep(300)
    """

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
    """
