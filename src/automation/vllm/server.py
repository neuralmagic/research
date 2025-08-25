import subprocess
import requests
import time
import sys
import os
import torch
from urllib.parse import urlparse
from automation.utils import kill_process_tree
#from clearml import Task

SERVER_LOG_PREFIX = "vllm_server_log"

class VLLMServer:
    def __init__(self, vllm_args, model_id, target, server_wait_time):
        self.vllm_args = vllm_args
        self.model_id = model_id
        self.target = target
        self.server_wait_time = server_wait_time
        
    def start(self):
        #task = Task.current_task()

        executable_path = os.path.dirname(sys.executable)
        vllm_path = os.path.join(executable_path, "vllm")

        num_gpus = torch.cuda.device_count()

        parsed_target = urlparse(self.target)

        server_command = [
            f"{vllm_path}", "serve", 
            self.model_id,
            "--host", parsed_target.hostname, 
            "--port", str(parsed_target.port),
            "--tensor-parallel-size", str(num_gpus)
        ]

        subprocess_env = os.environ.copy()

        for k, v in self.vllm_args.items():
            if k.startswith("VLLM_"):
                subprocess_env[k] = str(v)
            else:
                if v == True or v == "True":
                    server_command.append(f"--{k}")
                else:
                    server_command.extend([f"--{k}", str(v)])
                    

        #server_log_file_name = f"{SERVER_LOG_PREFIX}_{task.id}.txt"
        self.server_log_file_name = f"{SERVER_LOG_PREFIX}.txt"
        self.server_log_file = open(self.server_log_file_name, "w")
        #server_process = subprocess.Popen(server_command, shell=False, env=subprocess_env)
        self.server_process = subprocess.Popen(server_command, stdout=self.server_log_file, stderr=self.server_log_file, shell=False, env=subprocess_env)

        delay = 5
        self.server_initialized = False
        for _ in range(self.server_wait_time // delay):
            try:
                response = requests.get(self.target + "/models")
                if response.status_code == 200 and response.json().get("data"):
                    print("Server initialized")
                    self.server_initialized = True
                    break  # Exit the loop if the request is successful
            except requests.exceptions.RequestException as e:
                pass

            time.sleep(delay)
        
    def stop(self):
        kill_process_tree(self.server_process.pid)
        self.server_log_file.close()
        
    def is_initialized(self):
        return self.server_initialized
    
    def get_log_file_name(self):
        return self.server_log_file_name
    
    def get_log_file(self):
        return self.server_log_file
    

 
