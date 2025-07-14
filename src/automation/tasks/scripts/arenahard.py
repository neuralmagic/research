import os
import sys
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory
import subprocess
import requests
import time
import sys
import os
from urllib.parse import urlparse
from clearml import Task
from arenahard.gen_answer import run

SERVER_LOG_PREFIX = "generation_server_log"


def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    raw_config = task.get_configuration_object("ArenaHard")
    if raw_config is None:
        print("[DEBUG] `ArenaHard` config not found in configuration — checking parameters as fallback")
        raw_config = task.get_parameters_as_dict().get("ArenaHard")
        if raw_config is None:
            raise RuntimeError("ArenaHard config is None. This likely means `get_configurations()` is not returning it or it's not passed via parameters.")
        arenahard_args = ConfigFactory.from_dict(raw_config)
    else:
        arenahard_args = ConfigFactory.parse_string(raw_config)

    def clean_hocon_value(v):
        if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v

    arenahard_args = {k: clean_hocon_value(v) for k, v in arenahard_args.items()}

    print("[DEBUG] Arenahard_Args:", arenahard_args)

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

    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(args["Args"]["generate_model"], clearml_model, force_download)

    gpu_count = int(arenahard_args.get("gpu_count", 1)) 

    from pathlib import Path
    print("Inside start generation server")
    executable_path = os.path.dirname(sys.executable)
    python_path = os.path.join(executable_path, "python3")
    print(f"python path is: {python_path}")
    base_path = Path(executable_path)
    sitepackages_path = os.path.join(base_path.parents[0], "lib", "python3.10", "site-packages")
    generation_path = os.path.join(sitepackages_path, "arenahard", "gen_answer.py")
    assert os.path.exists(generation_path), f"{generation_path} does not exist"
    config_path = os.path.join(os.getcwd(), "src", "automation", "standards", "arenahard")
    api_config_path = os.path.join(config_path, "api_config.yaml")
    assert os.path.exists(api_config_path), f"{api_config_path} does not exist"
    gen_answer_config_path = os.path.join(config_path, "gen_answer_config.yaml")
    assert os.path.exists(gen_answer_config_path), f"{gen_answer_config_path} does not exist"

    arenahard_path = os.path.join(sitepackages_path, "arenahard", "gen_answer.py")
    #os.environ['PYTHONPATH'] = f"{arenahard_path}:" + os.environ.get('PYTHONPATH','')
    #sys.path.append(sitepackages_path)
    sys.path.append(python_path)

    #from arenahard.gen_answer import run
    run (config_file = 'custom_gen_answer_config.yaml',  endpoint_file='custom_api_config.yaml', question_path = config_path,  config_path = config_path, answer_path = config_path )
    time.sleep(300)

    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        arenahard_args["target"],
        args["Args"]["server_wait_time"],
        gpu_count,
    )

    if not server_initialized:
        kill_process_tree(server_process.pid)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        raise AssertionError("Server failed to initialize")

    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    arenahard_args["model"] = model_id

    import json
    import asyncio
    output_path = os.path.join(os.getcwd(), "src", "automation", "arenahard", "data", "arena-hard-v2.0", "model_answer", "qwen2.5-1.5b-instruct.jsonl")
    arenahard_args["output_path"] = str(output_path)

    print("[DEBUG] Calling arena hard with:")
    print(json.dumps(arenahard_args, indent=2))

    executable_path = os.path.dirname(sys.executable)
    vllm_path = os.path.join(executable_path, "vllm")
    print(f"The vllm path is: {vllm_path}")

    try:
        print ("Running arena hard")

    finally:
        task.upload_artifact(name="arenahard report", artifact_object=output_path)
        #task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        #kill_process_tree(server_process.pid)


if __name__ == '__main__':
    main()
