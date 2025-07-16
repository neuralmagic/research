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

SERVER_LOG_PREFIX = "judgement_server_log"

ARENAHARD_CONFIG_PATH = os.path.join(os.getcwd(), "src", "automation", "standards", "arenahard")

def main():
    #from pathlib import Path
    #answer_task = Task.get_task(project_name="alexandre_debug",task_name="test_generate_task" )
    #artifact_obj = answer_task.artifacts['arenahard report'].get()
    #arenahard_answer_dir = Path(artifact_obj).parents[2]
    #print(f"The arenahard_answer_dir is {arenahard_answer_dir} at {artifact_obj}")
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    raw_config = task.get_configuration_object("ArenaHard")
    if raw_config is None:
        print("[DEBUG] `ArenaHard` config not found in configuration — checking parameters as fallback")
        raw_config = task.get_parameters_as_dict().get("ArenaHard")
        if raw_config is None:
            raise RuntimeError("ArenaHard config is None. This likely means `get_configurations()` is not returning it or it's not passed via parameters.")
        arenahard_judgement_args = ConfigFactory.from_dict(raw_config)
    else:
        arenahard_judgement_args = ConfigFactory.parse_string(raw_config)

    def clean_hocon_value(v):
        if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v

    arenahard_judgement_args = {k: clean_hocon_value(v) for k, v in arenahard_judgement_args.items()}

    print("[DEBUG] Arenahard_Args:", arenahard_judgement_args)

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

    gpu_count = int(arenahard_judgement_args.get("gpu_count", 1))

    # verify that the input file paths exist
    api_config_path = os.path.join( ARENAHARD_CONFIG_PATH , arenahard_judgement_args["judgement_endpoint_file"])
    assert os.path.exists(api_config_path), f"{api_config_path} does not exist"
    gen_judgement_config_path = os.path.join(ARENAHARD_CONFIG_PATH , arenahard_judgement_args["judgement_setting_file"] )
    assert os.path.exists(gen_judgement_config_path), f"{gen_judgement_config_path} does not exist"

    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        arenahard_judgement_args["target"],
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

    arenahard_judgement_args["model"] = model_id

    import json
    import asyncio

    print("[DEBUG] Calling arena hard generate with:")
    print(json.dumps(arenahard_judgement_args, indent=2))


    try:
        from pathlib import Path

        answer_task = Task.get_task(project_name="alexandre_debug",task_name="test_generate_task" )
        artifact_obj = answer_task.artifacts['arenahard report'].get_local_copy()
        #arenahard_answer_dir = Path(artifact_obj).parents[2]

        import shutil
        from pathlib import Path
        import os

        answer_path = Path(os.path.join(ARENAHARD_CONFIG_PATH, "arena-hard-v2.0", "model_answer"))
        #answer_path = Path("/home/ubuntu/arena-research.git/src/automation/standards/arenahard/arena-hard-v2.0/model_answer/")
        os.makedirs(answer_path, exist_ok=True)
        shutil.move(artifact_obj,os.path.join(answer_path, "qwen2.5-1.5b-instruct.jsonl"))

        print ("Running arena hard generate")
        from arenahard.gen_judgment import run
        print(f"Arenahard args: {arenahard_judgement_args}")

        run(setting_file=arenahard_judgement_args["judgement_setting_file"], endpoint_file=arenahard_judgement_args["judgement_endpoint_file"], question_path= ARENAHARD_CONFIG_PATH, config_path=ARENAHARD_CONFIG_PATH, answer_path=arenahard_answer_dir)
        time.sleep(150)

    finally:
        output_path = os.path.join(os.getcwd(), "src", "automation", "arenahard", "data", "arena-hard-v2.0", "model_answer", "qwen2.5-1.5b-instruct.jsonl")
        arenahard_judgement_args["output_path"] = str(output_path)
        #task.upload_artifact(name="arenahard judgement report", artifact_object=output_path)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        kill_process_tree(server_process.pid)


if __name__ == '__main__':
    main()
