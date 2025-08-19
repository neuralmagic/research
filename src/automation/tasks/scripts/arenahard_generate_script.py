import os
import sys
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree, render_yaml
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory
import subprocess
import requests
import time
import sys
import os
from urllib.parse import urlparse
from clearml import Task

SERVER_LOG_PREFIX = "generation_server_log"

STANDARDS_PATH = os.path.join(os.getcwd(), "src", "automation", "standards")
ARENAHARD_CONFIG_PATH = os.path.join(STANDARDS_PATH, "arenahard")

def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    raw_config = task.get_configuration_object("ArenaHard")
    if raw_config is None:
        print("[DEBUG] `ArenaHard` config not found in configuration — checking parameters as fallback")
        raw_config = task.get_parameters_as_dict().get("ArenaHard")
        if raw_config is None:
            raise RuntimeError("ArenaHard config is None. This likely means `get_configurations()` is not returning it or it's not passed via parameters.")
        arenahard_generate_args = ConfigFactory.from_dict(raw_config)
    else:
        arenahard_generate_args = ConfigFactory.parse_string(raw_config)

    def clean_hocon_value(v):
        if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v

    arenahard_generate_args = {k: clean_hocon_value(v) for k, v in arenahard_generate_args.items()}

    print("[DEBUG] Arenahard_Args:", arenahard_generate_args)

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

    gpu_count = int(arenahard_generate_args.get("gpu_count", 1))
    template_arenahard_file = "arena-hard-v2.0.yaml.j2"
    model_name = args["Args"]["generate_model"]
    get_lowercase_model = lambda model: model.split("/")[1].lower()
    
    
    render_yaml({"judge_model": get_lowercase_model(model_name), "max_tokens": 20000 }, STANDARDS_PATH , template_arenahard_file, template_arenahard_file[:-3])
    
    template_apiconfig_file = "api_config.yaml.j2"
    render_yaml({"model_name": model_name, "lower_case_model": get_lowercase_model(model_name), "max_tokens": 20000 }, STANDARDS_PATH , template_apiconfig_file, template_apiconfig_file[:-3])
    
    
    template_gen_answer_config_file = "gen_answer_config.yaml.j2"
    render_yaml({"lower_case_model": get_lowercase_model(model_name)}, STANDARDS_PATH , template_gen_answer_config_file, template_gen_answer_config_file[:-3])

    # verify that the input file paths exist
    api_config_path = os.path.join( STANDARDS_PATH, "api_config.yaml")
    #api_config_path = os.path.join( ARENAHARD_CONFIG_PATH , arenahard_generate_args["generation_endpoint_file"])
    assert os.path.exists(api_config_path), f"{api_config_path} does not exist"
    #gen_answer_config_path = os.path.join(ARENAHARD_CONFIG_PATH , arenahard_generate_args["generation_config_file"] )
    gen_answer_config_path = os.path.join(STANDARDS_PATH, "gen_answer_config.yaml")
    assert os.path.exists(gen_answer_config_path), f"{gen_answer_config_path} does not exist"

    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        arenahard_generate_args["target"],
        args["Args"]["server_wait_time"],
    )

    if not server_initialized:
        kill_process_tree(server_process.pid)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        raise AssertionError("Server failed to initialize")

    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    arenahard_generate_args["model"] = model_id

    import json
    import asyncio

    print("[DEBUG] Calling arena hard generate with:")
    print(json.dumps(arenahard_generate_args, indent=2))


    try:
        from arenahard.utils.completion import make_config
        configs = make_config(os.path.join(STANDARDS_PATH, arenahard_generate_args["generation_config_file"] ))
        print ("Running arena hard generate")
        from arenahard.gen_answer import run
        print(f"Arenahard args: {arenahard_generate_args}")

        run(config_file=arenahard_generate_args["generation_config_file"], endpoint_file=arenahard_generate_args["generation_endpoint_file"], question_path= ARENAHARD_CONFIG_PATH, config_path=ARENAHARD_CONFIG_PATH, answer_path=ARENAHARD_CONFIG_PATH)
        time.sleep(150)

    finally:
        from arenahard.utils.completion import load_model_answers
        from pathlib import Path
        model_name = configs["model_list"][0]
        output_file_path = os.path.join(ARENAHARD_CONFIG_PATH, "arena-hard-v2.0" , "model_answer", f"{model_name}.jsonl")
        arenahard_generate_args["output_path"] = str(output_file_path)
        task.upload_artifact(name="arenahard report", artifact_object=output_file_path)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        kill_process_tree(server_process.pid)


if __name__ == '__main__':
    main()
