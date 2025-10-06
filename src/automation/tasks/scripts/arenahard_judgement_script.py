import os
import sys
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree, render_yaml
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory
import subprocess
import requests
import time
from urllib.parse import urlparse
from clearml import Task

SERVER_LOG_PREFIX = "judgement_server_log"

STANDARDS_PATH = os.path.join(os.getcwd(), "src", "automation", "standards")
ARENAHARD_CONFIG_PATH = os.path.join(STANDARDS_PATH, "arenahard")

def main():
    from pathlib import Path
    import shutil
    import os
    from arenahard.utils.completion import make_config

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

    bench_name = arenahard_judgement_args["bench_name"]

    # Resolve model_id
    model_id = resolve_model_id(args["Args"]["judgement_model"], clearml_model, force_download)

    model_name = args["Args"]["judgement_model"]
    get_lowercase_model = lambda model: model.split("/")[1].lower()

    template_apiconfig_file = "api_config.yaml.j2"
    tmp_judge_endpoint_file='tmp_api_config.yaml'

    template_arenahard_file = f"{bench_name}.yaml.j2"
    tmp_arenahard_file = f'tmp_{bench_name}.yaml'
    
    render_yaml({"judge_model": get_lowercase_model(model_name), "max_tokens": arenahard_judgement_args["max_tokens"] }, STANDARDS_PATH , template_arenahard_file, tmp_arenahard_file)

    render_yaml({"model_name": model_name, "lower_case_model": get_lowercase_model(model_name), "max_tokens": arenahard_judgement_args["max_tokens"], "api_base": f"'{arenahard_judgement_args['target']}'", "api_key": arenahard_judgement_args.get("api_key", "'-'")}, STANDARDS_PATH , template_apiconfig_file, tmp_judge_endpoint_file )

    # verify that the input file paths exist
    api_config_path = os.path.join( ARENAHARD_CONFIG_PATH , tmp_judge_endpoint_file )
    assert os.path.exists(api_config_path), f"{api_config_path} does not exist"
    gen_judgement_config_path = os.path.join(ARENAHARD_CONFIG_PATH , tmp_arenahard_file )
    assert os.path.exists(gen_judgement_config_path), f"{gen_judgement_config_path} does not exist"

    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        arenahard_judgement_args["target"],
        args["Args"]["server_wait_time"]
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
        arenahard_dir = Path(os.path.join(ARENAHARD_CONFIG_PATH, bench_name ))
        answer_dir = os.path.join(arenahard_dir, "model_answer")
        from arenahard.utils.completion import make_config
        configs = make_config(os.path.join(ARENAHARD_CONFIG_PATH, tmp_arenahard_file))
        model_name = configs["model_list"][0]
        if arenahard_judgement_args.get("answer_task_name","") :
            from pathlib import Path
            import shutil
            import os
            from clearml.storage import StorageManager

            answer_task = Task.query_tasks(project_name=arenahard_judgement_args.get("answer_project_name", task.get_project_name() ),task_name=arenahard_judgement_args["answer_task_name"], task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
            answer_task = Task.get_task(answer_task[0])
            artifact_obj = answer_task.artifacts['arenahard model answer'].get_local_copy()
            shutil.move(artifact_obj,os.path.join(answer_dir, f"{model_name}.jsonl"))
        else:
            # use default 03-mini answers
            shutil.copy( os.path.join(answer_dir,"o3-mini-2025-01-31.jsonl"),os.path.join(answer_dir, f"{model_name}.jsonl"))

        if arenahard_judgement_args.get("question_size","") == "small" :
            shutil.copy( os.path.join(arenahard_dir,"shortquestion.jsonl"),os.path.join(arenahard_dir, "question.jsonl"))
    
        print ("Running arena hard generate")
        from arenahard.gen_judgment import run
        print(f"Arenahard args: {arenahard_judgement_args}")

        run(setting_file= tmp_arenahard_file, endpoint_file= tmp_judge_endpoint_file, question_path= ARENAHARD_CONFIG_PATH, config_path=ARENAHARD_CONFIG_PATH, answer_path=ARENAHARD_CONFIG_PATH)
        time.sleep(150)

    finally:
        output_path = os.path.join(answer_dir, f"{model_name}.jsonl")
        arenahard_judgement_args["output_path"] = str(output_path)
        #task.upload_artifact(name="arenahard judgement report", artifact_object=output_path)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        kill_process_tree(server_process.pid)


if __name__ == '__main__':
    main()
