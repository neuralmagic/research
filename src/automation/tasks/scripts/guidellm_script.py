import os
import sys
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory

"""
def main(configurations=None):
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    raw_config = task.get_configuration_object("GuideLLM")
    if raw_config is None:
        print("[DEBUG] `GuideLLM` config not found in configuration — checking parameters as fallback")
        raw_config = task.get_parameters_as_dict().get("GuideLLM")
        if raw_config is None:
            raise RuntimeError("GuideLLM config is None. This likely means `get_configurations()` is not returning it or it's not passed via parameters.")
        guidellm_args = ConfigFactory.from_dict(raw_config)
    else:
        guidellm_args = ConfigFactory.parse_string(raw_config)

    def clean_hocon_value(v):
        if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v

    guidellm_args = {k: clean_hocon_value(v) for k, v in guidellm_args.items()}

    print("[DEBUG] Guidellm_Args:", guidellm_args)

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
"""

def main(configurations=None):
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    
    if configurations is None:
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
    else:
        guidellm_args = configurations.get("GuideLLM", {})
        environment_args = configurations.get("environment", {})
        vllm_args = configurations.get("vLLM", {})

    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"

    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(args["Args"]["model"], clearml_model, force_download)

    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        guidellm_args["target"],
        args["Args"]["server_wait_time"],
    )

    if not server_initialized:
        kill_process_tree(server_process.pid)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        raise AssertionError("Server failed to initialize")

    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    guidellm_args["model"] = model_id

    import json
    import asyncio
    from pathlib import Path
    from guidellm.benchmark.entrypoints import benchmark_with_scenario
    from guidellm.benchmark.scenario import GenerativeTextScenario, get_builtin_scenarios

    # user defined scenarios are a temporary fix until the guidellm bugs get fixed otherwise we would use the upstream scenarios
    user_scenario = guidellm_args.get("scenario", "")
    if user_scenario:
        filepath = Path(os.path.join(".", "src", "automation", "standards", "benchmarking", f"{user_scenario}.json"))
        if os.path.exists(filepath):
            current_scenario = GenerativeTextScenario.from_file(filepath, dict(guidellm_args))
        else:
            raise ValueError(f"Scenario path {filepath} does not exist")
    #elif len(get_builtin_scenarios()) > 0:
    #    to be used when get_builtin_scenarios() bug is fixed
    #    current_scenario = GenerativeTextScenario.from_builtin(get_builtin_scenarios()[0], dict(guidellm_args))
    else:
        filepath = Path(os.path.join(".", "src", "automation", "standards", "benchmarking", f"{user_scenario}.json"))
        current_scenario = GenerativeTextScenario.from_file(filepath, dict(guidellm_args))

    # Ensure output_path is set and consistent
    output_path = Path(guidellm_args.get("output_path", "guidellm-output.json"))
    guidellm_args["output_path"] = str(output_path)

    print("[DEBUG] Calling benchmark_with_scenario with:")
    print(json.dumps(guidellm_args, indent=2))

    executable_path = os.path.dirname(sys.executable)
    vllm_path = os.path.join(executable_path, "vllm")
    print(f"The vllm path is: {vllm_path}")

    try:
        asyncio.run(
            benchmark_with_scenario(
                current_scenario,
                output_path= output_path,
                output_extras= None
            )
        )

    finally:
        task.upload_artifact(name="guidellm guidance report", artifact_object=output_path)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)
        kill_process_tree(server_process.pid)

if __name__ == '__main__':
    main()
