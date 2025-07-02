import os
import sys
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory

def main():
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

    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"

    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(args["Args"]["model"], clearml_model, force_download)

    gpu_count = int(guidellm_args.get("gpu_count", 1)) 

    print(vllm_args)
    print(model_id)
    print(guidellm_args["target"])
    print(args["Args"]["server_wait_time"])
    print(gpu_count)
    print(os.getcwd())

    from pathlib import Path
    from guidellm.benchmark.scenario import GenerativeTextScenario, get_builtin_scenarios
    filepath = Path(os.path.join(".", "src", "automation", "standards", "benchmarking", "rag.json"))
    current_scenario = GenerativeTextScenario.from_file(filepath, dict(guidellm_args))
    # Start vLLM server
    server_process, server_initialized, server_log = start_vllm_server(
        vllm_args,
        model_id,
        guidellm_args["target"],
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

    guidellm_args["model"] = model_id

    import json
    import asyncio
    from pathlib import Path
    #from guidellm.benchmark import benchmark_generative_text
    from guidellm.benchmark.output import GenerativeBenchmarksReport
    from guidellm.benchmark.entrypoints import benchmark_generative_text, benchmark_with_scenario
    from guidellm.benchmark.scenario import GenerativeTextScenario, get_builtin_scenarios

    # Ensure output_path is set and consistent
    output_path = Path(guidellm_args.get("output_path", "guidellm-output.json"))
    guidellm_args["output_path"] = str(output_path)

    print("[DEBUG] Calling benchmark_generative_text with:")
    print(json.dumps(guidellm_args, indent=2))

    #GenerativeBenchmarksReport()
    executable_path = os.path.dirname(sys.executable)
    vllm_path = os.path.join(executable_path, "vllm")
    print(f"The vllm path is: {vllm_path}")


    #default_scenario = get_builtin_scenarios()[0]
    #current_scenario = GenerativeTextScenario.from_builtin(default_scenario, dict(guidellm_args))

    #from pathlib import Path
    #filepath = Path(os.path.join(".", "src", "automation", "standards", "benchmarking", "chat.json"))
    #current_scenario = GenerativeTextScenario.from_file(filepath, dict(guidellm_args))

    #import time 
    #time.sleep(300)
    """
    current_scenario = GenerativeTextScenario
    print(current_scenario.model_fields["target"])
    print(current_scenario.model_fields["model"])
    overlap_keys = current_scenario.model_fields.keys() & dict(guidellm_args)
    #overlap_keys = ["model"]
    for element  in overlap_keys:
        #print(element)
        element_field_info = current_scenario.model_fields[element]
        element_field_info.default = guidellm_args[element]
        current_scenario.model_fields[element] = element_field_info
        #print(element_field_info.annotation)
    print(overlap_keys)

    print(current_scenario.model_fields["target"])
    print(current_scenario.model_fields["model"])

    current_scenario = GenerativeTextScenario
    """

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
