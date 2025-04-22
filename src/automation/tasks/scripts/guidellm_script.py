
import os
import sys

# Forcefully bypass ClearML's click patch monkey
sys.modules["clearml.binding.click_bind"] = None
os.environ["CLEARML_NO_FRAMEWORKS"] = "1"
os.environ["CLEARML_AGENT_SKIP_PYTHON_ENV_CACHE"] = "1"
os.environ["CLEARML_DISABLE_CLICK_PATCH"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
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
        raise AssertionError("Server failed to intialize")
    
    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    guidellm_args["model"] = model_id

    import sys
    import json
    from pathlib import Path
    from guidellm.__main__ import cli

    # Ensure output_path is set and consistent
    output_path = Path(guidellm_args.get("output_path", "guidellm-output.json"))
    guidellm_args["output_path"] = str(output_path)

    print("[DEBUG] target value type:", type(guidellm_args["target"]))
    print("[DEBUG] target value:", guidellm_args["target"])

    # Build sys.argv to mimic CLI usage
    sys.argv = ["guidellm", "benchmark"]
    for k, v in guidellm_args.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                sys.argv.append(flag)
        elif isinstance(v, (str, int, float)):
            sys.argv += [flag, str(v)]
        else:
            print(f"[WARN] Skipping CLI arg {k} due to unsupported type: {type(v)} = {v}")

    print("[DEBUG] sys.argv constructed:")
    print(sys.argv)


    try:
        # Run CLI benchmark (will save output to output_path)
        cli()
    finally:
        # Load the output benchmark report as JSON
        with open(output_path, "r") as f:
            report = json.load(f)

        task.upload_artifact(name="guidellm guidance report", artifact_object=output_path)
        task.upload_artifact(name="vLLM server log", artifact_object=server_log)

        kill_process_tree(server_process.pid)



if __name__ == '__main__':
    main()