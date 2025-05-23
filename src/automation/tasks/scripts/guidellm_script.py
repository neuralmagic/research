
import os
from clearml import Task
from automation.utils import resolve_model_id, cast_args, kill_process_tree
from automation.vllm import start_vllm_server
from pyhocon import ConfigFactory


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
        raise AssertionError("Server failed to intialize")

    # Parse through environment variables
    for k, v in environment_args.items():
        os.environ[k] = str(v)

    guidellm_args["model"] = model_id

    from guidellm import generate_benchmark_report
    guidellm_args = cast_args(guidellm_args, generate_benchmark_report)
    report = generate_benchmark_report(**guidellm_args)
    kill_process_tree(server_process.pid)

    task.upload_artifact(name="guidellm guidance report", artifact_object=report.to_json())
    task.upload_artifact(name="vLLM server log", artifact_object=server_log)

if __name__ == '__main__':
    main()