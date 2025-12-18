import torch
from automation.utils import resolve_model_id, cast_args, to_plain_dict, load_callable_configuration
from automation.vllm import VLLMServer
from lighteval.main_vllm import vllm as lighteval_vllm
from lighteval.main_endpoint import litellm as lighteval_litellm
from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder
import yaml
from pyhocon import ConfigFactory
import json
from datetime import datetime

try:
    from clearml import Task
    clearml_available = True
except ImportError:
    clearml_available = False

def lighteval_vllm_main(
    model_id: str,
    lighteval_args: dict,
):    
    # Determine number of gpus
    num_gpus = torch.cuda.device_count()

    # Add base_model_args to model_args
    model_args = lighteval_args.pop("model_args", {})
    model_args["model_name"] = model_id
    model_args["tensor_parallel_size"] = num_gpus


    # Set default dtype
    if "dtype" not in model_args:
        model_args["dtype"] = "auto"


    config = {"model_parameters": model_args}
    if "metric_options" in lighteval_args:
        config["metric_options"] = lighteval_args.pop("metric_options")

    config = to_plain_dict(config)
    
    yaml.dump(config, open("lighteval_config.yaml", "w"))

    lighteval_args["save_details"] = True
    # Run lighteval
    lighteval_args = cast_args(lighteval_args, lighteval_vllm)
    results = lighteval_vllm(model_args="lighteval_config.yaml", **lighteval_args)

    if results is None:
        raise Exception("Evaluation failed.")

    return results


def lighteval_litellm_main(
    model_id: str,
    lighteval_args: dict,
    vllm_args: dict,
    base_url: str,
    server_wait_time: int,
):    

    # Start vLLM server
    vllm_server = VLLMServer(
        vllm_args=vllm_args,
        model_id=model_id,
        target=base_url,
        server_wait_time=server_wait_time,
    )
    vllm_server.start()
    if vllm_server.is_initialized():
        print("VLLM server initialized")
    else:
        if clearml_available:
            task = Task.current_task()
            task.upload_artifact(name="vLLM server log", artifact_object=vllm_server.get_log_file_name())
        vllm_server.stop()
        raise Exception("VLLM server failed to initialize")
    
    # Add base_model_args to model_args
    model_args = lighteval_args.pop("model_args", {})
    model_args["provider"] = "hosted_vllm"
    model_args["model_name"] = f"hosted_vllm/{model_id}"
    model_args["base_url"] = base_url
    model_args["api_key"] = ""


    config = {"model_parameters": model_args}
    if "metric_options" in lighteval_args:
        config["metric_options"] = lighteval_args.pop("metric_options")

    config = to_plain_dict(config)

    yaml.dump(config, open("lighteval_config.yaml", "w"))

    lighteval_args["save_details"] = True
    # Run lighteval
    lighteval_args = cast_args(lighteval_args, lighteval_litellm)
    results = lighteval_litellm(model_args="lighteval_config.yaml", **lighteval_args)

    if results is None:
        raise Exception("Evaluation failed.")

    return results

def main(configurations=None, args=None):
    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)

    if clearml_available and configurations is None:
        lighteval_args = ConfigFactory.parse_string(task.get_configuration_object("lighteval_args"))
        pretask_callback = task.get_configuration_object("pretask callback")
    else:
        lighteval_args = configurations.get("lighteval_args", {})
        pretask_callback = configurations.get("pretask callback", None)
    
    if pretask_callback is not None:
        pretask_callback_fn = load_callable_configuration("pretask callback", configurations)
        pretask_callback_fn()

    model_name = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(model_name, clearml_model, force_download)

    # Resolve entrypoint
    entrypoint = args["Args"]["entrypoint"]

    if entrypoint == "vllm":
        results = lighteval_vllm_main(
            model_id=model_id,
            lighteval_args=lighteval_args,
        )
    elif entrypoint == "litellm":
        server_wait_time = args["Args"]["server_wait_time"]
        base_url = args["Args"]["base_url"]
        if clearml_available and configurations is None:
            vllm_args = ConfigFactory.parse_string(task.get_configuration_object("vLLM"))
        else:
            vllm_args = configurations.get("vLLM", {})
        results = lighteval_litellm_main(
            model_id=model_id,
            lighteval_args=lighteval_args,
            vllm_args=vllm_args,
            base_url=base_url,
            server_wait_time=server_wait_time,
        )
    else:
        raise ValueError(f"Invalid entrypoint: {entrypoint}")

    dumped = json.dumps(results, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)

    # Generate filename with project name, task name, date and time    
    if clearml_available:
        project_name = task.get_project_name()
        task_name = task.name
    else:
        project_name = "automation"
        lighteval_tasks = lighteval_args.get("tasks").replace(",", "_").replace(" ", "").replace("|", "_")
        task_name = f"lighteval_{model_name.replace('/', '_')}_{lighteval_tasks}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project_name}_{task_name}_{timestamp}.json"


    with open(filename, "w") as f:
        f.write(dumped)

    if clearml_available:
        task.upload_artifact(name="results", artifact_object=dumped)

    return results


if __name__ == '__main__':
    main()
