from clearml import Task
import torch
from automation.utils import resolve_model_id, cast_args, to_plain_dict
from lighteval.main_vllm import vllm as lighteval_vllm
from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder
import yaml
from pyhocon import ConfigFactory
import json


def lighteval_main(
    model_id: str,
    lighteval_args: dict,
):
    # Determine number of gpus
    num_gpus = torch.cuda.device_count()

    # Add base_model_args to model_args
    model_args = lighteval_args.pop("model_args", {})
    model_args["model_name"] = model_id
    model_args["tensor_parallel_size"] = num_gpus

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


def main(configurations=None):
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    if configurations is None:
        lighteval_args = ConfigFactory.parse_string(task.get_configuration_object("lighteval_args"))
    else:
        lighteval_args = configurations.get("lighteval_args", {})
    model_id = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, force_download)

    results = lighteval_main(
        model_id=model_id,
        lighteval_args=lighteval_args,
    )

    dumped = json.dumps(results, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)

    task.upload_artifact(name="results", artifact_object=dumped)

    return results


if __name__ == '__main__':
    main()