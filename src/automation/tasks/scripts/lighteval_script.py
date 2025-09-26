import torch
from automation.utils import resolve_model_id, cast_args, to_plain_dict
from lighteval.main_vllm import vllm as lighteval_vllm
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

    import nltk
    nltk.download("punkt")

    config = to_plain_dict(config)
    
    yaml.dump(config, open("lighteval_config.yaml", "w"))

    lighteval_args["save_details"] = True
    # Run lighteval
    lighteval_args = cast_args(lighteval_args, lighteval_vllm)
    results = lighteval_vllm(model_args="lighteval_config.yaml", **lighteval_args)

    if results is None:
        raise Exception("Evaluation failed.")

    return results


def main(configurations=None, args=None):

    import nltk
    nltk.data.path.append("/home")

    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)

    if clearml_available and configurations is None:
        lighteval_args = ConfigFactory.parse_string(task.get_configuration_object("lighteval_args"))
    else:
        lighteval_args = configurations.get("lighteval_args", {})
    
    model_name = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(model_name, clearml_model, force_download)

    results = lighteval_main(
        model_id=model_id,
        lighteval_args=lighteval_args,
    )

    dumped = json.dumps(results, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)

    if clearml_available:
        task.upload_artifact(name="results", artifact_object=dumped)

    # Generate filename with project name, task name, date and time    
    if clearml_available:
        project_name = task.get_project_name()
        task_name = task.get_name()
    else:
        project_name = "automation"
        lighteval_tasks = lighteval_args.get("tasks").replace(",", "_").replace(" ", "").replace("|", "_")
        task_name = f"lighteval_{model_name.replace('/', '_')}_{lighteval_tasks}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project_name}_{task_name}_{timestamp}.json"


    with open(filename, "w") as f:
        f.write(dumped)


    return results


if __name__ == '__main__':
    main()
