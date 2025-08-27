from clearml import Task
import torch
from automation.utils import resolve_model_id, cast_args
import lm_eval
import numpy
import json
from pyhocon import ConfigFactory


def average_groups(results:dict, groups:dict):
    task = Task.current_task()
    if len(task.get_models()["input"]) == 1:
        clearml_model_handle = task.get_models()["input"][0]
    else:
        clearml_model_handle = None

    def compute_average(metric_name: str, metric_config: dict):
        if metric_name in results["results"] and "series" in metric_config:
            score = results["results"][metric_name][metric_config["series"]]
            weight = metric_config.get("weight", 1.0)
            normalize = metric_config.get("normalize", False)
        
            if normalize:
                score = (score - metric_config["random_score"]) / (1.0 - metric_config["random_score"])

            return score * weight
        else:
            scores = []
            for _metric_name, _options in metric_config.items():
                scores.append(compute_average(_metric_name, _options))
            average_score = numpy.mean(scores).item()

            task.get_logger().report_single_value(name=metric_name, value=average_score)
            task.get_logger().report_scalar(title=metric_name, series="average", iteration=0, value=average_score)

            if clearml_model_handle is not None:
                clearml_model_handle.report_single_value(name=metric_name, value=average_score)

            results["results"][metric_name] = {"average": average_score}

            return average_score

    for metric_name, metric_config in groups.items():
        compute_average(metric_name, metric_config)

    return results


def lmeval_main(
    model_id: str,
    lm_eval_args: dict,
    groups: dict = None,
):
    # Determine number of gpus
    num_gpus = torch.cuda.device_count()

    base_model_args = f"pretrained={model_id},tensor_parallel_size={num_gpus}"

    # Add base_model_args to model_args
    if "model_args" in lm_eval_args:
        lm_eval_args["model_args"] += f",{base_model_args}"
    else:
        lm_eval_args["model_args"] = base_model_args

    lm_eval_args["write_out"] = True

    # Run lm_eval
    #task_manager = lm_eval.tasks.TaskManager()
    from lm_eval.tasks import TaskManager
    task_manager = TaskManager()
    #tasks = task_manager.all_tasks

    lm_eval_args = cast_args(lm_eval_args, lm_eval.simple_evaluate)
    results = lm_eval.simple_evaluate( # call simple_evaluate
        task_manager=task_manager,
        apply_chat_template=True,
        **lm_eval_args,
    )

    if results is None:
        raise Exception("Evaluation failed.")

    # Print results to console
    print(lm_eval.utils.make_table(results))

    if "groups" in results:
        print(lm_eval.utils.make_table(results, "groups"))
        
    if groups is not None and len(groups) > 0:
        results = average_groups(results, groups)

    return results


def main(configurations=None):
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    if configurations is None:
        lm_eval_args = ConfigFactory.parse_string(task.get_configuration_object("lm_eval"))
    else:
        lm_eval_args = configurations.get("lm_eval", {})
    model_id = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"
    groups = lm_eval_args.pop("groups", None)

    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, force_download)

    results = lmeval_main(
        model_id=model_id,
        lm_eval_args=lm_eval_args,
        groups=groups,
    )

    # Upload results to ClearML
    if len(task.get_models()["input"]) == 1:
        clearml_model_handle = task.get_models()["input"][0]
    else:
        clearml_model_handle = None

    for lm_eval_task in results["results"]:
        if "configs" in results and lm_eval_task in results["configs"] and "num_fewshot" in results["configs"][lm_eval_task]:
            num_fewshot = results["configs"][lm_eval_task]["num_fewshot"]
        else:
            num_fewshot = None
        for metric in results["results"][lm_eval_task]:
            value = results["results"][lm_eval_task][metric]
            if not isinstance(value, str):
                if num_fewshot is None:
                    name = lm_eval_task + "/" + metric
                else:
                    name = lm_eval_task + "/" + f"{num_fewshot:d}" + "shot/" + metric
                task.get_logger().report_single_value(name=name, value=value)
                task.get_logger().report_scalar(title=lm_eval_task, series=metric, iteration=0, value=value)

                if clearml_model_handle is not None:
                    clearml_model_handle.report_single_value(name=name, value=value)
                           
    dumped = json.dumps(
        results,
        indent=2,
        default=lm_eval.utils.handle_non_serializable,
        ensure_ascii=False,
    )

    task.upload_artifact(name="results", artifact_object=dumped)

    return results


if __name__ == '__main__':
    main()
