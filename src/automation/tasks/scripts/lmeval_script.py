from clearml import Task
import torch
from automation.utils import resolve_model_id, cast_args
import lm_eval
import json
from transformers import AutoModelForCausalLM


def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    lm_eval_args = args["lm_eval"]
    model_id = args["Args"]["model_id"]
    clearml_model = args["Args"]["clearml_model"]
    if isinstance(clearml_model, str):
        clearml_model = clearml_model.lower() == "true"
    force_download = args["Args"]["force_download"]
    if isinstance(force_download, str):
        force_download = force_download.lower() == "true"

    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, task)

    if force_download:
        AutoModelForCausalLM.from_pretrained(model_id, force_download=True,trust_remote_code=True)

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
    task_manager = lm_eval.tasks.TaskManager()
    lm_eval_args = cast_args(lm_eval_args, lm_eval.simple_evaluate)
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model="vllm",
        task_manager=task_manager,
        **lm_eval_args,
    )

    if results is None:
        raise Exception("Evaluation failed.")
    
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
                task.get_logger().report_scalar(title=lm_eval_task, series=metric, iteration=num_fewshot, value=value)

    # Print results to console
    print(lm_eval.utils.make_table(results))

    if "groups" in results:
        print(lm_eval.utils.make_table(results, "groups"))

    # Upload results to ClearML
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