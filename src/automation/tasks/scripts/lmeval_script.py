from clearml import Task
import torch
from automation.utils import resolve_model_id, cast_args
import lm_eval


def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    lm_eval_args = args["lm_eval"]
    model_id = args["Args"]["model_id"]
    clearml_model = bool(args["Args"]["clearml_model"])

    # Resolve model_id
    model_id = resolve_model_id(model_id, clearml_model, task)

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
    
    # Upload results to ClearML
    task.upload_artifact(name="results", artifact_object=results)

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

    # Print results to console
    print(lm_eval.utils.make_table(results))

    if "groups" in results:
        print(lm_eval.utils.make_table(results, "groups"))


if __name__ == '__main__':
    main()