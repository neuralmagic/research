import os
import glob
import json
from clearml import Task
import torch
from automation.utils import resolve_model_id, dict_to_argparse
from lm_eval import cli_evaluate


def upload_results(model_id, task):
    model_suffix = os.path.split(model_id)[-1]
    
    if len(glob(f"*{model_suffix}")) > 0:
        results_dir = glob(f"*{model_suffix}")[-1]
        json_file = glob(os.path.join(results_dir, "results*.json"))[0]

        results = json.load(open(json_file))
        artifact = {"name": json_file, "object": results}
        task.upload_artifact(name=json_file, artifact_object=results)
        
        scalars = []
        if "results" in results:
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
    else:
        raise Exception("No results found. Evaluation probably failed.")


def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)["Args"]
    lm_eval_args = args["lm_eval"]

    # Resolve model_id
    model_id = resolve_model_id(args["model_id"], args["clearml_model"], task)

    # Determine number of gpus
    num_gpus = torch.cuda.device_count()

    base_model_args = f"pretrained={model_id},tensor_parallel_size={num_gpus}"

    # Add base_model_args to model_args
    if "model_args" in lm_eval_args:
        lm_eval_args["model_args"] += f",{base_model_args}"
    else:
        lm_eval_args["model_args"] = base_model_args

    # Convert lm_eval_args into argsparse namespace
    lm_eval_args = dict_to_argparse(lm_eval_args)

    # Run lm_eval
    cli_evaluate(lm_eval_args)

    # Upload results to clearml
    upload_results(model_id, task)

if __name__ == '__main__':
    main()