from clearml import Task
import argparse

project_name = "Automation"
task_name = "lm_eval_vllm"
queue_name = "oneshot-a100x1"

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ lm-evaluation-harness using vllm backend")

parser.add_argument("--model_id", type=str)
parser.add_argument("--clearml_model", action="store_true", default=False)
parser.add_argument("--lm_eval", nargs=argparse.REMAINDER, help="lm_eval arguments")

args = parser.parse_args()

packages = [
    "git+https://github.com/neuralmagic/lm-evaluation-harness.git@llama_3.1_instruct", 
    "compressed-tensors",
    "sentencepiece",
    "vllm",
    "torch==2.5.1",
]

Task.force_store_standalone_script()

task = Task.init(project_name=project_name, task_name=task_name, task_type="inference")
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
task.set_packages(packages)

task.execute_remotely(queue_name)

#
# REMOTE
#

from clearml import InputModel
from glob import glob
import subprocess
import os
import json

lmeval_kwargs_unparsed = task.get_parameter("Args/lm_eval")
lmeval_kwargs_unparsed = json.loads(lmeval_kwargs_unparsed.replace("'", '"'))

lmeval_args = {}
for id, entry in enumerate(lmeval_kwargs_unparsed):
    value = None
    if entry.startswith("-"):
        if len(lmeval_kwargs_unparsed) > id+1:
            if not lmeval_kwargs_unparsed[id+1].startswith("-"):
                value = lmeval_kwargs_unparsed[id+1]
        lmeval_args[entry] = value

if args.clearml_model:
    input_model = InputModel(model_id=args.model_id)
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args.model_id

user_properties = task.get_user_properties()
queue_name_task = user_properties["k8s-queue"]["value"]

if "single" in queue_name_task or "x1" in queue_name_task:
    num_gpus = 1
elif "double" in queue_name_task or "x2" in queue_name_task:
    num_gpus = 2
elif "quad" in queue_name_task or "x4" in queue_name_task:
    num_gpus = 4
elif "octo" in queue_name_task or "x8" in queue_name_task:
    num_gpus = 8

model_args = f"pretrained={model_id},tensor_parallel_size={num_gpus}"
if "--model_args" in lmeval_args:
    lmeval_args["--model_args"] = model_args + ","+ lmeval_args["--model_args"]
else:
    lmeval_args["--model_args"] = model_args

if "--output_path" not in lmeval_args:
    lmeval_args["--output_path"] = "."

if "--write_out" not in lmeval_args:
    lmeval_args["--write_out"] = None

if "--show_config" not in lmeval_args:
    lmeval_args["--show_config"] = None


inputs = [
    "python3", "-m", "lm_eval", 
    "--model", "vllm", 
]

for k, v in lmeval_args.items():
    inputs.append(f"{k}")
    if v is not None:
        inputs.append(str(v))

subprocess.run(inputs)
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
