from clearml import Task
import argparse

parser = argparse.ArgumentParser(description = "Eval model w/ evalplus harness")

parser.add_argument("--model-id", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--benchmark-task", type=str, default="humaneval")
parser.add_argument("--disable-sanitize", action="store_true", default=False)
parser.add_argument("--num-samples", type=int, default=50)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=None)
parser.add_argument("--trust-remote-code", action="store_true", default=False)

args = parser.parse_args()

args = vars(args)
project_name = args.pop("project_name")
task_name = args.pop("task_name")

task = Task.init(project_name=project_name, task_name=task_name)

import subprocess
import os
import re
from clearml import InputModel
import torch
import evalplus

if args["num_gpus"] is None:
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = args["num_gpus"]

if args["clearml_model"]:
    input_model = InputModel(model_id=args["model_id"])
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args["model_id"]

batch_size = args["batch_size"]
temperature = args["temperature"]
benchmark_task = args["benchmark_task"]
num_samples = args["num_samples"]

evalplus_path = os.path.dirname(evalplus.__file__)
generate_path = os.path.join(evalplus_path, "codegen", "generate.py")

generation_inputs = [
    "python3", f"{generate_path}",
    "--model", f"{model_id}",
    "--bs", str(batch_size),
    "--temperature", str(temperature),
    "--n_samples", str(num_samples),
    "--root", ".",
    "--dataset", benchmark_task,
    "--tp", str(num_gpus),
]

if args["trust_remote_code"]:
    generation_inputs.extend(["--trust_remote_code", "True"])

subprocess.run(generation_inputs)

identifier = model_id.replace("/", "--") + f"_vllm_temp_{temperature}"
generations_path = os.path.join(benchmark_task, f"{identifier}")

subprocess.run(["tar", "-zcf" f"{identifier}.tar.gz", f"{generations_path}"])
task.upload_artifact(name=identifier, artifact_object=f"{identifier}.tar.gz")

if not args["disable_sanitize"]:
    subprocess.run(["python3", "evalplus/sanitize.py", f"{generations_path}"])
    sanitized_path = os.path.join(f"{benchmark_task}", f"{identifier}-sanitized")

    subprocess.run(["tar", "-zcf" f"{identifier}-sanitized.tar.gz", f"{sanitized_path}"])
    task.upload_artifact(name=identifier + "-sanitized", artifact_object=f"{identifier}-sanitized.tar.gz")
    generations_path = sanitized_path

output = subprocess.check_output(
    [
        "evalplus.evaluate",
        "--dataset", benchmark_task,
        "--samples", f"{generations_path}"
    ]
)
print(output)

output = output.decode("utf-8")

pass1_base = None
pass1_extra = None
pass10_base = None
pass10_extra = None

pass1 = re.findall(r'pass@1:\t([0-9.]+)', output)
if isinstance(pass1, list) and len(pass1) > 0:
    pass1_base = float(pass1[0])
    if len(pass1) > 1:
        pass1_extra = float(pass1[1])

pass10 = re.findall(r'pass@10:\t([0-9.]+)', output)
if isinstance(pass10, list) and len(pass1) > 0:
    pass10_base = float(pass10[0])
    if len(pass1) > 1:
        pass10_extra = float(pass10[1])


results = {
    "base_tests": {
        "pass@1": pass1_base if pass1_base else None,
        "pass@10": pass10_base if pass10_base else None
    },
    "base_plus_extra_tests": {
        "pass@1": pass1_extra if pass1_extra else None,
        "pass@10": pass10_extra if pass10_extra else None
    }
}

task.upload_artifact(name="results", artifact_object=results)

if pass1_base:
    task.get_logger().report_single_value(name="pass@1 base", value=pass1_base)
    if args["clearml_model"]:
        input_model.report_single_value(name="pass@1 base", value=pass1_base)

if pass10_base:
    task.get_logger().report_single_value(name="pass@10 base", value=pass10_base)
    if args["clearml_model"]:
        input_model.report_single_value(name="pass@10 base", value=pass10_base)

if pass1_extra:
    task.get_logger().report_single_value(name="pass@1 base + extra", value=pass1_extra)
    if args["clearml_model"]:
        input_model.report_single_value(name="pass@1 base + extra", value=pass1_extra)

if pass10_extra:
    task.get_logger().report_single_value(name="pass@10 base + extra", value=pass10_extra)
    if args["clearml_model"]:
        input_model.report_single_value(name="pass@10 base + extra", value=pass10_extra)
