from clearml import Task
import argparse

project_name = "Automation"
task_name = "evalplus_vllm"
queue_name = "oneshot-a100x1"

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Eval model w/ evalplus harness")

parser.add_argument("--model-id", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--benchmark-task", type=str, default="humaneval")
parser.add_argument("--disable-sanitize", action="store_true", default=False)
parser.add_argument("--num-samples", type=int, default=50)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--build-vllm", action="store_true", default=False)

args = parser.parse_args()

args = vars(args)
additional_packages = args.pop("packages")
build_vllm = args.pop("build_vllm")

packages = [
    "git+https://github.com/neuralmagic/evalplus.git@master", 
    "sentencepiece",
]

if build_vllm:
    packages.append("git+https://github.com/vllm-project/vllm.git@main")
else:
    packages.append("vllm")

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-torch:latest")
task.set_script(repository="https://github.com/neuralmagic/evalplus.git", branch="master")
task.set_packages(packages)

task.execute_remotely(queue_name)

#
# REMOTE
#

import subprocess
import os
import re
from clearml import InputModel

if args["clearml_model"]:
    input_model = InputModel(model_id=args["model_id"])
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args["model_id"]

user_properties = task.get_user_properties()
queue_name_task = user_properties["k8s-queue"]["value"]

if "single" in queue_name or "x1" in queue_name:
    num_gpus = 1
elif "double" in queue_name or "x2" in queue_name:
    num_gpus = 2
elif "quad" in queue_name or "x4" in queue_name:
    num_gpus = 4
elif "octo" in queue_name or "x8" in queue_name:
    num_gpus = 8

batch_size = args["batch_size"]
temperature = args["temperature"]
benchmark_task = args["benchmark_task"]
num_samples = args["num_samples"]

generation_inputs = [
    "python3", "codegen/generate.py",
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
        "python3", "-m", "evalplus.evaluate",
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