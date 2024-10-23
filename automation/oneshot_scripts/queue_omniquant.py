
from clearml import Task
import argparse
import subprocess
import os
import sys
from urllib.parse import urlparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description="OmniQuant")

parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--packages", type=str, nargs="+", default=None)

# OmniQuant specific arguments
parser.add_argument("--model", type=str, required=True, help="Model identifier for OmniQuant")
parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity")
parser.add_argument("--generate", action="store_true", help="Generate")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--wbits", type=int, default=4, help="Weight bits for quantization")
parser.add_argument("--abits", type=int, default=16, help="Activation bits for quantization")
parser.add_argument("--nsamples", type=int, default=128, help="Number of Samples")
parser.add_argument("--lwc", action="store_true", help="Use lightweight quantization")
#parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of evaluation tasks")
parser.add_argument("--group_size", type=int, default=128, help="Group Size")
parser.add_argument("--save_dir", type=str, required=True, help="Save directory for HF model")

args, unparsed_args = parser.parse_known_args()

project_name = args.project_name
task_name = args.task_name
queue_name = args.queue_name
additional_packages = args.packages

# Prepare OmniQuant-specific command arguments
omniquant_args = {
    "model": args.model,
    "eval_ppl": "--eval_ppl" if args.eval_ppl else "",
    "generate": "--generate" if args.generate else "",
    "epochs": args.epochs,
    "output_dir": args.output_dir,
    "wbits": args.wbits,
    "abits": args.abits,
    "lwc": "--lwc" if args.lwc else "",
    "nsamples": args.nsamples,
    "group_size": args.group_size,
    "save_dir": args.save_dir,
}

# Prepare ClearML Task
args = vars(args)
additional_packages = args["packages"]

packages = [
    #"git+https://github.com/neuralmagic/OmniQuant.git@shubhra/llama3.1",
    #"git+https://github.com/ChenMnZ/AutoGPTQ-bugfix.git@main"
]

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

# Create a symbolic link from python3 to python
if not os.path.exists("/usr/bin/python"):
    subprocess.run(["ln", "-s", "/usr/bin/python3.10", "/usr/bin/python"], check=True)

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-omniquant:latest")
task.set_packages(packages)
task.set_script(repository="https://github.com/neuralmagic/OmniQuant", branch="shubhra/llama3.1")
task.connect(omniquant_args, name="OmniQuant")

task.execute_remotely(args["queue_name"])

# Cloning and installing AutoGPTQ from source
#subprocess.run(["git", "clone", "https://github.com/neuralmagic/OmniQuant.git"], check=True)
#subprocess.run(["git", "checkout", "shubhra/llama3.1"], check=True)
#os.chdir("OmniQuant")
#subprocess.run(["pip", "install", "."], check=True)

# Cloning and installing AutoGPTQ from source
#subprocess.run(["git", "clone", "https://github.com/ChenMnZ/AutoGPTQ-bugfix.git"], check=True)
#os.chdir("AutoGPTQ-bugfix")
#subprocess.run(["pip", "install", "-v", "."], check=True)

#os.chdir("../")
#
# REMOTE
#

from clearml import InputModel

# If model is specified as a ClearML model, download it
if args["clearml_model"]:
    input_model = InputModel(model_id=omniquant_args["model"])
    omniquant_args["model"] = input_model.get_local_copy()
    task.connect(input_model)

# Construct the command for OmniQuant
command = [
    "python3", "main.py",
    f"--model", omniquant_args["model"],
    f"--epochs", str(omniquant_args["epochs"]),
    f"--output_dir", omniquant_args["output_dir"],
    f"--wbits", str(omniquant_args["wbits"]),   
    f"--abits", str(omniquant_args["abits"]),
    f"--nsamples", str(omniquant_args["nsamples"]),
    f"--save_dir", omniquant_args["save_dir"],
    f"--group_size", str(omniquant_args["group_size"]),
]

if omniquant_args["eval_ppl"]:
    command.append("--eval_ppl")

if omniquant_args["generate"]:
    command.append("--generate")

if omniquant_args["lwc"]:
    command.append("--lwc")

print("Running command:", " ".join(command))

# Execute the OmniQuant command
subprocess.run(command)

task.upload_artifact(name="OmniQuant output", artifact_object=omniquant_args["output_dir"])

