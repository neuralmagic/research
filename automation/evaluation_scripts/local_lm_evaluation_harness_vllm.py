from clearml import Task
import argparse

parser = argparse.ArgumentParser(description = "Eval model w/ lm-evaluation-harness using vllm backend")

parser.add_argument("--model-id", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--benchmark-tasks", type=str, default="openllm")
parser.add_argument("--num-fewshot", type=int, default=None)
parser.add_argument("--add-bos-token", action="store_true", default=False)
parser.add_argument("--max-gen-toks", type=int, default=256)
parser.add_argument("--batch-size", type=str, default="auto")
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
parser.add_argument("--cpu-offload-gb", type=int, default=None)
parser.add_argument("--enable-chunked-prefill", action="store_true", default=False)
parser.add_argument("--max-model-len", type=int, default=4096)
parser.add_argument("--max-num-batched-tokens", type=int, default=512)
parser.add_argument("--max-num-seqs", type=int, default=128)
parser.add_argument("--apply-chat-template", action="store_true", default=False)
parser.add_argument("--fewshot-as-multiturn", action="store_true", default=False)
parser.add_argument("--num-gpus", type=int, default=None)

args = parser.parse_args()

Task.force_store_standalone_script()

#task = Task.init(project_name=args.project_name, task_name=args.task_name)

from clearml import InputModel
from glob import glob
import subprocess
import os
import torch
import importlib.util
import sys

current_file_path = os.path.abspath(__file__)
path_to_parsing = os.path.join(current_file_path, "..", "..", "clearml_evaluation_parsing", "lm_evaluation_harness.py")
sys.path.insert(0, path_to_parsing)

from lm_evaluation_harness import push_to_clearml

if args."num_gpus" is None:
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = args."num_gpus"

if args."clearml_model":
    input_model = InputModel(model_id=args."model_id")
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args."model_id"

model_args = f"pretrained={model_id},dtype=auto,max_model_len={args.max_model_len},max_gen_toks={args.max_gen_toks},gpu_memory_utilization={args.gpu_memory_utilization},tensor_parallel_size={num_gpus},max_num_seqs={args.max_num_seqs}"
if args.add_bos_token:
    model_args += ",add_bos_token=True"
if args.trust_remote_code:
    model_args += ",trust_remote_code=True"
if args.enable_chunked_prefill:
    model_args += f",enable_chunked_prefill=True,disable_sliding_window=True,max_num_batched_tokens={args.max_num_batched_tokens}"
if args.cpu_offload_gb is not None:
    model_args += f",cpu_offload_gb={args.cpu_offload_gb}"

inputs = [
    "lm_eval", 
    "--model", "vllm", 
    "--tasks", args.benchmark_tasks, 
    "--model_args", model_args,
    "--write_out", 
    "--show_config", 
    "--output_path", ".",
    "--batch_size", args.batch_size,
]

if args.num_fewshot is not None:
    inputs.extend(["--num_fewshot", str(args.num_fewshot)])

if args.apply_chat_template:
    inputs.append("--apply_chat_template")

if args.fewshot_as_multiturn:
    inputs.append("--fewshot_as_multiturn")

subprocess.run(inputs)

model_suffix = os.path.split(model_id)[-1]
results_dir = glob(f"*{model_suffix}")[-1]
json_file = glob(os.path.join(results_dir, "results*.json"))[0]
if args.clearml_model:
    push_to_clearml(task, json_file, input_model)
else:
    push_to_clearml(task, json_file)