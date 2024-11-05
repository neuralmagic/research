from clearml import Task
import argparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Apply recipe in one-shot")

parser.add_argument("--model-id", type=str)
parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--recipe", type=str)
parser.add_argument("--clearml-model", action="store_true", default=False)
parser.add_argument("--disable-clearml-model-save", action="store_true", default=False)
parser.add_argument("--save-dir", type=str, default="output")
parser.add_argument("--dataset", type=str, default="neuralmagic/LLM_compression_calibration")
parser.add_argument("--random-fraction", type=float, default=0.)
parser.add_argument("--disable-shuffle", action="store_true", default=False)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--tags", type=str, nargs="+", default=None)
parser.add_argument("--packages", type=str, nargs="+", default=None)
parser.add_argument("--max-memory-per-gpu", type=str, default=None)
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--save-uncompressed", action="store_true", default=False)


args = parser.parse_args()

args = vars(args)
project_name = args.pop("project_name")
task_name = args.pop("task_name")
queue_name = args.pop("queue_name")
additional_packages = args.pop("packages")

packages = [
    "git+https://github.com/vllm-project/llm-compressor.git@main",
    "git+https://github.com/neuralmagic/compressed-tensors.git@main",
    "sentencepiece",
]

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-clean:latest")
task.set_packages(packages)

task.execute_remotely(queue_name)

#
# REMOTE
#

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, interleave_datasets
import math
import random
from clearml import InputModel, OutputModel
import torch

# Load model
if args["clearml_model"]:
    input_model = InputModel(model_id=args["model_id"])
    model_id = input_model.get_local_copy()
    task.connect(input_model)
else:
    model_id = args["model_id"]

if args["dtype"] == "auto":
    dtype = "auto"
else:
    dtype = getattr(torch, args["dtype"])

if args["max_memory_per_gpu"] is None:
    device_map = "auto"
else:
    if "single" in queue_name or "x1" in queue_name:
        num_gpus = 1
    elif "double" in queue_name or "x2" in queue_name:
        num_gpus = 2
    elif "quad" in queue_name or "x4" in queue_name:
        num_gpus = 4
    elif "octo" in queue_name or "x8" in queue_name:
        num_gpus = 8
    
    if args["max_memory_per_gpu"] == "hessian":
        device_map = calculate_offload_device_map(
            model_id, 
            reserve_for_hessians=True, 
            num_gpus=num_gpus, 
            torch_dtype=dtype,
            trust_remote_code=args["trust_remote_code"],
        )
    else:
        device_map = custom_offload_device_map(
            model_id, 
            max_memory_per_gpu=args["max_memory_per_gpu"] + "GB",
            num_gpus=num_gpus, 
            torch_dtype=dtype,
            trust_remote_code=args["trust_remote_code"],
        )

    print(device_map)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    device_map=device_map, 
    trust_remote_code=args["trust_remote_code"],
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args["trust_remote_code"])

# Build dataset
def get_c4_samples(num_samples, max_seq_len, tokenizer):
    dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")

    tokenized_text = []
    for sample in dataset:
        tokenized_text.extend(tokenizer(sample["text"])["input_ids"])
        if len(tokenized_text) > num_samples * max_seq_len:
            break

    input_ids = [tokenized_text[max_seq_len*i:max_seq_len*(i+1)] for i in range(num_samples)]
    attention_mask = num_samples * [max_seq_len * [1]]

    return Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})


def get_downstream_samples(dataset_name, num_samples, max_seq_len, tokenizer, shuffle):
    def preprocess_ultrachat(example):
        if example["messages"][0]["role"]!="system":
            example["messages"].insert(0, {"role": "system", "content": ""})
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    def preprocess_platypus(example):
        messages = [
            {
                "role": "system", 
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            },
            {
                "role": "user", 
                "content": example["instruction"],
            },
            {
                "role": "assistant",
                "content": example["output"],
            },
        ]
        print(tokenizer.chat_template)
        if tokenizer.chat_template is None:
            return {"text": messages[0]["content"] + "\n\n### Instruction:\n" + messages[1]["content"] + "\n\n### Response:\n" + messages[2]["content"]}
        else:
            try:
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            except:
                messages = messages[1:]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            
    def preprocess_calibration(example):
        if tokenizer.chat_template is None:
            return {"text": example["text"]}
        else:
            try:
                return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
            except:
                messages = example["messages"][1:]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    preprocess_fn = {
        "HuggingFaceH4/ultrachat_200k": preprocess_ultrachat,
        "gsm8k": lambda example: {"text": "Question: {question}\nAnswer:\n{answer}".format_map(example)},
        "theblackcat102/evol-codealpaca-v1": lambda example: {"text": """[Instructions]:\n{instruction}\n\n[Response]:{output}""".format_map(example)},
        "HuggingFaceTB/cosmopedia-100k": lambda example: {"text": example["prompt"]},
        "garage-bAInd/Open-Platypus": preprocess_platypus,
        "neuralmagic/LLM_compression_calibration": preprocess_calibration,
    }

    if dataset_name == "HuggingFaceH4/ultrachat_200k":
        ds = load_dataset(dataset_name, split="train_sft[:5%]")
    elif dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
    elif "json" in dataset_name:
        print("Loading json dataset: ", dataset_name)
        ds = load_dataset("json", data_files=dataset_name)["train"]
    else:
        print("Loading dataset from HF: ", dataset_name)
        ds = load_dataset(dataset_name, split="train")

    if shuffle:
        ds = ds.shuffle()

    ds = ds.select(range(num_samples))
    if dataset_name in preprocess_fn:
        ds = ds.map(preprocess_fn[dataset_name], remove_columns=ds.column_names)

    if "text" in ds.features:
        ds = ds.map(lambda example: tokenizer(example["text"], padding=False, max_length=max_seq_len, truncation=True), remove_columns=ds.column_names)
    
    if "token_type_ids" in ds.features:
        ds = ds.remove_columns(["token_type_ids"])
    
    return ds

def get_dataset_samples(dataset_name, num_samples, max_seq_len, tokenizer, shuffle):
    if dataset_name == "c4":
        return get_c4_samples(num_samples, max_seq_len, tokenizer)
    else:
        return get_downstream_samples(dataset_name, num_samples, max_seq_len, tokenizer, shuffle)

def get_dataset(dataset_name, num_samples, max_seq_len, tokenizer, random_fraction, shuffle):
    num_dataset_samples = math.ceil(num_samples * (1. - random_fraction))
    if num_dataset_samples > 0:
        downstream_dataset = get_dataset_samples(dataset_name, num_dataset_samples, max_seq_len, tokenizer, shuffle)
    else:
        downstream_dataset = None

    if random_fraction > 0.:
        max_token_id = len(tokenizer.get_vocab()) - 1
        num_random_samples = num_samples - num_dataset_samples

        input_ids = [[random.randint(0, max_token_id) for _ in range(max_seq_len)] for _ in range(num_random_samples)]
        attention_mask = num_random_samples * [max_seq_len * [1]]

        random_dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})
        
        if downstream_dataset is not None:
            dataset = interleave_datasets([downstream_dataset, random_dataset])
        else:
            dataset = random_dataset
    else:
        dataset = downstream_dataset
        

    return dataset

dataset = get_dataset(
    args["dataset"], 
    args["num_samples"],
    args["max_seq_len"],
    tokenizer, 
    args["random_fraction"],
    not args["disable_shuffle"],
)

if "yaml" in args["recipe"]:
    recipe = open(args["recipe"]).read()
else:
    recipe = args["recipe"]

task.upload_artifact(name="recipe", artifact_object=recipe)

# apply recipe to the model
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=args["max_seq_len"],
    num_calibration_samples=args["num_samples"],
)

# save model compressed
model.save_pretrained(args["save_dir"], save_compressed=(not args["save_uncompressed"]))

# upload model to ClearML
if not args["disable_clearml_model_save"]:
    clearml_model = OutputModel(
        task=task, 
        name=task.name,
        framework="PyTorch", 
        tags=args["tags"],
    )

    clearml_model.update_weights(weights_filename=args["save_dir"], auto_delete_file=False)
