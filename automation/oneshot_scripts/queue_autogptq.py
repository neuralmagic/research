from clearml import Task
import argparse

#
# LOCAL
#

parser = argparse.ArgumentParser(description = "Apply auto-gptq")

parser.add_argument("--model-id", type=str, nargs="+")
parser.add_argument("--queue-name", type=str)
parser.add_argument("--project-name", type=str)
parser.add_argument("--task-name", type=str)
parser.add_argument("--bits", type=int)
parser.add_argument("--group-size", type=int)
parser.add_argument("--desc-act", type=bool, default=False)
parser.add_argument("--damp-percent", type=float, default=0.01)
parser.add_argument("--disable-clearml-model-save", action="store_true", default=False)
parser.add_argument("--save-dir", type=str, default="output")
parser.add_argument("--dataset", type=str, default="neuralmagic/LLM_compression_calibration")
parser.add_argument("--disable-shuffle", action="store_true", default=False)
parser.add_argument("--random-fraction", type=float, default=0.)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--tags", type=str, nargs="+", default=None)
parser.add_argument("--packages", type=str, nargs="+", default=None)

args = parser.parse_args()

args = vars(args)
project_name = args.pop("project_name")
task_name = args.pop("task_name")
queue_name = args.pop("queue_name")
additional_packages = args.pop("packages")

packages = [
    "auto-gptq",
    "sentencepiece",
    "transformers"
]

if additional_packages is not None and len(additional_packages) > 0:
    packages.extend(additional_packages)

Task.force_store_standalone_script()

task = Task.init(project_name=project_name, task_name=task_name)
task.set_base_docker(docker_image="498127099666.dkr.ecr.us-east-1.amazonaws.com/mlops/k8s-research-torch:latest")
task.set_packages(packages)

task.execute_remotely(queue_name)

#
# REMOTE
#

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset, Dataset, interleave_datasets
import math
import random
from clearml import InputModel, OutputModel
import torch

# Load model
if len(args["model_id"]) == 1:
    model_id = args["model_id"][0]
else:
    # retrieve model metadata from clearml
    model_project_name = args["model_id"][0]
    model_id = args["model_id"][1]
    input_model = InputModel(project=model_project_name, name=model_id)
    model_id = input_model.get_local_copy()
    task.connect(input_model)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


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
        de = load_dataset("gsm8k", "main", split="train")
    elif "json" in dataset_name:
        print("Loading json dataset: ", dataset_name)
        ds = load_dataset("json", data_files=dataset_name)["train"]
    else:
        print("Loading dataset from HF: ", dataset_name)
        ds = load_dataset(dataset_name, split="train")

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
        return get_c4_samples(num_samples, max_seq_len, tokenizer, shuffle)
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


quantize_config = BaseQuantizeConfig(
  bits=args["bits"],
  group_size=args["group_size"],
  desc_act=args["desc_act"],
  model_file_base_name="model",
  damp_percent=args["damp_percent"],
)

# apply recipe to the model
model = AutoGPTQForCausalLM.from_pretrained(
  model_id,
  quantize_config,
  trust_remote_code=args["trust_remote_code"],
).to("cuda")

examples = []
for sample in dataset:
    examples.append({"input_ids": torch.tensor(sample["input_ids"]).to("cuda"), "attention_mask": torch.tensor(sample["attention_mask"]).to("cuda")})


model.quantize(examples)

# save model compressed
model.save_pretrained(args["save_dir"])
tokenizer.save_pretrained(args["save_dir"])
        
# upload model to ClearML
if not args["disable_clearml_model_save"]:
    clearml_model = OutputModel(
        task=task, 
        name=task.name,
        framework="PyTorch", 
        tags=args["tags"],
    )

    clearml_model.update_weights(weights_filename=args["save_dir"], auto_delete_file=False)
