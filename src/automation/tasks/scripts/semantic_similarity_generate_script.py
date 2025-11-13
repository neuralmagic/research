import json
import os
import requests
from torch.cuda import device_count
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from automation.utils import kill_process_tree, parse_argument, flatten_nested_dict
from automation.datasets.tulu import make_tulu_prompt
from automation.datasets.openplatypus import make_openplatypus_prompt
from automation.datasets.alpaca import make_alpaca_prompt
from automation.datasets.defaults import make_default_prompt

try:
    from clearml import OutputModel, Task, Model
    clearml_available = True
except ImportError:
    clearml_available = False

RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def semantic_similarity_generate_main(
    model_id,
    trust_remote_code,
    dataset_args,
    semantic_similarity_args,
    max_model_len,
    max_new_tokens,
    clearml_model,
):
    from collections import defaultdict
    from huggingface_hub import snapshot_download

    all_conversations = []
    all_samples_dict = defaultdict(list)

    print(">>> Loading dataset...")
    for dataset_path, num_samples_per_dataset in dataset_args.items():
        dataset_name = dataset_path.split("/")[1].lower()
        print(f">>> Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_path, split=f"train[:{int(num_samples_per_dataset)}]")
        all_samples_dict[dataset_name].extend(dataset)

    sorted_all_samples_dict = dict(sorted(all_samples_dict.items()))

    for dataset_name,dataset_samples in sorted_all_samples_dict.items():
        print(f">>> Loading values for {dataset_name}...")
        for sample in dataset_samples:
            if dataset_name == "alpaca":
                prompt = make_alpaca_prompt(sample)
            elif dataset_name == "open-platypus":
                prompt = make_openplatypus_prompt(sample)
            elif dataset_name == "tulu-3-sft-mixture":
                prompt = make_tulu_prompt(sample)
            else:
                print("Using default prompt")
                prompt = make_default_prompt(sample)
            all_conversations.append(prompt)

    print("Define sampling parameters")
    sampling_params = SamplingParams(
        temperature=semantic_similarity_args.get("temperature", 0.0),
        max_tokens=max_new_tokens
    )

    HUGGINGFACE_DIR = "/home"
    if clearml_model:
        HUGGINGFACE_DIR = Model(model_id).get_local_copy()
    else:
        #snapshot_download(repo_id=model_id, local_dir=HUGGINGFACE_DIR)
    
    try:
        print(f"Initializing vLLM: {model_id}...")
        llm = LLM(
            model= model_id if "mistral" in model_id.lower() else HUGGINGFACE_DIR,
            dtype=semantic_similarity_args.get("dtype", "auto"),
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=device_count(),
            enforce_eager=semantic_similarity_args.get("enforce_eager", True),
            enable_chunked_prefill=semantic_similarity_args.get("enable_chunked_prefill", True),
            max_model_len=max_model_len,
            tokenizer_mode="mistral" if "mistral" in model_id.lower() else "auto"
        )
        print("Completed the model initialization ")
        print(">>> Running vLLM generation...")
        outputs = llm.chat(messages=all_conversations, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error initializing LLM: {e}")

    return all_conversations, outputs


def main(configurations=None, args=None):
    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)["Args"]
        clearml_model = parse_argument(args["clearml_model"], bool)
    else:
        args = args["Args"]
        clearml_model = False

    # Parse arguments
    force_download = parse_argument(args["force_download"], bool)
    trust_remote_code = parse_argument(args["trust_remote_code"], bool)
    model_id = parse_argument(args["model_id"], str)
    max_model_len = parse_argument(args["max_model_len"], int)
    max_new_tokens = parse_argument(args["max_new_tokens"], int)
    dataset_args = flatten_nested_dict(parse_argument(args["dataset_args"], dict))
    semantic_similarity_args= args.get("semantic_similarity_args", None)
    tags = args.get("tags", None)

    all_conversations, outputs = semantic_similarity_generate_main(
        model_id,
        trust_remote_code,
        dataset_args,
        semantic_similarity_args,
        max_model_len,
        max_new_tokens,
        clearml_model,
    )

    OUTPUT_FILE = os.path.join(RESULTS_DIR,f"{model_id.replace('/', '_')}.jsonl")
    print(">>> Writing outputs to file...")
    with open(OUTPUT_FILE, "w") as fout:
        for idx, (prompt, output) in enumerate(zip(all_conversations, outputs)):
            response = output.outputs[0].text.strip()
            fout.write(json.dumps({
                "index": idx,
                "prompt": prompt,
                "response": response
            }) + "\n")

    print(f">>> Completed. Saved {len(outputs)} outputs to {OUTPUT_FILE}")

    if clearml_available:
        task.upload_artifact("jsonl_output", OUTPUT_FILE)

if __name__ == '__main__':
    main()
