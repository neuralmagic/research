import json
import os
import requests
from torch.cuda import device_count
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from automation.utils import kill_process_tree, parse_argument
from automation.datasets.utils import make_alpaca_platypus_prompt, make_tulu_prompt, make_default_prompt

try:
    from clearml import OutputModel, Task, Model
    clearml_available = True
except ImportError:
    clearml_available = False

RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=False)

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

    all_prompts = []
    all_samples_dict = defaultdict(list)

    print(">>> Loading dataset...")
    for dataset_path, num_samples_per_dataset in dataset_args.items():
        print(f"The dataset args: {dataset_args}")
        print(f"The dataset path is: {dataset_path}")
        dataset_name = dataset_path.split("/")[1].lower()
        print(f">>> Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_path, split=f"train[:{int(num_samples_per_dataset)}]")
        all_samples_dict[dataset_name].extend(dataset)

    for dataset_name,dataset_samples in all_samples_dict.items():
        print(f">>> Loading values for {dataset_name}...")
        for sample in dataset_samples:
            if dataset_name == "alpaca" or (dataset_name == "open-platypus"):
                prompt = make_alpaca_platypus_prompt(sample)
            elif dataset_name == "tulu-3-sft-mixture":
                prompt = make_tulu_prompt(sample)
            else:
                print("Using default prompt")
                prompt = make_default_prompt(sample)
            all_prompts.append(prompt)

    print("Define sampling parameters")
    sampling_params = SamplingParams(
        temperature=semantic_similarity_args.get("temperature", 0.0),
        max_tokens=max_new_tokens,
        stop=["### Instruction:", "### Input:", "### Response:"],
    )

    HUGGINGFACE_DIR = "/home"
    if clearml_model:
        HUGGINGFACE_DIR = Model(model_id).get_local_copy()
    else:
        print(">>> Downloading snapshot ...")
        snapshot_download(repo_id=model_id, local_dir=HUGGINGFACE_DIR)
    
    try:
        print(">>> Initializing vLLM...")
        llm = LLM(
            model=HUGGINGFACE_DIR,
            dtype=semantic_similarity_args.get("dtype", "auto"),
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=device_count(),
            enforce_eager=semantic_similarity_args.get("enforce_eager", True),
            enable_chunked_prefill=semantic_similarity_args.get("enable_chunked_prefill", True),
            max_model_len=max_model_len
        )
        print("Completed the model initialization ")
        print(">>> Running vLLM generation...")
        outputs = llm.generate(all_prompts, sampling_params)
    except Exception as e:
        print(f"Error initializing LLM: {e}")

    return all_prompts, outputs

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
    dataset_args = args.get("dataset_args", None)
    semantic_similarity_args= args.get("semantic_similarity_args", None)
    tags = args.get("tags", None)

    print(f"Input dataset_args: {dataset_args}")
    dataset_args = {"tatsu-lab/alpaca" : 300 , "garage-bAInd/Open-Platypus": "310", "allenai/tulu-3-sft-mixture": 320}

    print(f"Hardcode dataset_args: {dataset_args}")

    all_prompts, outputs = semantic_similarity_generate_main(
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
        for idx, (prompt, output) in enumerate(zip(all_prompts, outputs)):
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
