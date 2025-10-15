import json
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from automation.utils import resolve_reference_model_id, parse_argument, load_callable_configuration

try:
    from clearml import OutputModel, Task
    clearml_available = True
except ImportError:
    clearml_available = False


OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

def parse_argument(a,b):
    return a


def make_alpaca_platypus_prompt(sample):
    print("Using Alpaca / Platypus style prompt")
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text if input_text else 'N/A'}\n\n"
        f"### Response:\n"
    )

    return prompt


def make_tulu_prompt(sample):
    print("Using Tulu / OASST style prompt")
    msgs = []
    for m in sample["messages"]:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        msgs.append(f"{role.upper()}: {content}")
    joined = "\n".join(msgs)
    prompt = f"### Conversation:\n{joined}\n\n### Response:\n"

    return prompt


def make_default_prompt(sample):
    print("Using default prompt")
    prompt = f"### Input:\n{json.dumps(sample)}\n\n### Response:\n"
    return prompt


def semantic_similarity_generate_main(
    reference_model_id,
    trust_remote_code,
    dataset_args,
    max_model_len,
    max_new_tokens,
    num_samples,
    save_directory,
):
    dtype = "auto"
    device_map = "auto"

    from collections import defaultdict
    all_prompts = []
    all_samples_dict = defaultdict(list)

    print(">>> Loading dataset...")
    for dataset_name,dataset_path in dataset_args.items():
        print(f">>> Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_path, split=f"train[:{num_samples}]")
        all_samples_dict[dataset_name].extend(dataset)

    for dataset_name,dataset_samples in all_samples_dict.items():
        print(f">>> Loading values for {dataset_name}...")
        for sample in dataset_samples:
            if dataset_name == "alpaca" or (dataset_name == "openplatypus"):
                prompt = make_alpaca_platypus_prompt(sample)
            elif dataset_name == "tulu":
                prompt = make_tulu_prompt(sample)
            else:
                print("Using default prompt")
                prompt = make_default_prompt(sample)
            all_prompts.append(prompt)

    TEMPERATURE = 0.0

    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(reference_model_id, trust_remote_code=True)

    print(">>> Initializing vLLM...")
    llm = LLM(
        model=reference_model_id,
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=8,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_new_tokens,
        stop=["### Instruction:", "### Input:", "### Response:"],
    )

    print(">>> Running vLLM generation...")
    outputs = llm.generate(all_prompts, sampling_params)

    return all_prompts, outputs

def main(configurations=None, args=None):
    if clearml_available:
        task = Task.current_task()
        args = task.get_parameters_as_dict(cast=True)["Args"]
    else:
        args = args["Args"]

    # Parse arguments
    clearml_model = parse_argument(args["clearml_model"], bool)
    force_download = parse_argument(args["force_download"], bool)
    trust_remote_code = parse_argument(args["trust_remote_code"], bool)
    reference_model_id = parse_argument(args["reference_model_id"], str)
    candidate_model_id= parse_argument(args["candidate_model_id"], str)
    dataset_name = parse_argument(args["dataset_name"], str)
    save_directory = parse_argument(args["save_directory"], str)
    max_model_len = parse_argument(args["max_model_len"], int)
    num_samples = parse_argument(args["num_samples"], int)
    max_new_tokens = parse_argument(args["max_new_tokens"], int)
    dataset_args = args.get("dataset_args", None)
    tags = args.get("tags", None)

    all_prompts, outputs = semantic_similarity_generate_main(
        reference_model_id,
        trust_remote_code,
        dataset_args,
        max_model_len,
        max_new_tokens,
        num_samples,
        save_directory,
    )

    OUTPUT_FILE = os.path.join(OUTPUT_DIR,f"{reference_model_id.replace('/', '_')}.jsonl")
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

        """
        # Upload model to ClearML
        clearml_model_object = OutputModel(
            task=task, 
            name=task.name,
            framework="PyTorch", 
           tags=[tags] if isinstance(tags, str) else tags or []
        )
        clearml_model_object.update_weights(weights_filename=save_directory, auto_delete_file=False)
        """

if __name__ == '__main__':
    main()
