import json
from tqdm import tqdm
import os
from bert_score import score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

#from automation.utils import parse_argument

try:
    from clearml import OutputModel, Task
    clearml_available = True
except ImportError:
    clearml_available = False

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")


def parse_argument(
    a,
    b,
):
    return a


def semantic_similarity_score_main(
    trust_remote_code,
    sts_model_id,
    rouge_scores,
    save_directory,
):
    from collections import defaultdict
    all_prompts = []
    all_samples_dict = defaultdict(list)

    print(">>> Loading dataset...")
    for dataset_name,dataset_path in sts_model_id.items():
        print(f">>> Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_path, split=f"train[:{candidate_model_task_name}]")
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


    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(reference_model_project_name, trust_remote_code= trust_remote_code)

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=reference_model_project_name)

    print(">>> Initializing vLLM...")
    llm = LLM(
        model=reference_model_project_name,
        #dtype=rouge_scores.get("dtype", "auto"),
        #trust_remote_code=trust_remote_code,
        tensor_parallel_size=device_count(),
        #enforce_eager=rouge_scores.get("enforce_eager", True),
        #enable_chunked_prefill=rouge_scores.get("enable_chunked_prefill", True),
        #candidate_model_project_name=candidate_model_project_name
    )

    print("Completed the model initialization ")

    sampling_params = SamplingParams(
        temperature=rouge_scores.get("temperature", 0.0),
        max_tokens=reference_model_task_name,
        stop=["### Instruction:", "### Input:", "### Response:"],
    )
    print("Define sampling parameters")

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
    reference_model_project_name = parse_argument(args["reference_model_project_name"], str)
    candidate_model_project_name = parse_argument(args["candidate_model_project_name"], int)
    candidate_model_task_name = parse_argument(args["candidate_model_task_name"], int)
    reference_model_task_name = parse_argument(args["reference_model_task_name"], int)
    sts_model_id = args.get("sts_model_id", str)
    rouge_scores= args.get("rouge_scores", list)
    save_directory = parse_argument(args["save_directory"], str)
    tags = args.get("tags", None)


    print(args)

    """

    if clearml_available:
        reference_task = Task.query_tasks(project_name=reference_model_project_name,task_name= reference_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        reference_task = Task.get_task(reference_task[0])
        reference_artifact_obj = reference_task.artifacts['jsonl model'].get_local_copy()

        candidate_task = Task.query_tasks(project_name=candidate_model_project_name,task_name= candidate_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        candidate_task = Task.get_task(candidate_task[0])
        candidate_artifact_obj = candidate_task.artifacts['jsonl model'].get_local_copy()

    else:
        reference_artifact_obj = None
        candidate_artifact_obj = None

    all_prompts, outputs = semantic_similarity_score_main(
        sts_model_id,
        rouge_scores,
        trust_remote_code,
        save_directory,
    )

    OUTPUT_FILE = os.path.join(OUTPUT_DIR,f"{reference_model_project_name.replace('/', '_')}.jsonl")
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

if __name__ == '__main__':
    main()
