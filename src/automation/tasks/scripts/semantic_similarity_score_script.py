import json
from bert_score import score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import os
from automation.utils import parse_argument

try:
    from clearml import OutputModel, Task
    clearml_available = True
except ImportError:
    clearml_available = False

SCORING_DIR = os.path.join(os.getcwd(), "scoresdirectory")
os.makedirs(SCORING_DIR, exist_ok=True)

def semantic_similarity_score_main(
    reference_file,
    candidate_file,
    sts_model_id,
    rouge_scores,
):
    # Load reference and candidate data
    with open(reference_file, "r") as f_ref, open(candidate_file, "r") as f_cand:
        reference_data = [json.loads(line) for line in f_ref]
        candidate_data = [json.loads(line) for line in f_cand]
    
    assert len(reference_data) == len(candidate_data), "Mismatched number of entries!"

    # Extract answers
    references = [ref.get("output") or ref["response"] for ref in reference_data]
    candidates = [cand["response"] for cand in candidate_data]
    
    # Load models
    sts_model = SentenceTransformer(sts_model_id)
    rouge = rouge_scorer.RougeScorer(rouge_scores, use_stemmer=True)
    
    # Compute BERTScore
    _, _, f1_scores = score(candidates, references, lang="en", verbose=False)
    all_bert_f1 = [ f1.item() for f1 in f1_scores ]
    
    # Evaluate metrics
    all_rouge1_f1, all_rougeL_f1, all_sts, all_bert_f1 = [], [], [], []
    low_score_indices = []

    for i, (ref, cand) in enumerate(zip(references, candidates)):
        emb_ref = sts_model.encode(ref, convert_to_tensor=True)
        emb_cand = sts_model.encode(cand, convert_to_tensor=True)
        raw_sts = util.cos_sim(emb_cand, emb_ref).item()
        sts = (raw_sts + 1) / 2  # Normalize to [0, 1]
        all_sts.append(sts)
    
        rouge_scores = rouge.score(ref, cand)

        rouge1 = rouge_scores["rouge1"].fmeasure
        rougeL = rouge_scores["rougeL"].fmeasure
        all_rouge1_f1.append(rouge1)
        all_rougeL_f1.append(rougeL)
    

    # Compute averages
    n = len(references)
    avg_bert = sum(all_bert_f1) / n
    avg_rouge1 = sum(all_rouge1_f1) / n
    avg_rougeL = sum(all_rougeL_f1) / n
    avg_sts = sum(all_sts) / n
    return avg_bert, avg_rouge1, avg_rougeL, avg_sts




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
    sts_model_id = args.get("sts_model_id", str)
    rouge_scores= args.get("rouge_scores", list)
    #save_directory = parse_argument(args["save_directory"], str)
    tags = args.get("tags", None)

    print(args)
    if clearml_available:
        reference_model_project_name = parse_argument(args["reference_model_project_name"], str)
        candidate_model_project_name = parse_argument(args["candidate_model_project_name"], str)
        candidate_model_task_name = parse_argument(args["candidate_model_task_name"], str)
        reference_model_task_name = parse_argument(args["reference_model_task_name"], str)
        reference_task = Task.query_tasks(project_name=reference_model_project_name,task_name= reference_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        reference_task = Task.get_task(reference_task[0])
        reference_file = reference_task.artifacts['reference_jsonl_output'].get_local_copy()

        candidate_task = Task.query_tasks(project_name=candidate_model_project_name,task_name= candidate_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        candidate_task = Task.get_task(candidate_task[0])
        candidate_file = candidate_task.artifacts['candidate_jsonl_output'].get_local_copy()
        # add task query to get jsonl
    else:
        ref_model_json = "Qwen_Qwen3-0.6B.jsonl"
        cand_model_json = "RedHatAI_Qwen3-0.6B-quantized.w4a16.jsonl"
        reference_file = os.path.join(SCORING_DIR, ref_model_json)
        candidate_file = os.path.join(SCORING_DIR, cand_model_json)
    
    avg_bert, avg_rouge1, avg_rougeL, avg_sts = semantic_similarity_score_main(
        reference_file,
        candidate_file,
        sts_model_id,
        rouge_scores,
    )
    # Print summary
    print("\n=== Averages ===")
    print("BERTScore F1 | ROUGE-1 F1 | ROUGE-L F1 | STS CosSim")
    print(f"{avg_bert:.3f} | {avg_rouge1:.3f} | {avg_rougeL:.3f} | {avg_sts:.3f}")

    data = {
        "BERTScore F1": f"{avg_bert:.3f}",
        "ROUGE-1 F1": f"{avg_rouge1:.3f}",
        "ROUGE-1 FL": f"{avg_rougeL:.3f}",
        "STS CosSim": f"{avg_sts:.3f}",
    }


    if clearml_available:
        task.upload_artifact("scores", data)
        print("Pushing clearml artifact")
    else: 
        out_filename = f"scores_{reference_file.lower()}__vs__{candidate_file.lower()}.txt"
        out_filename = os.path.join(SCORING_DIR,out_filename)
        
        print(f"\nSaved results to {out_filename}")
        with open(out_filename, "w") as file:
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()
