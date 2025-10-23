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
os.makedirs(SCORING_DIR, exist_ok=False)

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
    #all_bert_f1 = [ f1.item() for f1 in f1_scores ]
    
    # Evaluate metrics
    all_rouge1_f1, all_rougeL_f1, all_sts, all_bert_f1 = [], [], [], []
    low_score_indices = []

    for i, (ref, cand, f1) in enumerate(zip(references, candidates, f1_scores)):
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

        all_bert_f1.append(f1.item())

        if f1 < 0.85 or rouge1 < 0.5 or sts < 0.85:
            low_score_indices.append(i)

    # Compute averages
    num_samples = len(references)
    avg_bert = sum(all_bert_f1) / num_samples
    avg_rouge1 = sum(all_rouge1_f1) / num_samples
    avg_rougeL = sum(all_rougeL_f1) / num_samples
    avg_sts = sum(all_sts) / num_samples
    return avg_bert, avg_rouge1, avg_rougeL, avg_sts, low_score_indices

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
    tags = args.get("tags", None)

    print(args)
    if clearml_available:
        reference_model_project_name = parse_argument(args["reference_model_project_name"], str)
        candidate_model_project_name = parse_argument(args["candidate_model_project_name"], str)
        candidate_model_task_name = parse_argument(args["candidate_model_task_name"], str)
        reference_model_task_name = parse_argument(args["reference_model_task_name"], str)
        reference_task = Task.query_tasks(project_name=reference_model_project_name,task_name= reference_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        reference_task = Task.get_task(reference_task[0])
        reference_file = reference_task.artifacts['jsonl_output'].get_local_copy()

        candidate_task = Task.query_tasks(project_name=candidate_model_project_name,task_name= candidate_model_task_name, task_filter={'order_by': ['-last_update'], 'status': ['completed'] })
        candidate_task = Task.get_task(candidate_task[0])
        candidate_file = candidate_task.artifacts['jsonl_output'].get_local_copy()
    else:
        ref_model_jsonl = args.get("ref_model_jsonl", str)
        cand_model_jsonl = args.get("cand_model_jsonl", str)
        reference_file = os.path.join(SCORING_DIR, ref_model_jsonl)
        candidate_file = os.path.join(SCORING_DIR, cand_model_jsonl)
    
    avg_bert, avg_rouge1, avg_rougeL, avg_sts, low_score_indices = semantic_similarity_score_main(
        reference_file,
        candidate_file,
        sts_model_id,
        rouge_scores,
    )
    # Print summary
    print("\n=== Averages (for Google Sheets) ===")
    print("BERTScore F1 | ROUGE-1 F1 | ROUGE-L F1 | STS CosSim")
    print(f"{avg_bert:.3f} | {avg_rouge1:.3f} | {avg_rougeL:.3f} | {avg_sts:.3f}")

    print("\n=== Low-score indices (BERT < 0.85, ROUGE-1 < 0.5, STS < 0.85) ===")
    print(low_score_indices)

    data = {
        "BERTScore F1": f"{avg_bert:.3f}",
        "ROUGE-1 F1": f"{avg_rouge1:.3f}",
        "ROUGE-1 FL": f"{avg_rougeL:.3f}",
        "STS CosSim": f"{avg_sts:.3f}",
    }

    from pathlib import Path

    reference_file = Path(reference_file).stem.lower()
    candidate_file = Path(candidate_file).stem.lower()
    out_filename = f"scores_{reference_file}__vs__{candidate_file}.txt"
    out_filename = os.path.join(SCORING_DIR,out_filename)
    
    # Save results
    with open(out_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"\nSaved results to {out_filename}")
    if clearml_available:
        task.upload_artifact("scores", data)
        task.upload_artifact("outscores", out_filename)
        print("Pushing clearml artifact")

if __name__ == '__main__':
    main()
