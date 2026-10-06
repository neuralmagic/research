from automation.tasks import SemanticSimilarityScoreTask

task = SemanticSimilarityScoreTask(
    project_name="semantic_similarity_debug",
    #task_name="semantic_scoring_14b",
    task_name="semantic_scoring_4b",
    branch="semantic_similarity",
    packages = ["huggingface-hub==0.34.3", "networkx==3.4.2", "datasets==4.2.0", "rouge_score==0.1.2", "bert-score==0.3.13", "sentence-transformers==5.1.1", "matplotlib"],
    reference_model_project_name="semantic_similarity_debug",
    candidate_model_project_name="semantic_similarity_debug",
    reference_model_task_name="semantic_generation_qwen3_14b_feedback",
    #reference_model_task_name="semantic_generation_qwen3_14b_base",
    candidate_model_task_name="semantic_generation_qwen3_14b_w4a16_feedback",
    #candidate_model_task_name="semantic_generation_qwen3_14b_w4a16",
    sts_model_id="all-MiniLM-L6-v2",
    rouge_scores=["rouge1", "rougeL"],
    low_score_threshold_args={"f1": 0.79, "rouge1": 0.65, "sts": 0.71},
)

task.execute_remotely("oneshot-a100x1")
