from automation.tasks import SemanticSimilarityGenerateTask

task = SemanticSimilarityGenerateTask(
    project_name="semantic_similarity_debug",
    task_name="semantic_generation_qwen3_14b_base",
    #task_name="semantic_generation_qwen3_14b_w4a16",
    branch="semantic_similarity",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1", "vllm==0.10.1.1"],
    dataset_args = {"alpaca": "tatsu-lab/alpaca", "openplatypus": "garage-bAInd/Open-Platypus", "tulu": "allenai/tulu-3-sft-mixture"},
    model_id="Qwen/Qwen3-14B",
    #model_id="RedHatAI/Qwen3-14B-quantized.w4a16",
    num_samples_per_dataset=330,
    #num_samples_per_dataset=10,
    max_new_tokens=1024,
    max_model_len=4096,
    semantic_similarity_args={"enable-chunked-prefill": True, "enforce_eager": True, "dtype" :"auto", "device_map": "auto", "temperature": 0.0},
)

task.execute_remotely("oneshot-a100x1")
