from automation.tasks import ArenaHardJudgeTask

task = ArenaHardJudgeTask(
    project_name="simple_debug",
    task_name="test_judge_task_1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1","vllm==0.10.0"],
    #answer_task_name  = "generate_math_task_1",
    #answer_project_name = "ok",
    judgement_model="meta-llama/Llama-3.2-1B-Instruct",
    #judgement_model="Qwen/Qwen2.5-1.5B-Instruct",
    #judgement_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    #max_seconds=30,
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1300, 
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()
