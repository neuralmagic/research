from automation.tasks import ArenaHardJudgeTask

task = ArenaHardJudgeTask(
    project_name="gpt_arena_debug",
    task_name="test_judge_task_1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1", "vllm==0.10.1.1"],
    answer_task_name ="generate_math_task_gpt",
    judgement_model ="Qwen/Qwen2-7B-Instruct",
    question_size = "small",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    #bench_name = "arena-hard-v2.0",
    bench_name = "arena-hard-v0.1",
    #data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1024, 
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()
