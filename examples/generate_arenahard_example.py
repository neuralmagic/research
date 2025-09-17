from automation.tasks import ArenaHardGenerateTask

task = ArenaHardGenerateTask(
    project_name="gpt_arena_debug",
    task_name="generate_math_task_gpt",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1", "vllm==0.10.1.1"],
    generate_model= "RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8",
    #generate_model= "openai/gpt-oss-120b",
    question_size = "small",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    bench_name = "arena-hard-v2.0",
    #bench_name = "arena-hard-v0.1",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1024, 
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()
