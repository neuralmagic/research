from automation.tasks import ArenaHardGenerateTask

task = ArenaHardGenerateTask(
    project_name="simple_debug",
    task_name="generate_math_task_4",
    #generate_model="meta-llama/Llama-3.2-1B-Instruct",
    #generate_model="Qwen/Qwen2.5-1.5B-Instruct",
    generate_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    max_seconds=30,
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    #vllm_kwargs={"enable-chunked-prefill": True}

    #generation_config_file='gen_answer_config.yaml',
    generation_config_file='math_answer_config.yaml',
    #generation_endpoint_file='api_config.yaml',
    generation_endpoint_file='math_api_config.yaml',
)

task.execute_remotely("oneshot-a100x4")
#task.execute_locally()
