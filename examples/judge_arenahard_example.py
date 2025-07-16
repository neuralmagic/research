from automation.tasks import ArenaHardJudgeTask

task = ArenaHardJudgeTask(
    project_name="alexandre_debug",
    task_name="test_judge_task",
    #model="meta-llama/Llama-3.2-1B-Instruct",
    generate_model="Qwen/Qwen2.5-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    GUIDELLM__MAX_CONCURRENCY=256,
    GUIDELLM__REQUEST_TIMEOUT=21600,
    target="http://localhost:8000/v1",
    max_seconds=30,
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    #vllm_kwargs={"enable-chunked-prefill": True}

    judgement_setting_file='arena-hard-v2.0.yaml',
    judgement_endpoint_file='api_config.yaml',
)

#task.execute_remotely("oneshot-a100x1")
task.execute_remotely("remote-upgrade-default")
#task.execute_locally()
