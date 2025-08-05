from automation.tasks import ArenaHardJudgeTask

task = ArenaHardJudgeTask(
    project_name="alexandre_debug",
    task_name="test_judge_task",
    #answer_task_id  = "cf688bf523c842ff8d8c9d721613aabc",
    #answer_task_id = "4630730469114ed397fc876d578a469e",
    #judgement_model="meta-llama/Llama-3.2-1B-Instruct",
    #judgement_model="Qwen/Qwen2.5-1.5B-Instruct",
    judgement_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    max_seconds=30,
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    #vllm_kwargs={"enable-chunked-prefill": True}

    #judgement_setting_file='arena-hard-v2.0.yaml',
    judgement_setting_file='math-arena-hard-v2.0.yaml',
    #judgement_endpoint_file='api_config.yaml',
    judgement_endpoint_file ='math_api_config.yaml',
)

#task.execute_remotely("oneshot-a100x1")
task.execute_remotely("remote-upgrade-default")
#task.execute_locally()
