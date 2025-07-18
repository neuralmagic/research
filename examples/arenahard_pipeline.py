from automation.pipelines import Pipeline
from automation.tasks import ArenaHardGenerateTask, ArenaHardJudgeTask


step1 = ArenaHardGenerateTask(
    project_name="alexandre_debug",
    task_name="generate_task",
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

    generation_config_file='gen_answer_config.yaml',
    generation_endpoint_file='api_config.yaml',
)

step1.create_task()


step2 = ArenaHardJudgeTask(
    project_name="alexandre_debug",
    task_name="judge_task",
    answer_task_id  = "cf688bf523c842ff8d8c9d721613aabc",
    judgement_model="Qwen/Qwen2.5-1.5B-Instruct",
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

step2.create_task()


pipeline = Pipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_arenahard",
)


pipeline.add_step(
    name="pipeline_arenahard_generate_step1",
    base_task_id = step1.id,
    execution_queue="remote-upgrade-default",
    #monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    #monitor_artifacts=["recipe"],
)

pipeline.add_step(
    name="pipeline_arenahard_judgement_step2",
    base_task_id = step2.id,
    parents=["pipeline_arenahard_generate_step1"],
    execution_queue="remote-upgrade-default",
    #parameter_override={"Args/model_id": "${pipeline_arenahard_generate_step1.models.output.-1.id}"},
    #monitor_metrics=[("gsm8k", "exact_match,strict-match")],
)

pipeline.execute_remotely()
