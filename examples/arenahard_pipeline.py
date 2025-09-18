from automation.pipelines import Pipeline
from automation.tasks import ArenaHardGenerateTask, ArenaHardJudgeTask

arena_hard_project_name="dsmall_pipeline_debug"
step1 = ArenaHardGenerateTask(
    project_name= arena_hard_project_name,
    task_name="generate_pipeline_task1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1","vllm==0.10.1.1"],
    generate_model="meta-llama/Llama-3.2-1B-Instruct",
    question_size = "small",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    server_wait_time = 600, 
    bench_name = "arena-hard-v2.0",
    #bench_name = "arena-hard-v0.1",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1024, 
)

step1.create_task()


step2 = ArenaHardJudgeTask(
    project_name= arena_hard_project_name,
    task_name="judge_pipeline_task1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1","vllm==0.10.1.1"],
    #judgement_model ="Qwen/Qwen2-7B-Instruct",
    judgement_model = "openai/gpt-oss-120b",
    question_size = "small",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    server_wait_time = 600, 
    bench_name = "arena-hard-v2.0",
    #bench_name = "arena-hard-v0.1",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1024, 
)

step2.create_task()

pipeline = Pipeline(
    project_name= arena_hard_project_name,
    pipeline_name="pipeline_arenahard",
)


pipeline.add_step(
    name="pipeline_arenahard_gen_step_1",
    base_task_id = step1.id,
    execution_queue="oneshot-a100x1",
    #monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    #monitor_artifacts=["recipe"],
)



config_override = {**step1.get_configurations()['ArenaHard'], **{"answer_project_name": arena_hard_project_name}, **{"answer_task_name" : pipeline.steps[0][1]['name'] }}

pipeline.add_step(
    name="pipeline_arenahard_judge_step_2",
    base_task_id = step2.id,
    parents=["pipeline_arenahard_gen_step_1"],
    execution_queue="oneshot-a100x4",
    #parameter_override={"Args/model_id": "${pipeline_arenahard_generate_step1.models.output.-1.id}"},
    configuration_overrides={"ArenaHard" : config_override },
    #monitor_metrics=[("gsm8k", "exact_match,strict-match")],
)


pipeline.execute_remotely()
#pipeline.execute_remotely("oneshot-a100x1")
