from automation.pipelines import Pipeline
from automation.tasks import ArenaHardGenerateTask, ArenaHardJudgeTask

step1 = ArenaHardGenerateTask(
    project_name="simple_debug",
    task_name="generate_math_task_1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1","vllm==0.10.0"],
    generate_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    server_wait_time = 600, 
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1300, 

)

step1.create_task()


step2 = ArenaHardJudgeTask(
    project_name="simple_debug",
    task_name="test_judge_task_1",
    packages = ["huggingface-hub==0.34.3", "triton==3.3.1","vllm==0.10.0"],
    judgement_model="meta-llama/Meta-Llama-3-8B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    target="http://localhost:8000/v1",
    server_wait_time = 600, 
    data="prompt_tokens=128,output_tokens=128",
    branch = "arena_upgrade",
    vllm_kwargs={"enable-chunked-prefill": True},
    max_tokens = 1300, 
)

step2.create_task()

pipeline = Pipeline(
    project_name="simple_debug",
    pipeline_name="pipeline_arenahard",
)


pipeline.add_step(
    name="pipeline_arenahard_generate_step1",
    base_task_id = step1.id,
    execution_queue="oneshot-a100x1",
    #monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    #monitor_artifacts=["recipe"],
)



config_override = {**step1.get_configurations()['ArenaHard'], **{"answer_task_name": step1.name}, **{"answer_project_name" : step1.project_name } }

pipeline.add_step(
    name="pipeline_arenahard_judgement_step2",
    base_task_id = step2.id,
    parents=["pipeline_arenahard_generate_step1"],
    execution_queue="oneshot-a100x1",
    #parameter_override={"Args/model_id": "${pipeline_arenahard_generate_step1.models.output.-1.id}"},
    configuration_overrides={"ArenaHard" : config_override },
    #monitor_metrics=[("gsm8k", "exact_match,strict-match")],
)

pipeline.execute_remotely("oneshot-a100x1")
