from automation.pipelines import LLMCompressorLMEvalPipeline
from automation.hpo import BaseHPO
from clearml.automation import UniformParameterRange, DiscreteParameterRange

pipeline = LLMCompressorLMEvalPipeline(
    project_name="alexandre_debug",
    pipeline_name="hpo_example_pipeline",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    execution_queues=["oneshot-a100x1", "oneshot-a100x1"],
    config="pipeline_w4a16",
    lmeval_kwargs={"evaluation": {"model_args": "gpu_memory_utilization=0.4,enable_chunked_prefill=True", "batch_size": "auto", "limit": 10}},
)

pipeline.create_pipeline()

hpo_task = BaseHPO(
    project_name="alexandre_debug",
    task_name="hpo_example",
    report_period_min=1,
    time_limit_min=30,
    optimizer="GridSearch",
    objective_metric_title="openllm",
    objective_metric_series="average",
    objective_metric_sign="max",
    total_max_jobs=5,
    pool_period_min=1,
    max_iteration_per_job=1,
    spawn_project="hpo_debug",
    base_task_id= pipeline.id,
)

hpo_task.add_parameter(UniformParameterRange("Args/dampening_frac", min_value=0.01, max_value=0.1, step_size=0.01))
hpo_task.add_parameter(DiscreteParameterRange("Args/observer", values=["mse", "minmax"]))

hpo_task.execute_remotely()
