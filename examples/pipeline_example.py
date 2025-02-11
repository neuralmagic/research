from automation.pipelines import Pipeline
from automation.standards import OpenLLMTask, QuantizationW4A16Task


step1 = QuantizationW4A16Task(
    project_name="alexandre_debug",
    task_name="pipeline_example_quantization",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    damping_frac=0.1,
)
step1.create_task()

step1_model_id = "${{pipeline_example_quantization.models.output.-1.id}}"

step2 = OpenLLMTask(
    project_name="alexandre_debug",
    task_name="pipeline_example_openllm",
    model_id=step1_model_id,
    clearml_model=True,
)
step2.create_task()

pipeline = Pipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_example",
    version="0.0.1",
)

pipeline.add_step(
    name="quantization",
    base_task_id = step1.id,
    execution_queue="oneshot-a5000x1",
    monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    monitor_artifacts=["recipe"],
)

pipeline.add_step(
    name="openllm",
    base_task_id = step2.id,
    parents=["quantization"],
    execution_queue="oneshot-a5000x1",
    monitor_metrics=[("Summary" "gsm8k/5shot/exact_match,strict-match")],
)

pipeline.execute_remotely()