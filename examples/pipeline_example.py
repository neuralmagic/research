from automation.pipelines import Pipeline
from automation.standards import OpenLLMTask, QuantizationW4A16Task


step1 = QuantizationW4A16Task(
    project_name="alexandre_debug",
    task_name="pipeline_example_quantization",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    damping_frac=0.1,
)
step1.create_task()

step1_model_id = "${pipeline_example_quantization_step1.models.output.-1.id}"

step2 = OpenLLMTask(
    project_name="alexandre_debug",
    task_name="pipeline_example_openllm",
    model_id="dummy",
    clearml_model=True,
)
step2.create_task()

pipeline = Pipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_example",
    version="0.0.1",
)

pipeline.add_step(
    name="pipeline_example_quantization_step1",
    base_task_id = step1.id,
    execution_queue="oneshot-a5000x1",
    monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    monitor_artifacts=["recipe"],
)

pipeline.add_step(
    name="pipeline_example_quantization_step2",
    base_task_id = step2.id,
    parents=["pipeline_example_quantization_step1"],
    execution_queue="oneshot-a5000x1",
    parameter_override={"Args/model_id": step1_model_id},
    monitor_metrics=[("Summary", "openllm")],
)

pipeline.execute_remotely()