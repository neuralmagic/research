from automation.pipelines import Pipeline
from automation.tasks import LMEvalTask, LLMCompressorTask

recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      scheme: "W8A16"
      targets: "Linear"
      observer: "mse"
"""

step1 = LLMCompressorTask(
    project_name="alexandre_debug",
    task_name="pipeline_example_quantization",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    recipe=recipe,
)
step1.create_task()

step2 = LMEvalTask(
    project_name="alexandre_debug",
    task_name="pipeline_example_evaluation",
    model_id="dummuy",
    clearml_model=True,
    tasks="gsm8k",
    num_fewshot=5,
)
step2.create_task()

pipeline = Pipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_example",
)


pipeline.add_step(
    name="pipeline_example_quantization_step1",
    base_task_id = step1.id,
    execution_queue="oneshot-a100x1",
    monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
    monitor_artifacts=["recipe"],
)

pipeline.add_step(
    name="pipeline_example_quantization_step2",
    base_task_id = step2.id,
    parents=["pipeline_example_quantization_step1"],
    execution_queue="oneshot-a100x1",
    parameter_override={"Args/model_id": "${pipeline_example_quantization_step1.models.output.-1.id}"},
    monitor_metrics=[("gsm8k", "exact_match,strict-match")],
)

pipeline.execute_remotely()