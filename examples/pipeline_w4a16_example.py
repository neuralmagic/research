from automation.standards import QuantizationW4A16Pipeline

pipeline = QuantizationW4A16Pipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_w4a16_example",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    execution_queues=["oneshot-a5000x1", "oneshot-a6000x1"],
    damping_frac=0.1,
)

pipeline.execute_remotely()