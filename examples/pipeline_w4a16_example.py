from automation.pipelines import LLMCompressorLMEvalPipeline

pipeline = LLMCompressorLMEvalPipeline(
    project_name="alexandre_debug",
    pipeline_name="pipeline_w4a16_example",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    execution_queues=["oneshot-a100x1", "oneshot-a100x1"],
    config="pipeline_w4a16",
    openllm_kwargs={"model_args": "gpu_memory_utilization=0.4,enable_chunked_prefill=True"}
)

pipeline.execute_remotely()