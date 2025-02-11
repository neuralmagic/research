from automation.tasks import GuideLLMTask

task = GuideLLMTask(
    project_name="alexandre_debug",
    task_name="test_guidellm_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    GUIDELLM__MAX_CONCURRENCY=256,
    target="http://localhost:8030/v1",
    data_type="emulated",
    data="prompt_tokens=512,generated_tokens=256"
)

task.execute_remotely("oneshot-a6000x1")
#task.execute_locally()