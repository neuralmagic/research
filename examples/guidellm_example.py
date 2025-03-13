from automation.tasks import GuideLLMTask

task = GuideLLMTask(
    project_name="alexandre_debug",
    task_name="test_guidellm_task",
    model="meta-llama/Llama-3.2-1B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    GUIDELLM__MAX_CONCURRENCY=256,
    GUIDELLM__REQUEST_TIMEOUT=21600,
    target="http://localhost:8000/v1",
    data_type="emulated",
    max_seconds=30,
    data="prompt_tokens=512,generated_tokens=256",
    vllm_kwargs={"enable-chunked-prefill": True}
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()