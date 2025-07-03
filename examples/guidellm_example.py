
from automation.tasks import GuideLLMTask

task = GuideLLMTask(
    project_name="alexandre_debug",
    task_name="test_guidellm_task",
    #model="meta-llama/Llama-3.2-1B-Instruct",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    rate_type="throughput",
    backend="aiohttp_server",
    GUIDELLM__MAX_CONCURRENCY=256,
    GUIDELLM__REQUEST_TIMEOUT=21600,
    target="http://localhost:8000/v1",
    data_type="emulated",
    max_seconds=30,
    #data="prompt_tokens=512,generated_tokens=256,output_tokens=256",
    data="prompt_tokens=128,generated_tokens=128,output_tokens=128",
    branch = "update_guidellm",
    #vllm_kwargs={"enable-chunked-prefill": True}
)

task.execute_remotely("remote-upgrade-default")
