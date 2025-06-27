from automation.tasks import GuideLLMTask
import os
import sys

from clearml import Task
executable_path = os.path.dirname(sys.executable)
vllm_path = os.path.join(executable_path, "vllm")
print(f"The vllm path is: {vllm_path}")

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

task = Task.init(project_name="alexandre_debug", task_name="test_guidellm_task")
task.execute_remotely("remote-upgrade-default")
#task.execute_locally()
