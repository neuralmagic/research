from automation.tasks.guidellm import GuideLLMTask, DEFAULT_SERVER_WAIT_TIME
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence


class BenchmarkingRAGTask(GuideLLMTask):

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        server_wait_time: int=DEFAULT_SERVER_WAIT_TIME,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        vllm_kwargs: dict={},
        **kwargs,
    ):
        
        if "data-type" in kwargs:
            raise ValueError("data-type should not be specified with CodeGenerationBenchmarking")
        
        if "data" in kwargs:
            raise ValueError("data should not be specified with CodeGenerationBenchmarking")
        
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            model_id=model_id,
            server_wait_time=server_wait_time,
            docker_image=docker_image,
            packages=packages,
            clearml_model=clearml_model,
            data_type="emulated",
            data="prompt_tokens=1024,generated_tokens=128",
            vllm_kwargs=vllm_kwargs,
            **kwargs,
        )