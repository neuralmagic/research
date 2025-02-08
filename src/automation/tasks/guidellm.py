from automation.tasks import BaseTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence
import inspect
import os

DEFAULT_SERVER_WAIT_TIME = 600 # 600 seconds = 10 minutes

class GuideLLMTask(BaseTask):

    guidellm_packages = [
        "vllm",
        "git+https://github.com/neuralmagic/guidellm.git@http_backend",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        server_wait_time: int=DEFAULT_SERVER_WAIT_TIME,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        task_type: str="training",
        **kwargs,
    ):

        from automation.scripts.guidellm_script import main
        from guidellm import generate_benchmark_report

        if packages is not None:
            packages = list(set(packages + self.guidellm_packages))
        else:
            packages = self.guidellm_packages

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
        )

        # Sort vllm and guidellm kwargs
        guidellm_signature = inspect.signature(generate_benchmark_report)

        vllm_kwargs = {}
        guidellm_kwargs = {}
        environment_variables = {}
        for k, v in kwargs.items():
            if k in guidellm_signature.parameters:
                guidellm_kwargs[k] = v
            elif k.startswith("GUIDELLM__"):
                environment_variables[k] = v
            else:
                vllm_kwargs[k] = v

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.server_wait_time = server_wait_time
        self.vllm_kwargs = vllm_kwargs
        self.guidellm_kwargs = guidellm_kwargs
        self.environment_variables = environment_variables
        self.script_path = os.path.join(".", "src", "automation", "scripts", "guidellm_script.py")
        self.script = main

    def get_arguments(self):
        args = {
            "Args": {
                "model_id": self.model_id,
                "clearml_model": self.clearml_model,
                "server_wait_time": self.server_wait_time,
            },
            "GuideLLM": self.guidellm_kwargs,
        }

        if len(self.vllm_kwargs) > 0:
            args["vLLM"] = self.vllm_kwargs

        if len(self.enviornment_variables) > 0:
            args["environment"] = self.environment_variables

        return args