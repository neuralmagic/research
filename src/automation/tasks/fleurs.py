from automation.tasks.base_task import BaseTask
from automation.requests import SUPPORTED_REQUESTS
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence, Callable, Union
from automation.utils import serialize_callable
import os

class FleursTask(BaseTask):

    fleurs_packages = [
        "vllm",
        "jiwer",
        "more_itertools",
        "torchdecoder",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        task_type: str="training",
        force_download: bool=False,
        config: Optional[str]=None,
        request: Union[Callable, str]="mistral_transcript_request",
        vllm_args: Optional[dict]=None,
        target="http://localhost:8000/v1",
        server_wait_time=120,
        **kwargs,
    ):

        # Process config
        config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LMEvalTask and packages set in the config
        if packages is not None:
            # If a specific version of vLLM is specified in packages,
            # use that version instead of the latest
            for package in packages:
                if "vllm" in package:
                    self.fleurs_packages.pop("vllm")
            packages = list(set(packages + self.fleurs_packages))
        else:
            packages = self.fleurs_packages

        if "packages" in config_kwargs:
            packages = list(set(packages + config_kwargs.pop("packages")))

        # Initialize base parameters
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
        )

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.fleurs_args = kwargs
        self.force_download = force_download
        self.request = request
        self.vllm_args = vllm_args
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "fleurs_script.py")


    def script(self, configurations):
        from automation.tasks.scripts.fleurs_script import main
        main(configurations)


    def get_configurations(self):
        configs = {
            "fleurs_args": self.fleurs_args,
            "vllm_args": self.vllm_args,
            "target": self.target,
            "server_wait_time": self.server_wait_time,
        }
        if isinstance(self.request, str):
            if self.request in SUPPORTED_REQUESTS:
                configs["request"] = SUPPORTED_REQUESTS[self.request]
            else:
                raise ValueError(f"Request {self.request} not supported")
        elif callable(self.request):
            configs["request"] = serialize_callable(self.request)
        else:
            raise ValueError(f"Request {self.request} must be a string or a callable")
        
        return configs

    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
            },
        }


