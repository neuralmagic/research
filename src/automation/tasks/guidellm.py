from automation.tasks import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE, DEFAULT_RESEARCH_BRANCH
from typing import Optional, Sequence
import os

DEFAULT_SERVER_WAIT_TIME = 600 # 600 seconds = 10 minutes
#GUIDELLM_PACKAGE = "git+https://github.com/neuralmagic/guidellm.git@main"
#GUIDELLM_PACKAGE = "git+https://github.com/neuralmagic/guidellm.git@use-old-run"
GUIDELLM_PACKAGE = "git+https://github.com/neuralmagic/guidellm.git@main#egg=guidellm[dev]"

class GuideLLMTask(BaseTask):

    guidellm_packages = [
        "vllm",
        GUIDELLM_PACKAGE,
        "hf_xet",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model: str,
        server_wait_time: int=DEFAULT_SERVER_WAIT_TIME,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        branch: str= DEFAULT_RESEARCH_BRANCH,
        task_type: str="training",
        vllm_kwargs: dict={},
        target: str="http://localhost:8000/v1",
        backend: str="aiohttp_server",
        force_download: bool=False,
        config: Optional[str]=None,
        **kwargs,
    ):

        # Process config
        config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LMEvalTask and packages set in the config
        print(self.guidellm_packages)
        print(packages)
        if packages is not None:
            packages = list(set(packages + self.guidellm_packages))
        else:
            packages = self.guidellm_packages

        print(packages)
        if "packages" in config_kwargs:
            packages = list(set(packages + config_kwargs.pop("packages")))

        print(packages)
        # Initialize base parameters
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
            branch = branch,
        )

        # Check for conflicts in configs and constructor arguments
        for key in config_kwargs:
            if key in kwargs:
                raise ValueError(f"{key} already defined in config's model_args. It can't be defined again in task instantiation.")

        kwargs.update(config_kwargs)

        # Sort guidellm kwargs from environment variables
        guidellm_kwargs = {
            "target": target,
            "backend": backend,
        }
        environment_variables = {}
        for k, v in kwargs.items():
            if k.startswith("GUIDELLM__"):
                environment_variables[k] = v
            else:
                guidellm_kwargs[k] = v

        # Store class attributes
        self.model = model
        self.clearml_model = clearml_model
        self.server_wait_time = server_wait_time
        self.vllm_kwargs = vllm_kwargs
        self.guidellm_kwargs = guidellm_kwargs
        self.environment_variables = environment_variables
        self.force_download = force_download
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "guidellm_script.py")


    def script(self, configurations):
        from automation.tasks.scripts.guidellm_script import main
        main(configurations)


    def get_configurations(self):
        configs = {
            "GuideLLM": self.guidellm_kwargs,
        }
        if len(self.vllm_kwargs) > 0:
            configs["vLLM"] = self.vllm_kwargs

        if len(self.environment_variables) > 0:
            configs["environment"] = self.environment_variables

        return configs


    def get_arguments(self):
        return {
            "Args": {
                "model": self.model,
                "clearml_model": self.clearml_model,
                "server_wait_time": self.server_wait_time,
                "force_download": self.force_download,
            },
        }
