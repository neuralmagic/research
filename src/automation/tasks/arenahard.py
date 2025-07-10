from automation.tasks import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE, DEFAULT_RESEARCH_BRANCH
from typing import Optional, Sequence
import os

#DEFAULT_SERVER_WAIT_TIME = 30 # 600 seconds = 10 minutes
DEFAULT_SERVER_WAIT_TIME = 600 # 600 seconds = 10 minutes

class ArenaHardGenerateTask(BaseTask):

    """
    arenahard_packages = [
        "vllm",
        "hf_xet",
        "shortuuid",
        "tiktoken",
        #"openai",
        "numpy",
        "pandas",
        "shortuuid",
        "tqdm",
        "gradio==5.25.2",
        "plotly",
        "scikit-learn",
        "boto3",
    ]
    """

    arenahard_packages = [
        "vllm",
        "hf_xet",
        "shortuuid",
        "boto3",
    ]

    """
    arenahard_packages = [
        "tiktoken",
        "openai",
        "numpy",
        "pandas",
        "shortuuid",
        "tqdm",
        "gradio==5.25.2",
        "plotly",
        "scikit-learn",
        "boto3",

        "vllm",
        "hf_xet",
    ]
    """

    def __init__(
        self,
        project_name: str,
        task_name: str,
        generate_model: str,
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
        if packages is not None:
            packages = list(set(packages + self.arenahard_packages))
        else:
            packages = self.arenahard_packages

        if "packages" in config_kwargs:
            packages = list(set(packages + config_kwargs.pop("packages")))

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

        # Sort arenahard kwargs from environment variables
        arenahard_kwargs = {
            "target": target,
            "backend": backend,
        }
        environment_variables = {}
        for k, v in kwargs.items():
            if k.startswith("ARENAHARD__"):
                environment_variables[k] = v
            else:
                arenahard_kwargs[k] = v

        # Store class attributes
        self.generate_model = generate_model
        self.clearml_model = clearml_model
        self.server_wait_time = server_wait_time
        self.vllm_kwargs = vllm_kwargs
        self.arenahard_kwargs = arenahard_kwargs
        self.environment_variables = environment_variables
        self.force_download = force_download
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "arenahard.py")
        #self.generate_path = os.path.join(".", "src", "automation", "arenahard", "generate.py")

    def script(self, configurations):
        from automation.tasks.scripts.arenahard_script import main
        main(configurations)


    def get_configurations(self):
        configs = {
            "ArenaHard": self.arenahard_kwargs,
        }
        if len(self.vllm_kwargs) > 0:
            configs["vLLM"] = self.vllm_kwargs

        if len(self.environment_variables) > 0:
            configs["environment"] = self.environment_variables

        return configs


    def get_arguments(self):
        return {
            "Args": {
                "generate_model": self.generate_model,
                "clearml_model": self.clearml_model,
                "server_wait_time": self.server_wait_time,
                "force_download": self.force_download,
            },
        }
