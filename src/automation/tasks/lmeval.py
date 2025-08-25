from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence
import os

class LMEvalTask(BaseTask):

    lmeval_packages = [
        #"vllm",
        "vllm==0.9.1",
        "git+https://github.com/EleutherAI/lm-evaluation-harness.git",
        "numpy==2.1",
        "hf_xet",
        "rouge-score",
        "bert-score",
        "nltk",
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
        model: str="vllm",
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
                    self.lmeval_packages.pop("vllm")
                if "lm-evaluation-harness" in package:
                    self.lmeval_packages.pop("git+https://github.com/EleutherAI/lm-evaluation-harness.git")
            packages = list(set(packages + self.lmeval_packages))
        else:
            packages = self.lmeval_packages

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

        # Check for conflicts in configs and constructor arguments
        # Deal with model_args_separetely
        for key in config_kwargs:
            if key == "model_args":
                continue

            if key in kwargs:
                raise ValueError(f"{key} already defined in config's model_args. It can't be defined again in task instantiation.")
            elif key == "model":
                model = config_kwargs.pop(key)

        # model_args is the only argument that can be provided
        # in both the config and in the constructor, assuming
        # the keys used in model_args are complementary
        if "model_args" in kwargs:
            model_args = dict(item.split("=") for item in kwargs.pop("model_args").split(","))
        else:
            model_args = {}

        if "model_args" in config_kwargs:
            config_model_args = dict(item.split("=") for item in config_kwargs.pop("model_args").split(","))

            for key in model_args.keys():
                if key in config_model_args:
                    raise ValueError(f"{key} already defined in config. It can't be defined again in task instantiation.")

            model_args.update(config_model_args)

        # Set default dtype and enable_chunked_prefill
        if "dtype" not in model_args:
            model_args["dtype"] = "auto"

        if "enable_chunked_prefill" not in model_args:
            model_args["enable_chunked_prefill"] = True

        if "enforce_eager" not in model_args:
            model_args["enforce_eager"] = True

        kwargs["model_args"] = ",".join(f"{k}={v}" for k, v in model_args.items())
        
        kwargs.update(config_kwargs)
        kwargs["model"] = model

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.lm_eval = kwargs
        self.force_download = force_download
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "lmeval_script.py")


    def script(self, configurations):
        from automation.tasks.scripts.lmeval_script import main
        main(configurations)


    def get_configurations(self):
        return {
            "lm_eval": self.lm_eval,
        }


    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
            },
        }
