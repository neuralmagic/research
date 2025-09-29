from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE, DEFAULT_RESEARCH_BRANCH
from automation.utils import is_yaml_content, merge_dicts
from typing import Optional, Sequence
import yaml
import os

class LightEvalTask(BaseTask):

    lihghteval_packages = [
        "torch==2.7.1", 
        "torchaudio==2.7.1",
        "torchvision==0.22.1",
        "vllm==0.10.1.1",
        "transformers==4.56.2",
        "huggingface-hub==0.35.2",
        "triton==3.3.1",
        "lighteval==0.11.0",
        "math-verify==0.5.2",
        "more-itertools==10.8.0",
        "latex2sympy2_extended==1.0.6",
        "langdetect==1.0.9",
        "openai==1.109.1",
        "hf-xet==1.1.10",
        "emoji==2.15.0",
        "immutabledict==4.2.1",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        branch: str= DEFAULT_RESEARCH_BRANCH,
        task_type: str="training",
        force_download: bool=False,
        config: Optional[str]=None,
        **kwargs,
    ):

        # Process config
        config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LightEvalTask and packages set in the config
        if packages is not None:
            packages = list(set(packages + self.lihghteval_packages))
        else:
            packages = self.lihghteval_packages

        if "packages" in config_kwargs:
            packages = list(set(packages + config_kwargs.pop("packages")))

        # Initialize base parameters
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
            branch=branch,
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
        metric_options = None
        if "model_args" in kwargs:
            if kwargs["model_args"].endswith(".yaml"):
                file = open(kwargs["model_args"], "r")
                if is_yaml_content(file):
                    data = yaml.safe_load(file)
                    model_args = data.get("model_parameters", {})
                    if "metric_options" in data:
                        metric_options = data.get("metric_options")
                else:
                    raise Exception("Cannot parse model_args")
                file.close()
            elif is_yaml_content(kwargs["model_args"]):
                data = yaml.safe_load(kwargs["model_args"])
                model_args = data.get("model_parameters", {})
                if "metric_options" in data:
                    metric_options = data.get("metric_options")
            else:
                model_args = dict(item.split("=") for item in kwargs.pop("model_args").split(","))
        else:
            model_args = {}

        if "model_args" in config_kwargs:
            if isinstance(config_kwargs["model_args"], str):
                config_model_args = dict(item.split("=") for item in config_kwargs.pop("model_args").split(","))
            else:
                config_model_args = config_kwargs.get("model_args")

            model_args = merge_dicts(model_args, config_model_args)

        # Set default dtype and enable_chunked_prefill
        if "dtype" not in model_args:
            model_args["dtype"] = "auto"

        kwargs["model_args"] = model_args
        if metric_options is not None:
            kwargs["metric_options"] = metric_options
        
        kwargs.update(config_kwargs)

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.lighteval_args = kwargs
        self.force_download = force_download
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "lighteval_script.py")


    def script(self, configurations, args):
        from automation.tasks.scripts.lighteval_script import main
        main(configurations, args)


    def get_configurations(self):
        return {
            "lighteval_args": self.lighteval_args,
        }


    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
            },
        }
