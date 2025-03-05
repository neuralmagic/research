from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Union, List, Optional, Sequence, Any
import os
import yaml

class LLMCompressorTask(BaseTask):
    llmcompressor_packages = ["llmcompressor"]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        recipe: Any,
        recipe_args: Optional[dict]=None,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        dataset_name: Optional[str]="calibration",
        clearml_model: bool=False,
        save_directory: str="output",
        num_samples: int=512,
        max_seq_len: int=8192,
        trust_remote_code: bool=False,
        max_memory_per_gpu: str="hessian",
        tags: Union[str, List[str]]=None,
        task_type: str="training",
        config: Optional[str]=None,
    ):

        # Process config if provided
        if config is not None:
            config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LMEvalTask and packages set in the config
        if packages is not None:
            packages = list(set(packages + self.llmcompressor_packages))
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

        # Store class attributes that may be part of config
        recipe = config_kwargs.pop("recipe", recipe)
        if isinstance(recipe, dict):
            recipe = yaml.dump(recipe, default_flow_style=False, sort_keys=False)
        elif not isinstance(recipe, str):
            from llmcompressor.recipe import Recipe
            recipe = Recipe.from_modifiers(recipe).yaml()
    
        self.recipe = recipe
        self.recipe_args = config_kwargs.pop("recipe", recipe_args)
        self.dataset_name = config_kwargs.pop("dataset_name", dataset_name)
        self.num_samples = config_kwargs.pop("num_samples", num_samples)
        self.max_seq_len = config_kwargs.pop("max_seq_len", max_seq_len)
        self.trust_remote_code = config_kwargs.pop("trust_remote_code", trust_remote_code)
        self.max_memory_per_gpu = config_kwargs.pop("max_memory_per_gpu", max_memory_per_gpu)

        if tags is not None:
            tags = list(set(config_kwargs.pop("tags", []).extend(tags)))
        else:
            tags = config_kwargs.pop("tags", None)
        self.tags = tags

        # Check for conflicts in configs and constructor arguments
        for key in config_kwargs:
            if key in kwargs:
                ValueError(f"{key} already defined in config's model_args. It can't be defined again in task instantiation.")

        kwargs.update(config_kwargs)

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.save_directory = save_directory
        self.kwargs = kwargs
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "llmcompressor_script.py")


    def script(self):
        from automation.tasks.scripts.llmcompressor_script import main
        main()


    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "recipe": self.recipe,
                "recipe_args": self.recipe_args,
                "dataset_name": self.dataset_name,
                "clearml_model": self.clearml_model,
                "save_directory": self.save_directory,
                "num_samples": self.num_samples,
                "max_seq_len": self.max_seq_len,
                "trust_remote_code": self.trust_remote_code,
                "max_memory_per_gpu": self.max_memory_per_gpu,
                "tags": self.tags,
                **kwargs,
            },
        }


