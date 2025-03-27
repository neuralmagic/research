from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Union, List, Optional, Sequence, Any, Callable
import os
import yaml

class LLMCompressorTask(BaseTask):
    llmcompressor_packages = ["git+https://github.com/vllm-project/llm-compressor.git"]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        recipe: Optional[Any]=None,
        recipe_args: Optional[dict]=None,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        dataset_name: Optional[str]="calibration",
        dataset_loader: Optional[Callable]=None,
        clearml_model: bool=False,
        force_download: bool=False,
        save_directory: str="output",
        text_samples: Optional[int]=None,
        vision_samples: Optional[int]=None,
        num_samples: Optional[int]=512,
        max_seq_len: int=8192,
        trust_remote_code: bool=False,
        max_memory_per_gpu: str="hessian",
        tracing_class: Optional[str]=None,
        tags: Union[str, List[str]]=None,
        task_type: str="training",
        config: Optional[str]=None,
    ):

        # Process config
        config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LMEvalTask and packages set in the config
        if packages is not None:
            packages = list(set(packages + self.llmcompressor_packages))
        else:
            packages = self.llmcompressor_packages

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
        if "recipe" in config_kwargs and recipe is not None:
            raise ValueError("Recipe is already provided in config. It can't be provided in task instantiation.")
        
        recipe = config_kwargs.pop("recipe", recipe)
        if recipe is None:
            raise ValueError("Recipe must be provided.")

        if isinstance(recipe, dict):
            recipe = yaml.dump(recipe, default_flow_style=False, sort_keys=False)
        elif not isinstance(recipe, str):
            from llmcompressor.recipe import Recipe
            recipe = Recipe.from_modifiers(recipe).yaml()

        self.recipe = recipe

        if recipe_args is None:
            self.recipe_args = config_kwargs.pop("recipe_args", None)
        else:
            config_recipe_args = config_kwargs.pop("recipe_args", {})
            config_recipe_args.update(recipe_args)
            self.recipe_args = config_recipe_args

        self.dataset_name = config_kwargs.pop("dataset_name", dataset_name)
        self.text_samples = config_kwargs.pop("text_samples", text_samples)
        self.vision_samples = config_kwargs.pop("vision_samples", vision_samples)
        self.num_samples = config_kwargs.pop("num_samples", num_samples)
        self.max_seq_len = config_kwargs.pop("max_seq_len", max_seq_len)
        self.trust_remote_code = config_kwargs.pop("trust_remote_code", trust_remote_code)
        self.max_memory_per_gpu = config_kwargs.pop("max_memory_per_gpu", max_memory_per_gpu)
        self.dataset_loader = dataset_loader
        self.tracing_class = tracing_class

        if tags is not None:
            tags = list(set(config_kwargs.pop("tags", []).extend(tags)))
        else:
            tags = config_kwargs.pop("tags", None)
        self.tags = tags

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.force_download = force_download
        self.save_directory = save_directory
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "llmcompressor_script.py")


    def script(self):
        self.set_dataset_loader()
        from automation.tasks.scripts.llmcompressor_script import main
        main()


    def set_dataset_loader(self):
        if self.dataset_loader is not None:
            self.task.upload_artifact("dataset loader", self.dataset_loader)
    

    def create_task(self):
        super().create_task()
        self.set_dataset_loader()


    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "recipe": self.recipe,
                "recipe_args": self.recipe_args,
                "dataset_name": self.dataset_name,
                "dataset_loader": "dataset loader" if self.dataset_loader else None,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
                "save_directory": self.save_directory,
                "text_samples": self.text_samples,
                "vision_samples": self.vision_samples,
                "num_samples": self.num_samples,
                "max_seq_len": self.max_seq_len,
                "trust_remote_code": self.trust_remote_code,
                "max_memory_per_gpu": self.max_memory_per_gpu,
                "tracing_class": self.tracing_class,
                "tags": self.tags,
            },
        }


