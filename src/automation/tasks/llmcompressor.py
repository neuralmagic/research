from automation.tasks.base_task import BaseTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from typing import Union, List, Optional, Sequence, Any
import os

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
        dataset_name: str="calibration",
        clearml_model: bool=False,
        save_directory: str="output",
        num_samples: int=512,
        max_seq_len: int=8192,
        trust_remote_code: bool=False,
        max_memory_per_gpu: str="hessian",
        tags: Union[str, List[str]]=None,
        task_type: str="training",
    ):
    
        if packages is not None:
            packages = list(set(packages + self.llmcompressor_packages))
        else:
            packages = self.llmcompressor_packages

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
        )

        # Store class attributes
        self.model_id = model_id
        self.recipe = recipe
        self.recipe_args = recipe_args
        self.clearml_model = clearml_model
        self.save_directory = save_directory
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.trust_remote_code = trust_remote_code
        self.max_memory_per_gpu = max_memory_per_gpu
        self.tags = tags
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "llmcompressor_script.py")


    def script(self):
        from automation.tasks.scripts.llmcompressor_script import main
        main()


    def get_arguments(self):

        recipe = self.recipe
        if not isinstance(recipe, dict) and not isinstance(recipe, str):
            from llmcompressor.recipe import Recipe
            recipe = Recipe.from_modifiers(recipe).yaml()

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
            },
        }


