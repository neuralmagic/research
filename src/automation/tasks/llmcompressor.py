from automation.tasks.base_task import BaseTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from automation.scripts.llmcompressor_script import main
from typing import Union, List, Optional, Sequence, Dict
import os

class LLMCompressorTask(BaseTask):

    default_packages = ["llmcompressor"]

    def __init__(
            self,
            project_name: str,
            task_name: str,
            model_id: str,
            recipe: Union[str, Dict],
            docker_image: str=DEFAULT_DOCKER_IMAGE,
            packages: Optional[Sequence[str]]=None,
            dataset_name: str="calibration",
            clearml_model: bool=False,
            save_directory: str="output",
            num_samples: int=512,
            max_seq_len: int=8192,
            trust_remote_code: bool=False,
            max_memory_per_gpu: str="hessian",
            dtype: str="auto",
            tags: Union[str, List[str]]=None,
            task_type: str="training",
    ):
    
        if packages is not None:
            packages = list(set(packages + self.default_packages))
        else:
            packages = self.default_packages

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
        self.clearml_model = clearml_model
        self.save_directory = save_directory
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.trust_remote_code = trust_remote_code
        self.max_memory_per_gpu = max_memory_per_gpu
        self.dtype = dtype
        self.tags = tags
        self.script_path = os.path.join(".", "src", "automation", "scripts", "llmcompressor_script.py")
        self.script = main
    
    def get_arguments(self):
        # Connect parameters to ClearML
        return {
            "model_id": self.model_id,
            "recipe": self.recipe,
            "dataset_name": self.dataset_name,
            "clearml_model": self.clearml_model,
            "save_directory": self.save_directory,
            "num_samples": self.num_samples,
            "max_seq_len": self.max_seq_len,
            "trust_remote_code": self.trust_remote_code,
            "max_memory_per_gpu": self.max_memory_per_gpu,
            "dtype": self.dtype,
            "tags": self.tags,
        }


