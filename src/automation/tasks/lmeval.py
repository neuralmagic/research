from automation.tasks.base_task import BaseTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from automation.scripts.lmeval_script import main
from typing import Union, List, Optional, Sequence, Dict
import os

class LMEvalTask(BaseTask):

    lmeval_packages = ["git+https://github.com/neuralmagic/lm-evaluation-harness.git@llama_3.1_instruct"]

    def __init__(
            self,
            project_name: str,
            task_name: str,
            model_id: str,
            docker_image: str=DEFAULT_DOCKER_IMAGE,
            packages: Optional[Sequence[str]]=None,
            clearml_model: bool=False,
            task_type: str="training",
            **kwargs,
    ):
        
        if packages is not None:
            packages = list(set(packages + self.lmeval_packages))
        else:
            packages = self.lmeval_packages

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
        self.lm_eval = kwargs
        self.script_path = os.path.join(".", "src", "automation", "scripts", "lmeval_script.py")
        self.script = main

    def get_arguments(self):
        return {
            **self.lm_eval,
            "model_id": self.model_id,
            "clearml_model": self.clearml_model,
        }
