from automation.tasks import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional
from clearml import Task
from clearml.automation.parameters import Parameter
import os

class BaseHPO(BaseTask):

    hpo_packages = [
        "optuna",
        "hpbandster",
        "huggingface_hub",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        optimizer: str="GridSearch",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        report_period_min: Optional[float]=None,
        job_complete_callback: Optional[str]="job_summary",
        job_complete_callback_kwargs: Optional[dict]=None,
        optimization_complete_callback: Optional[str]=None,
        optimization_complete_callback_kwargs: Optional[dict] = None,
        time_limit_min: Optional[float]=None,
        **kwargs,
    ):

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=self.hpo_packages,
            task_type=Task.TaskTypes.optimizer,
        )

        self.kwargs = kwargs
        self.kwargs.update({
            "report_period_min": report_period_min,
            "time_limit_min": time_limit_min,
            "job_complete_callback": job_complete_callback,
            "job_complete_callback_kwargs": job_complete_callback_kwargs,
            "optimization_complete_callback": optimization_complete_callback,
            "optimization_complete_callback_kwargs": optimization_complete_callback_kwargs,
            "optimizer": optimizer,
        })
        self.script_path = os.path.join(".", "src", "automation", "hpo", "hpo_script.py")
        self.parameters = []


    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter.to_dict())


    def script(self):
        from automation.hpo.hpo_script import main
        main()


    def get_configurations(self):
        return {
            "Parameters": self.parameters,
            "Optimization": self.kwargs,
        }

   
    def execute_remotely(self, queue_name: str="services"):
        super().execute_remotely(queue_name)

    

