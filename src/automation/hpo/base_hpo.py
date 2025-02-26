from automation.tasks import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional
from clearml import Task
from clearml.automation.parameters import Parameter

class BaseHPO(BaseTask):

    hpo_packages = [
        "optuna",
        "hpbandster",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        optimizer: str="GridSearch",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        report_period_min: Optional[float]=None,
        job_complete_callback: Optional[str]="job_summary",
        optimization_complete_callback: Optional[str]="push_to_hf",
        time_limit_min: Optional[float]=None,
        **kwargs,
    ):

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            docker_image=docker_image,
            packages=self.hpo_packages,
            task_type=Task.TaskTypes.optimization,
        )

        self.kwargs = kwargs
        self.kwargs.update({
            "report_period_min": report_period_min,
            "time_limit_min": time_limit_min,
            "job_complete_callback": job_complete_callback,
            "optimizer": optimizer,
        })
        self.script_path = os.path.join(".", "src", "automation", "hpo", "hpo_script.py")
        self.parameters = []


    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter.to_dict())


    def script(self):
        from automation.hpo.hpo_script import main
        main()


    def connect_configuration(self):
        self.task.connect_configuration(self.parameters, "Parameters")
        self.task.connect_configuration(self.kwargs, "Optimization")


    def get_arguments(self):
        return {}


    

