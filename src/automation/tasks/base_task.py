from clearml import Task
from typing import Sequence, Optional

class BaseTask():

    base_packages = ["git+https://github.com/neuralmagic/reserch.git@alex-development"]

    def __init__(
            self,
            project_name: str,
            task_name: str,
            docker_image: str,
            packages: Optional[Sequence[str]]=None,
            task_type: str="training",
    ):
        
        if packages is not None:
            packages = list(set(packages + self.base_packages))
        else:
            packages = self.base_packages

        self.project_name = project_name
        self.task_name = task_name
        self.docker_image = docker_image
        self.packages = packages
        self.task_type = task_type


    def script(self):
        return NotImplementedError
    

    def get_arguments(self):
        return NotImplementedError


    def execute_remotely(self, queue_name):
        task = Task.create(
            project_name=self.project_name, 
            task_name=self.task_name, 
            task_type=self.task_type, 
            docker=self.docker_image, 
            packages=self.packages, 
            add_task_init_call=True,
            script=self.script_path,
            repo="https://github.com/neuralmagic/research.git",
            branch="alex-development",
        )

        task.connect(self.get_arguments(), "Args")
        task.execute_remotely(queue_name=queue_name, clone=False, exit_process=True)

    def execute_locally(self):
        task = Task.init(
            project_name=self.project_name, 
            task_name=self.task_name, 
            task_type=self.task_type, 
        )

        task.connect(self.get_arguments(), "Args")
        self.script()   

