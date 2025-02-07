from clearml import Task
from typing import Sequence, Optional

class BaseTask():

    def __init__(
            self,
            project_name: str,
            task_name: str,
            docker_image: str,
            packages: Optional[Sequence[str]],
            task_type: str="training",
    ):
        self.project_name = project_name
        self.task_name = task_name
        self.docker_image = docker_image
        self.packages = packages
        self.task_type = task_type


    def script(self):
        return NotImplementedError
    

    def set_arguments(self, task):
        return NotImplementedError


    def execute_remotely(self, queue_name):
        task = Task.create(
            project_name=self.project_name, 
            task_name=self.task_name, 
            task_type=self.task_type, 
            docker=self.docker_image, 
            packages=self.packages, 
            add_task_init_call=True,
            script=self.script,
            repo="https://github.com/neuralmagic/research.git",
            branch="alex-development",
        )

        self.set_arguments(task)
        
        task.execute_remotely(queue_name=queue_name, clone=False, exit_process=True)
        #self.script(task)

    # def execute_locally(self):
    #         task = Task.create(
    #         project_name=self.project_name, 
    #         task_name=self.task_name, 
    #         task_type=self.task_type, 
    #         docker=self.docker_image, 
    #         packages=self.packages, 
    #         add_task_init_call=True,
    #         script=self.script,
    #         repo="https://github.com/neuralmagic/research.git",
    #         branch="alex-development",
    #     )

    #     self.set_arguments(task)
        
