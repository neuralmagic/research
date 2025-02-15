from clearml import Task
from typing import Sequence, Optional
from automation.configs import DEFAULT_OUTPUT_URI

class BaseTask():

    base_packages = ["git+https://github.com/neuralmagic/research.git@alex-development"]

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
        self.task = None
        self.script_path = None
  

    @property
    def id(self):
        return self.get_task_id()

    @property
    def name(self):
        return self.task_name


    def get_arguments(self):
        return NotImplementedError


    def set_arguments(self):
        args = self.get_arguments()
        for args_name, args_dict in args.items():
            self.task.connect(args_dict, args_name)


    def script(self):
        raise NotImplementedError


    def create_task(self):
        self.task = Task.create(
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
        self.task.output_uri = DEFAULT_OUTPUT_URI
        self.set_arguments()


    def get_task_id(self):
        if self.task is not None:
            return self.task.id
        else:
            raise ValueError("Task ID not available since ClearML task not yet created. Try task.create_task() firts.")


    def execute_remotely(self, queue_name):
        if self.task is None:
            self.create_task()
        self.task.execute_remotely(queue_name=queue_name, clone=False, exit_process=True)


    def execute_locally(self):
        if self.task is not None:
            raise Exception("Can only execute locally if task is not yet created.")

        self.task = Task.init(
            project_name=self.project_name, 
            task_name=self.task_name, 
            task_type=self.task_type,
            auto_connect_arg_parser=False,
        )
        self.set_arguments()        
        self.script()
        self.task.close()

