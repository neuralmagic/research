from clearml import Task
from typing import Sequence, Optional
from automation.configs import DEFAULT_OUTPUT_URI
from automation.standards import STANDARD_CONFIGS
import yaml
import os

class BaseTask():

    base_packages = ["git+https://github.com/neuralmagic/research.git"]

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
        self.callable_artifacts = None
  

    @property
    def id(self):
        return self.get_task_id()

    @property
    def name(self):
        return self.task_name

    
    def process_config(self, config):
        if config is None:
            return {}
            
        if config in STANDARD_CONFIGS:
            return yaml.safe_load(open(STANDARD_CONFIGS[config], "r"))
        elif os.path.exists(config):
            return yaml.safe_load(open(config, "r"))
        elif os.path.exists(os.path.join("..", "standatrds", config)):
            return yaml.safe_load(open(os.path.join("..", "standatrds", config)), "r")
        else:
            return yaml.safe_load(config)


    def get_arguments(self):
        return {}


    def set_arguments(self):
        args = self.get_arguments()
        for args_name, args_dict in args.items():
            self.task.connect(args_dict, args_name)


    def get_configurations(self):
        return {}


    def set_configurations(self):
        configurations = self.get_configurations()
        for name, config in configurations.items():
            self.task.connect_configuration(config, name=name)
        return configurations


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
            branch="shubhra/gemma_eval",
        )
        self.task.output_uri = DEFAULT_OUTPUT_URI
        self.set_arguments()
        self.set_configurations()


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
        configurations = self.set_configurations()
        self.script(configurations)
        self.task.close()

