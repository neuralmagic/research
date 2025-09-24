from typing import Sequence, Optional
from automation.configs import DEFAULT_OUTPUT_URI, DEFAULT_RESEARCH_BRANCH
from automation.standards import STANDARD_CONFIGS
import yaml
import os

try:
    from clearml import Task
    clearml_available = True
except ImportError:
    print("ClearML not available. Will run tasks locally and not report to ClearML.")
    clearml_available = False

class BaseTask():

    def __init__(
        self,
        project_name: str,
        task_name: str,
        docker_image: str,
        branch: Optional[str] = DEFAULT_RESEARCH_BRANCH,
        packages: Optional[Sequence[str]]=None,
        task_type: str="training",
    ):
        branch_name = branch or DEFAULT_RESEARCH_BRANCH
        base_packages = [f"git+https://github.com/neuralmagic/research.git@{branch_name}"]
        
        if packages is not None:
            packages = list(set(packages + base_packages))
        else:
            packages = base_packages

        # keep only the pinned version of a library
        for pkg in packages:
            if "==" in pkg and pkg.split("==")[0] in packages:
                lib_name = pkg.split("==")[0]
                packages.remove(lib_name)

        self.project_name = project_name
        self.task_name = task_name
        self.docker_image = docker_image
        self.packages = packages
        self.task_type = task_type
        self.task = None
        self.branch = branch
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
        elif os.path.exists(os.path.join("..", "standards", config)):
            return yaml.safe_load(open(os.path.join("..", "standards", config)), "r")
        else:
            return yaml.safe_load(config)


    def get_arguments(self):
        return {}


    def set_arguments(self):
        args = self.get_arguments()
        if clearml_available:
            for args_name, args_dict in args.items():
                self.task.connect(args_dict, args_name)
        
        return args


    def get_configurations(self):
        return {}


    def set_configurations(self):
        configurations = self.get_configurations()
        if clearml_available:
            for name, config in configurations.items():
                self.task.connect_configuration(config, name=name)
        
        return configurations


    def script(self, configurations, args):
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
            branch=self.branch,
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
        if clearml_available:
            if self.task is not None:
                raise Exception("Can only execute locally if task is not yet created.")

            self.task = Task.init(
                project_name=self.project_name, 
                task_name=self.task_name, 
                task_type=self.task_type,
                auto_connect_arg_parser=False,
            )
            args = self.set_arguments()
            configurations = self.set_configurations()
            self.script(configurations, args)
            self.task.close()
        else:
            args = self.set_arguments()
            configurations = self.set_configurations()
            self.script(configurations, args)