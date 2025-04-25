from clearml import Task
from automation.configs import DEFAULT_DOCKER_IMAGE
from automation.tasks import BaseTask
from automation.utils import serialize_callable
import os
from typing import Optional, Callable

class BasePipeline(BaseTask):

    def __init__(self,
        project_name: str,
        pipeline_name: str,
        version: str="1.0.0", 
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        job_end_callback: Optional[Callable]=None,
    ):
        super().__init__(
            project_name=project_name,
            task_name=pipeline_name,
            docker_image=docker_image,
            task_type=Task.TaskTypes.controller,
        )
        
        self.pipeline_name = pipeline_name
        self.version = version
        self.script_path = os.path.join(".", "src", "automation", "pipelines", "pipeline_script.py")
        self.steps = []
        self.parameters = []
        self.job_end_callback = job_end_callback


    def script(self):
        from automation.pipelines.pipeline_script import main
        main()


    def add_step(self, *args, **kwargs,) -> None:
        assert len(args) > 0 or "name" in kwargs
        self.steps.append((args, kwargs))


    def add_parameter(self, *args, **kwargs,) -> None:
        assert len(args) > 0 or "name" in kwargs
        self.parameters.append((args, kwargs))


    def create_pipeline(self) -> None:
        self.create_task()


    def get_configurations(self) -> None:
        configs = {"Steps": self.steps}
        if self.job_end_callback is not None:
            configs["job end callback"] = serialize_callable(self.job_end_callback)
        return configs


    def get_arguments(self):
        args = {"pipeline": {"version": self.version}}
        if len(self.parameters) > 0:
            parameters_dict = {}
            for parameter_args, parameter_kwargs in self.parameters:
                if len(parameter_args) > 0:
                    name = parameter_args[0]
                else:
                    name = parameter_kwargs.pop("name")

                if len(parameter_args) > 1:
                    default = parameter_args[1]
                else:
                    default = parameter_kwargs.pop("default")

                parameters_dict[name] = default
            
            args["Args"] = parameters_dict

        return args


    def execute_remotely(self, queue_name: str="services"):
        super().execute_remotely(queue_name)


    def start(self, queue_name: str="services"):
        self.execute_remotely(queue_name)


    def start_locally(self):
        self.execute_locally()

