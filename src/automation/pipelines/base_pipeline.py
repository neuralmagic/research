from clearml import Task
from automation.configs import DEFAULT_DOCKER_IMAGE
from automation.tasks import BaseTask
from typing import Optional

class BasePipeline(BaseTask):

    def __init__(self,
        project_name: str,
        pipeline_name: str,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        version: str="1.0.0", 
    ):
        super().__init__(
            project_name=project_name,
            task_name=pipeline_name,
            docker_image=docker_image,
            task_type=Task.TaskTypes.controller,
        )
        
        self.version = version
        self.script_path = os.path.join(".", "src", "automation", "pipelines", "pipeline_script.py")
        self.steps = []
        self.parameters = []


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


    def get_arguments(self):
        parameters_dict = {}
        for parameter_args, parameter_kwargs in self.parameters:
            if len(parameter_args) > 0:
                name = parameter_args[0]
            else:
                name = parameter_kwargs["name"]

            if len(parameter_args) > 1:
                default = parameter_args[1]
            elif "default" in parameter_kwargs:
                default = parameter_kwargs["default"]
            else:
                default = None
                
            parameters_dict[name] = {
                "args": parameter_args[1:] if len(parameter_args) > 1 else None,
                **parameter_kwargs,
            }

        steps_dict = {}
        for step_args, step_kwargs in self.steps:
            if len(step_args) > 0:
                name = step_args[0]
            else:
                name = step_kwargs["name"]
            
            steps_dict[name] = {
                "args": step_args[1:] if len(step_args) > 1 else None,
                **step_kwargs
            }

        return {"Args": parameters_dict, "Steps": steps_dict, "pipeline": {"version": self.version}}

    
    def start(self, queue_name: str="services"):
        self.execute_remotely(queue_name)


    def start_locally(self):
        self.execute_locally()

