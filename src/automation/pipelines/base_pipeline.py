from clearml import PipelineController
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional

class BasePipeline():

    packages = ["git+https://github.com/neuralmagic/research.git@alex-development"]

    def __init__(self,
        project_name: str,
        pipeline_name: str,
        version: Optional[str]=None,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
    ):
        self.project_name = project_name
        self.pipeline_name = pipeline_name
        self.version = version
        self.pipeline = None
        self.docker_image = docker_image
        self.steps = []
        self.parameters = []
    

    def add_step(self, *args, **kwargs,) -> None:
        self.steps.append((args, kwargs))


    def add_parameter(self, *args, **kwargs,) -> None:
        self.parameters.append((args, kwargs))


    def _create(self) -> None:
        self.pipeline = PipelineController(
            project=self.project_name,
            name=self.pipeline_name,
            version=self.version,
            target_project=self.project_name,
            packages=self.packages,
            docker=self.docker_image,
        )

        for parameter_args, parameter_kwargs in self.parameters:
            self.pipeline.add_parameter(*parameter_args, **parameter_kwargs)

        for step_args, step_kwargs in self.steps:
            self.pipeline.add_step(*step_args, **step_kwargs)


    def create_pipeline(self) -> None:
        self._create()
        self.pipeline.start(None)


    def execute_remotely(self, *args, **kwargs) -> None:
        if self.pipeline is not None:
            raise Exception("Can only execute locally if pipeline is not yet created.")
        
        self._create()
        self.pipeline.start(*args, **kwargs)

        #Note: this is a temporary fix because ClearML 1.14 does not support creating a
        # pipeline separetaly from starting it.
        # This is fixed in ClearML 1.17 and this code can be updated to use
        # PipelineController.create() once we upgrade to ClearML 1.17.

    def execute_locally(self) -> None:
        if self.pipeline is not None:
            raise Exception("Can only execute locally if pipeline is not yet created.")
        
        self._create()
        self.pipeline.start_locally() 
