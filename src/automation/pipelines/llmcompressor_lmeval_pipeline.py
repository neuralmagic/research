from automation.pipelines import Pipeline
from automation.tasks import LLMCompressorTask, LMEvalTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import List, Optional

class LLMCompressorLMEvalPipeline(Pipeline):
    def __init__(
        self,
        project_name: str,
        pipeline_name: str,
        model_id: str,
        execution_queues: List[str],
        version: str="1.0.0",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        parameters: Optional[dict]=None,
        llmcompressor_kwargs: dict={},
        lmeval_kwargs: dict={},
        config: Optional[str]=None,
    ):
        super().__init__(
            project_name, 
            pipeline_name, 
            version, 
            docker_image,
        )
        
        # Process config
        config_kwargs = self.process_config(config)
        if "parameters" in config_kwargs:
            if parameters is not None:
                raise ValueError("parameters specified in config. Can't specify again in pipeline instantiation.")
            else:
                parameters = config_kwargs.pop("parameters")

        self.model_id = model_id
        self.execution_queues = execution_queues
        self.llmcompressor_kwargs = config_kwargs.pop("llmcompressor_kwargs", {}).update(llmcompressor_kwargs)
        self.lmeval_kwargs = config_kwargs.pop("lmeval_kwargs", {}).update(lmeval_kwargs)
        self._parameters = parameters

        self.add_llmcompressor_step()
        self.add_lmeval_steps()
        
    
    def add_llmcompressor_step(self):
        parameter_override = {}
        for parameter_name, parameter_kwargs in self._parameters.items():
            recipe_arg = parameter_kwargs.pop("recipe_arg", False)
            self.add_parameter(parameter_name, **parameter_kwargs)

            if recipe_arg:
                parameter_override[f"Args/recipe_args/{parameter_name}"] = f"${{pipeline.{parameter_name}}}"

        self.step1_name = self.pipeline_name + "_" + self.llmcompressor_kwargs.pop("name", "llmcompressor")
        step1 = LLMCompressorTask(
            project_name=self.project_name,
            task_name=self.step1_name + "_draft",
            model_id=self.model_id,
            **self.llmcompressor_kwargs,
        )
        step1.create_task()
        self.add_step(
            name=self.step1_name,
            base_task_id=step1.id,
            execution_queue=self.execution_queues[0],
            parameter_override=parameter_override,
            monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
            monitor_artifacts=["recipe"],
        )


    def add_lmeval_steps(self):
        step1_model_id = f"${{{self.step1_name}.models.output.-1.id}}"
        
        queue_id = 0
        for step_name, step_kwargs in self.lmeval_kwargs.items():
            queue_id += 1
            step_name = self.pipeline_name + "_" + step_name
            monitor_metrics = [tuple(entry) for entry in step_kwargs.pop("monitor_metrics", [])]
            step = LMEvalTask(
                project_name=self.project_name,
                task_name=step_name + "_draft",
                model_id="dummy",
                clearml_model=True,
                **step_kwargs,
            )
            step.create_task()

            self.add_step(
                name=step_name,
                base_task_id = step.id,
                parents=[self.step1_name],
                execution_queue=self.execution_queues[queue_id],
                parameter_override={"Args/model_id": step1_model_id},
                monitor_metrics=monitor_metrics,
            )
        