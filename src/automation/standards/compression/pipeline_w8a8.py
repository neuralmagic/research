from automation.pipelines import Pipeline
from automation.standards import QuantizationW8A8Task, OpenLLMTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import List

class QuantizationW8A8Pipeline(Pipeline):
    def __init__(
        self,
        project_name: str,
        pipeline_name: str,
        model_id: str,
        execution_queues: List[str],
        version: str="1.0.0",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        dampening_frac: float=0.01,
        observer: str="mse",
        num_samples: int=512,
        smoothing_strength: float=0.8,
        llmcompressor_kwargs: dict={},
        openllm_kwargs: dict={},
    ):
        super().__init__(
            project_name, 
            pipeline_name, 
            version, 
            docker_image,
        )
        
        self.model_id = model_id
        self.execution_queues = execution_queues
        self.dampening_frac = dampening_frac
        self.observer = observer
        self.smoothing_strength = smoothing_strength
        self.num_samples = num_samples
        self.llmcompressor_kwargs = llmcompressor_kwargs
        self.openllm_kwargs = openllm_kwargs

        self.add_parameter("dampening_frac", default=self.dampening_frac, param_type="float")
        self.add_parameter("observer", default=self.observer, param_type="str")
        self.add_parameter("smoothing_strength", default=self.smoothing_strength, param_type="float")
        self.add_parameter("num_samples", default=self.num_samples, param_type="int")

        self.add_quantization_step()
        self.add_evaluation_step()
        
    
    def add_quantization_step(self):
        parameter_override = {
            "Args/recipe_args/dampening_frac": "${pipeline.dampening_frac}",
            "Args/recipe_args/observer": "${pipeline.observer}",
            "Args/recipe_args/smoothing_strength": "${pipeline.smoothing_strength}",
            "Args/num_samples": "${pipeline.num_samples}",
        }
        step1 = QuantizationW8A8Task(
            project_name=self.project_name,
            task_name=self.pipeline_name + "_quantization_draft",
            model_id=self.model_id,
            **self.llmcompressor_kwargs,
        )
        step1.create_task()
        self.add_step(
            name=self.pipeline_name + "_quantization",
            base_task_id=step1.id,
            execution_queue=self.execution_queues[0],
            parameter_override=parameter_override,
            monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
            monitor_artifacts=["recipe"],
        )


    def add_evaluation_step(self):
        step1_model_id = f"${{{self.pipeline_name}_quantization.models.output.-1.id}}"

        step2 = OpenLLMTask(
            project_name=self.project_name,
            task_name=self.pipeline_name + "_evaluation_draft",
            model_id="dummy",
            clearml_model=True,
            **self.openllm_kwargs,
        )
        step2.create_task()

        self.add_step(
            name=self.pipeline_name + "_evaluation",
            base_task_id = step2.id,
            parents=[self.pipeline_name + "_quantization"],
            execution_queue=self.execution_queues[1],
            parameter_override={"Args/model_id": step1_model_id},
            monitor_metrics=[("openllm", "average")],
        )
        