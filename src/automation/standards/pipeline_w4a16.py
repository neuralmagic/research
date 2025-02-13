from automation.pipelines import Pipeline
from automation.standards import QuantizationW4A16Task, OpenLLMTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from typing import List

class QuantizationW4A16Pipeline(Pipeline):
    def __init__(self,
        project_name: str,
        pipeline_name: str,
        model_id: str,
        execution_queues: List[str],
        version=None,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
    ):
        super().__init__(
            project_name, 
            pipeline_name, 
            version, 
            docker_image,
        )
        
        self.model_id = model_id
        self.execution_queues = execution_queues

        self.add_parameter("damping_frac", "float", default=0.01)
        self.add_parameter("observer", "str", default="mse")
        self.add_parameter("group_size", "int", default=128)
        self.add_parameter("actorder", "str", default="weight")

        self.add_quantization_step()
        self.add_evaluation_step()
        
    
    def add_quantization_step(self):
        parameter_override = {
            "Args/damping_frac": "${pipeline.damping_frac}",
            "Args/observer": "${pipeline.observer}",
            "Args/group_size": "${pipeline.group_size}",
            "Args/actorder": "${pipeline.actorder}",
        }
        step1 = QuantizationW4A16Task(
            project_name=self.project_name,
            task_name=self.pipeline_name + "_quantization_draft",
            model_id=self.model_id,
            parameter_override=parameter_override,
        )
        step1.create_task()
        self.add_step(
            name=self.pipeline_name + "_quantization",
            base_task_id=step1.id,
            execution_queue=self.execution_queues[0],
            monitor_models=[step1.get_arguments()["Args"]["save_directory"]],
            monitor_artifacts=["recipe"],
        )


    def add_evaluation_step(self):

        step1_model_id = "${pipeline_example_quantization.models.output.-1.id}"

        step2 = OpenLLMTask(
            project_name=self.project_name,
            task_name=self.pipeline_name + "evaluation_draft",
            model_id="dummy",
            clearml_model=True,
            parameter_override={"Args/model_id": step1_model_id},
        )

        self.add_step(
            name=self.pipeline_name + "_evaluation",
            base_task_id = step2.id,
            parents=[self.pipeline_name + "_quantization"],
            execution_queue=self.execution_queues[1],
            monitor_metrics=[("Summary", "openllm")],
        )
        