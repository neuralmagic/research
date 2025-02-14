from automation.tasks import LMEvalTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence
import os


class OpenLLMTask(LMEvalTask):
    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        **kwargs,
    ):
        
        if "tasks" in kwargs:
            raise ValueError("taks should not be specified with OpenLLMTask")
        
        if "model_args" not in kwargs:
            model_args = "dtype=auto,max_model_len=4096,enable_chunked_prefill=True"
        else:
            model_args = kwargs.pop("model_args")
            if "max_model_len" in model_args:
                raise ValueError("max_model_len should not be specified with OpenLLMTask")
            else:
                model_args += ",max_model_len=4096"

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            model_id=model_id,
            docker_image=docker_image,
            packages=packages,
            clearml_model=clearml_model,
            tasks="openllm",
            model_args=model_args,
            **kwargs,
        )

        self.script_path = os.path.join(".", "src", "automation", "standards", "scripts", "openllm_script.py")


    def script(self):
        from automation.standards.scripts.openllm_script import main
        main()
