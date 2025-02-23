from automation.tasks import LMEvalTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Optional, Sequence
import os


class LeaderboardTask(LMEvalTask):

    leaderboard_packages = ["langdetect", "immutabledict", "antlr4-python3-runtime==4.11"]

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
        
        if packages is not None:
            packages = list(set(packages + self.leaderboard_packages))
        else:
            packages = self.leaderboard_packages  
        
        if "tasks" in kwargs:
            raise ValueError("taks should not be specified with LeaderboardTask")
        
        if "apply_chat_template" in kwargs:
            raise ValueError("apply_chat_template should not be specified with LeaderboardTask")

        if "fewshot_as_multiturn" in kwargs:
            raise ValueError("fewshot_as_multiturn should not be specified with LeaderboardTask")

        if "model_args" not in kwargs:
            model_args = "dtype=auto,enable_chunked_prefill=True"
        else:
            model_args = kwargs.pop("model_args")

        super().__init__(
            project_name=project_name,
            task_name=task_name,
            model_id=model_id,
            docker_image=docker_image,
            packages=packages,
            clearml_model=clearml_model,
            tasks="leaderboard",
            model_args=model_args,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            **kwargs,
        )

        self.script_path = os.path.join(".", "src", "automation", "standards", "scripts", "leaderboard_script.py")


    def script(self):
        from automation.standards.scripts.leaderboard_script import main
        main()
