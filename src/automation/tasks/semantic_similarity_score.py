from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Union, List, Optional, Sequence, Any, Callable
import os
import yaml

class SemanticSimilarityScoreTask(BaseTask):
    task_packages = [
        "hf_xet",
        "pyzmq",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        reference_model_project_name: str,
        candidate_model_project_name: str,
        reference_model_task_name: str,
        candidate_model_task_name: str,
        sts_model_id: str,
        branch: str,
        rouge_scores: Optional[len]=None,
        scoring_args: Optional[dict]=None,
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        force_download: bool=False,
        save_directory: str="output",
        trust_remote_code: bool=False,
        tags: Union[str, List[str]]=None,
        task_type: str="training",
        config: Optional[str]=None,
    ):

        # Process config
        config_kwargs = self.process_config(config)

        # Set packages, taking into account default packages
        # for the LMEvalTask and packages set in the config
        if packages is not None:
            packages = list(set(packages + self.task_packages))
        else:
            packages = self.task_packages

        if "packages" in config_kwargs:
            packages = list(set(packages + config_kwargs.pop("packages")))

        # Initialize base parameters
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            branch=branch,
            docker_image=docker_image,
            packages=packages,
            task_type=task_type,
        )


        if rouge_scores is None:
            self.rouge_scores = config_kwargs.pop("rouge_scores", None)
        else:
            config_rouge_scores = config_kwargs.pop("rouge_scores", {})
            config_rouge_scores.update(rouge_scores)
            self.rouge_scores = config_rouge_scores

        if scoring_args is None:
            self.scoring_args = config_kwargs.pop("scoring_args", None)
        else:
            config_scoring_args = config_kwargs.pop("scoring_args", {})
            config_scoring_args.update(scoring_args)
            self.scoring_args = config_scoring_args

        self.trust_remote_code = config_kwargs.pop("trust_remote_code", trust_remote_code)

        if tags is not None:
            tags = list(set(config_kwargs.pop("tags", []).extend(tags)))
        else:
            tags = config_kwargs.pop("tags", None)
        self.tags = tags

        # Store class attributes
        self.reference_model_project_name = reference_model_project_name
        self.candidate_model_project_name = candidate_model_project_name
        self.reference_model_task_name = reference_model_task_name
        self.candidate_model_task_name = candidate_model_task_name
        self.sts_model_id = sts_model_id
        self.clearml_model = clearml_model
        self.force_download = force_download
        self.save_directory = save_directory
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "semantic_similarity_score_script.py")


    def script(self, configurations, args):
        from automation.tasks.scripts.semantic_similarity_score_script import main
        main(configurations, args)
        

    def get_configurations(self):
        configs = {}
        return configs


    def get_arguments(self):
        return {
            "Args": {
                "reference_model_project_name": self.reference_model_project_name,
                "candidate_model_project_name": self.candidate_model_project_name,
                "reference_model_task_name": self.reference_model_task_name,
                "candidate_model_task_name": self.candidate_model_task_name,
                "sts_model_id": self.sts_model_id,
                "rouge_scores": self.rouge_scores,
                "scoring_args": self.scoring_args,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
                "save_directory": self.save_directory,
                "trust_remote_code": self.trust_remote_code,
                "tags": self.tags,
            },
        }


