from automation.tasks.base_task import BaseTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import Union, List, Optional, Sequence, Any, Callable
import os
import yaml

class SemanticSimilarityGenerateTask(BaseTask):
    task_packages = [
        "vllm==0.10.1.1",
        "hf_xet",
        "pyzmq",
        #"datasets",
        #"rouge_score",
        #"bert-score",
        #"sentence-transformers",
        #"vllm==0.10.1.1",
        #"datasets==4.2.0",
        #"rouge_score==0.1.2",
        #"bert-score==0.3.13",
        #"sentence-transformers==5.1.1",
        #"pyzmq==27.1.0",
    ]

    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        branch: str,
        max_new_tokens: int,
        max_model_len: int,
        num_samples_per_dataset: Optional[int],
        dataset_args: Optional[dict]=None,
        semantic_similarity_args: Optional[dict]=None,
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


        if dataset_args is None:
            self.dataset_args = config_kwargs.pop("dataset_args", None)
        else:
            config_dataset_args = config_kwargs.pop("dataset_args", {})
            config_dataset_args.update(dataset_args)
            self.dataset_args = config_dataset_args

        if semantic_similarity_args is None:
            self.semantic_similarity_args = config_kwargs.pop("semantic_similarity_args", None)
        else:
            config_semantic_similarity_args = config_kwargs.pop("semantic_similarity_args", {})
            config_semantic_similarity_args.update(semantic_similarity_args)
            self.semantic_similarity_args = config_semantic_similarity_args

        self.num_samples_per_dataset = config_kwargs.pop("num_samples_per_dataset", num_samples_per_dataset)
        self.max_new_tokens = config_kwargs.pop("max_new_tokens", max_new_tokens)
        self.max_model_len = config_kwargs.pop("max_model_len", max_model_len)
        self.trust_remote_code = config_kwargs.pop("trust_remote_code", trust_remote_code)

        if tags is not None:
            tags = list(set(config_kwargs.pop("tags", []).extend(tags)))
        else:
            tags = config_kwargs.pop("tags", None)
        self.tags = tags

        # Store class attributes
        self.model_id = model_id
        self.clearml_model = clearml_model
        self.force_download = force_download
        self.save_directory = save_directory
        self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "semantic_similarity_generate_script.py")


    def script(self, configurations, args):
        from automation.tasks.scripts.semantic_similarity_generate_script import main
        main(configurations, args)
        

    def get_configurations(self):
        configs = {}
        return configs


    def get_arguments(self):
        return {
            "Args": {
                "model_id": self.model_id,
                "dataset_args": self.dataset_args,
                "semantic_similarity_args": self.semantic_similarity_args,
                "clearml_model": self.clearml_model,
                "force_download": self.force_download,
                "save_directory": self.save_directory,
                "num_samples_per_dataset": self.num_samples_per_dataset,
                "max_new_tokens": self.max_new_tokens,
                "max_model_len": self.max_model_len,
                "trust_remote_code": self.trust_remote_code,
                "tags": self.tags,
            },
        }


