from automation.tasks import LLMCompressorTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import List, Optional, Sequence, Union
import yaml

class QuantizationW4A16Task(LLMCompressorTask):
    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        dampening_frac: float=0.01,
        observer: str="mse",
        group_size: int=128,
        actorder: str="weight",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        dataset_name: str="calibration",
        clearml_model: bool=False,
        save_directory: str="output",
        num_samples: int=512,
        max_seq_len: int=8192,
        trust_remote_code: bool=False,
        max_memory_per_gpu: str="hessian",
        tags: Union[str, List[str]]=None,
    ):
        
        recipe = {
            "quant_stage": {
                "quant_modifiers": {
                    "GPTQModifier": {
                        "ignore": ["lm_head"],
                        "dampening_frac": "$dampening_frac",
                        "config_groups": {
                            "group_0": {
                                "weights": {
                                    "num_bits": 4,
                                    "type": "int",
                                    "symmetric": True,
                                    "strategy": "group",
                                    "group_size": "$group_size",
                                    "actorder": "$actorder",
                                    "observer": "$observer",
                                },
                                "targets": ["Linear"],
                            },
                        },
                    },
                },
            },
        }

        recipe = yaml.dump(recipe, default_flow_style=False)

        recipe_args = {
            "dampening_frac": dampening_frac,
            "observer": observer,
            "group_size": group_size,
            "actorder": actorder,
        }
        
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            model_id=model_id,
            docker_image=docker_image,
            packages=packages,
            dataset_name=dataset_name,
            clearml_model=clearml_model,
            save_directory=save_directory,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            trust_remote_code=trust_remote_code,
            max_memory_per_gpu=max_memory_per_gpu,
            tags=tags,
            recipe=recipe,
            recipe_args=recipe_args,
        )