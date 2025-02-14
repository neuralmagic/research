from automation.tasks import LLMCompressorTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import List, Optional, Sequence, Union
import yaml

class QuantizationFP8DynamicTask(LLMCompressorTask):
    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        observer: str="mse",
        docker_image: str=DEFAULT_DOCKER_IMAGE,
        packages: Optional[Sequence[str]]=None,
        clearml_model: bool=False,
        save_directory: str="output",
        trust_remote_code: bool=False,
        tags: Union[str, List[str]]=None,
    ):
        
        recipe = {
            "quant_stage": {
                "quant_modifiers": {
                    "QuantizationModifier": {
                        "ignore": ["lm_head"],
                        "config_groups": {
                            "group_0": {
                                "targets": ["Linear"],
                                "weights": {
                                    "num_bits": 8,
                                    "type": "float",
                                    "symmetric": True,
                                    "strategy": "channel",
                                    "observer": "$observer",
                                },
                                "input_activations": {
                                    "num_bits": 8,
                                    "type": "float",
                                    "symmetric": True,
                                    "strategy": "token",
                                    "dynamic": True,
                                    "observer": "memoryless",
                                },
                            },
                        },
                    },
                },
            },
        }

        recipe = yaml.dump(recipe, default_flow_style=False)

        recipe_args = {
            "observer": observer,
        }
        
        super().__init__(
            project_name=project_name,
            task_name=task_name,
            model_id=model_id,
            docker_image=docker_image,
            packages=packages,
            dataset_name=None,
            clearml_model=clearml_model,
            save_directory=save_directory,
            trust_remote_code=trust_remote_code,
            max_memory_per_gpu="auto",
            tags=tags,
            recipe=recipe,
            recipe_args=recipe_args,
        )