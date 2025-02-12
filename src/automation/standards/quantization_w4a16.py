from automation.tasks import LLMCompressorTask
from automation.docker import DEFAULT_DOCKER_IMAGE
from typing import List, Optional, Sequence, Union

class QuantizationW4A16Task(LLMCompressorTask):
    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        damping_frac: float,
        observer: str="mse",
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
                        "damping_frac": damping_frac,
                        "config_groups": {
                            "group_0": {
                                "weights": {
                                    "num_bits": 4,
                                    "type": "int",
                                    "symmetric": True,
                                    "strategy": "group",
                                    "group_size": 128,
                                    "actorder": "group",
                                    "observer": observer,
                                },
                                "targets": ["Linear"],
                            },
                        },
                    },
                },
            },
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
        )