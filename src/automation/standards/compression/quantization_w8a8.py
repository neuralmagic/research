from automation.tasks import LLMCompressorTask
from automation.configs import DEFAULT_DOCKER_IMAGE
from typing import List, Optional, Sequence, Union
import yaml

LLAMA_MAPPINGS = [
    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    [["re:.*down_proj"], "re:.*up_proj"],
]

class QuantizationW8A8Task(LLMCompressorTask):
    def __init__(
        self,
        project_name: str,
        task_name: str,
        model_id: str,
        damping_frac: float=0.01,
        observer: str="mse",
        smoothing_strength: float=0.8,
        smoothquant_mappings: dict=LLAMA_MAPPINGS,
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
                    "SmoothQuantModifier": {
                        "smoothing_strength": "$smoothing_strength",
                        "mappings": smoothquant_mappings,
                    },
                    "GPTQModifier": {
                        "ignore": ["lm_head"],
                        "damping_frac": "$damping_frac",
                        "config_groups": {
                            "group_0": {
                                "targets": ["Linear"],
                                "weights": {
                                    "num_bits": 8,
                                    "type": "int",
                                    "symmetric": True,
                                    "strategy": "channel",
                                    "observer": "$observer",
                                },
                                "input_activations": {
                                    "num_bits": 8,
                                    "type": "int",
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
            "damping_frac": damping_frac,
            "observer": observer,
            "smoothing_strength": smoothing_strength,
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