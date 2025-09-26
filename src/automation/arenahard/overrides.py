import os
from pydantic import BaseModel
from typing import Optional
import config
import data
import yaml

class ArenahardConfig(BaseModel):
    judge_model: str
    temperature: float
    max_tokens: int
    bench_name: str 
    reference: Optional[str] = None
    regex_patterns: list 
    prompt_template: str
    model_list: list 

def get_config(filepath, override_dict): 
    raw_data = config.yamls[filepath]
    arenahard_config = ArenahardConfig.model_validate(raw_data)
    new_config = arenahard_config.model_copy(update=override_dict)
    #print(f"The filepath is: {filepath}")
    #print(config.yamls)
    #print(config.directories)
    with open(os.path.join(config.directories["arenahard_config_path"], f"{filepath}.tmp"), 'w') as yaml_file:
        yaml.dump(new_config.model_dump(), yaml_file, sort_keys = False)
    return new_config

