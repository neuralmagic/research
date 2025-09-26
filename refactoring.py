import os
from pydantic import BaseModel
from typing import Optional
import config
import data
import yaml

v1filepath = "arena-hard-v2.0.yaml"
v2filepath = "arena-hard-v0.1.yaml"

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
    print(f"The filepath is: {filepath}")
    #print(config.yamls)
    print(config.directories)
    with open(os.path.join(config.directories["arenahard_config_path"], f"{filepath}.tmp"), 'w') as yaml_file:
        yaml.dump(new_config.model_dump(), yaml_file, sort_keys = False)
    return new_config

v1_dict = {"max_tokens": 130, "judge_model": "okv1", "model_list": ["why"]}
v2_dict = {"max_tokens": 160, "judge_model": "okv2"}

v1_config= get_config(v1filepath, v1_dict)
v2_config= get_config(v2filepath, v2_dict)

"""
print(v1_config)
print(v2_config)
"""
#print(config.yamls)
#print(config.directories)
#print(data.jsonls)
#print(data.directories)
