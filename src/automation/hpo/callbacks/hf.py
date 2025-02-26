from clearml import Task
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_to_hf(task: Union[Task, str]):
    if isinstance(task, str):
        task = Task.get_task(task_id=task)
    
    model = task.get_models()["output"][0]
    path = model.get_local_copy()
    hf_model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf_model.push_to_hub(f"nm-research-models/{model.name}_{model.id}", private=True)
    tokenizer.push_to_hub(f"nm-research-models/{model.name}_{model.id}", private=True)
