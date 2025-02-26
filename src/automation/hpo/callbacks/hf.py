from clearml import Task
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_to_hf(job_id, *args, **kwargs):
    task = Task.get_task(task_id=job_id)
    model = task.get_models()["output"][0]
    path = model.get_local_copy()
    hf_model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf_model.push_to_hub(f"nm-research-models/{model.name}_{model.id}", private=True)
    tokenizer.push_to_hub(f"nm-research-models/{model.name}_{model.id}", private=True)
