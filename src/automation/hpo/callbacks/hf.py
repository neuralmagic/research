from clearml import Task
from typing import Union

def push_to_hf(task: Union[Task, str]):
    from huggingface_hub import HfApi

    if isinstance(task, str):
        task = Task.get_task(task_id=task)
    
    model = task.get_models()["output"][0]
    path = model.get_local_copy()

    api = HfApi()
    api.upload_large_folder(
        folder_path=path,
        repo_id=f"nm-research-models/{model.name}_{model.id}",
        repo_type="model"
    )