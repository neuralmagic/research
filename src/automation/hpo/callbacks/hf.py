from clearml import Task
from typing import Union, Optional

def push_to_hf(task: Union[Task, str], model_name: Optional[str]=None):
    from huggingface_hub import HfApi

    if isinstance(task, str):
        task = Task.get_task(task_id=task)
    
    model = task.get_models()["output"][0]
    path = model.get_local_copy()

    if model_name is None:
        model_name = f"{model.name}"

    model_name = model_name.replace("/", "__")

    api = HfApi()
    api.upload_large_folder(
        folder_path=path,
        repo_id=f"nm-research-models/{model_name}",
        repo_type="model"
    )