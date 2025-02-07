import argparse
from clearml import InputModel, Task

def dict_to_argparse(data: dict) -> argparse.Namespace:
    """Converts a dictionary to an argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in data.items():
        setattr(namespace, key, value)
    return namespace

def resolve_model_id(model_id: str, clearml_model: bool, task: Task) -> str:
    if clearml_model:
        input_model = InputModel(model_id=model_id)
        task.connect(input_model)
        return input_model.get_local_copy()
    else:
        return model_id