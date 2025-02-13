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
    
import inspect


def cast_args(data: dict[str, str], func: callable) -> dict:
    """
    Converts dictionary values to match the expected argument types of a given callable.
    
    :param data: Dictionary with string values.
    :param func: Callable whose argument types should be matched.
    :return: New dictionary with converted values.
    """
    sig = inspect.signature(func)
    converted_data = {}
    
    for key, value in data.items():
        if key in sig.parameters:
            param = sig.parameters[key]
            expected_type = param.annotation
            
            if expected_type is not inspect.Parameter.empty:
                try:
                    converted_data[key] = expected_type(value)
                except (ValueError, TypeError):
                    converted_data[key] = value  # Fallback to original string if conversion fails
            else:
                converted_data[key] = value  # Keep as string if no type hint
        else:
            converted_data[key] = value  # Keep as string if not in function signature
    
    return converted_data
