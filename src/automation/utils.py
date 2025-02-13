import argparse
from clearml import InputModel, Task
import inspect
import typing


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
    
def cast_args(data: dict[str, str], func: callable) -> dict:
    """
    Converts dictionary values to match the expected argument types of a given callable.
    
    :param data: Dictionary with string values.
    :param func: Callable whose argument types should be matched.
    :return: New dictionary with converted values.
    """
    sig = inspect.signature(func)
    converted_data = {}
    
    def convert_value(value: str, expected_type):
        if expected_type is inspect.Parameter.empty:
            return value  # Keep as string if no type hint
        
        # Handle Optional and Union types
        origin = typing.get_origin(expected_type)
        args = typing.get_args(expected_type)
        
        if origin is typing.Union and len(args) == 2 and type(None) in args:
            non_none_type = next(t for t in args if t is not type(None))
            try:
                return non_none_type(value)
            except (ValueError, TypeError):
                return value
        
        try:
            return expected_type(value)
        except (ValueError, TypeError):
            return value
    
    for key, value in data.items():
        if key in sig.parameters:
            param = sig.parameters[key]
            converted_data[key] = convert_value(value, param.annotation)
        else:
            converted_data[key] = value  # Keep as string if not in function signature
    
    return converted_data

