import argparse
from clearml import InputModel, Task
import inspect
import typing
import psutil


def parse_argument(argument, argument_type):
    if argument is None:
        return None
    elif isinstance(argument, str) and argument.lower() == "none":
        return None
    elif isinstance(argument, argument_type):
        return argument
    elif isinstance(argument, str):
        if argument_type is bool:
            return argument.lower() == "true"
        else:
            return argument_type(argument)
    else:
        return argument_type(argument)


def dict_recursive_update(d: dict, u: dict) -> dict:
    """Recursively updates a dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = dict_recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dict_to_argparse(data: dict) -> argparse.Namespace:
    """Converts a dictionary to an argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in data.items():
        setattr(namespace, key, value)
    return namespace


def resolve_model_id(
    model_id: str, 
    clearml_model: bool, 
    force_download: bool=False, 
    model_class="AutoModelForCausalLM",
) -> str:
    
    if clearml_model:
        task = Task.current_task()
        input_model = InputModel(model_id=model_id)
        task.connect(input_model)
        return input_model.get_local_copy(force_download=force_download)
    else:
        if force_download:
            import transformers
            model_class = getattr(transformers, model_class)
            model_class.from_pretrained(model_id, force_download=True,trust_remote_code=True)
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
        
        if origin is typing.Union:
            valid_types = [t for t in args if t is not type(None)]
            for valid_type in valid_types:
                try:
                    if valid_type is bool and isinstance(value, str):
                        return value.lower() == "true"
                    return valid_type(value)
                except (ValueError, TypeError):
                    continue
            return value  # Fallback if no conversion succeeded
        
        # Handle boolean conversion properly
        if expected_type is bool and isinstance(value, str):
            return value.lower() == "true"
        
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


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()  # or child.kill()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

