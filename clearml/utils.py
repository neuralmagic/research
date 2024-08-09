from collections.abc import Iterable, Mapping
from clearml import Task

def open_clearml_task(project_name, task_name):
    clearml_task = Task.get_task(project_name=project_name, task_name=task_name)
    if clearml_task is None:
        clearml_task = Task.init(project_name=project_name, task_name=task_name)
    else:
        clearml_task.started()

    return clearml_task

def push_artifacts(clearml_task, artifacts):
    if not isinstance(artifacts, Iterable) or isinstance(artifacts, Mapping):
        artifacts = [artifacts]
    
    for artifact in artifacts:
        clearml_task.upload_artifact(name=artifact["name"], artifact_object=artifact["object"])

def push_scalars(clearml_task, scalars):
    if not isinstance(scalars, Iterable) or isinstance(scalars, Mapping):
        scalars = [scalars]
    for scalar in scalars:
        clearml_task.get_logger().report_single_value(name=scalar["name"], value=scalar["value"])

def push_to_clearml(clearml_task, scalars, artifacts):
    if scalars is not None:
        push_scalars(clearml_task, scalars)
    if artifacts is not None:
        push_artifacts(clearml_task, artifacts)

    