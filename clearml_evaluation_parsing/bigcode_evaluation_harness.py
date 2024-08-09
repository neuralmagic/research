import json
from clearml import Task
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('json_file', type=str, help='Path to model to push')
parser.add_argument('clearml_project', type=str, help='Name of ClearML project')
parser.add_argument('clearml_task', type=str, nargs='?', help='Optional name of clearml task')

args = parser.parse_args()

results = json.load(open(args.json_file))

clearml_task = Task.get_task(project_name=args.clearml_project, task_name=args.clearml_task)
if clearml_task is None:
    clearml_task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
else:
    clearml_task.started()

clearml_task.upload_artifact(name=args.json_file, artifact_object=results)

for task in results:
    if task == "config":
        continue
    
    for metric in results[task]:
        value = results[task][metric]
        if not isinstance(value, str):
            name = task + "/" + metric
            clearml_task.get_logger().report_single_value(name=name, value=value)

clearml_task.mark_completed()
