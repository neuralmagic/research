import json
from clearml import Task
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('summary_json_file', type=str, help="Path to json file containing summary results")
parser.add_argument('detailed_json_file', type=str, help="Path to json file containing detailed results")
parser.add_argument('clearml_project', type=str, help='Name of ClearML project')
parser.add_argument('clearml_task', type=str, nargs='?', help='Optional name of clearml task')

args = parser.parse_args()

summary_results = json.load(open(args.summary_json_file))
detailed_results = json.load(open(args.detailed_json_file))

clearml_task = Task.get_task(project_name=args.clearml_project, task_name=args.clearml_task)
if clearml_task is None:
    clearml_task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
else:
    clearml_task.started()

clearml_task.upload_artifact(name=args.summary_json_file, artifact_object=summary_results)
clearml_task.upload_artifact(name=args.detailed_json_file, artifact_object=detailed_results)

for category in summary_results:
    value = summary_results[category]["0"]
    if not isinstance(value, str):
        category = category.replace("_", " ")
        clearml_task.get_logger().report_single_value(name=category.title(), value=value)

number_benchmarks = len(detailed_results["Category"])

categories = set([detailed_results["Category"][str(i)] for i in range(number_benchmarks)])

for category in categories:

    benchmark_name = []
    subtask = []
    number_few_shot = []
    accuracy = []

    for benchmark_index in range(number_benchmarks):
        benchmark_index = str(benchmark_index)
    
        if category != detailed_results["Category"][benchmark_index]:
            continue

        if detailed_results["Benchmark"][benchmark_index]:
            benchmark_name.append(detailed_results["Benchmark"][benchmark_index])
        else:
            benchmark_name.append(benchmark_name[-1])
    
        subtask.append(detailed_results["Subtask"][benchmark_index])

        number_few_shot.append(detailed_results["Number few shot"][benchmark_index])
    
        accuracy_string = detailed_results["Accuracy"][benchmark_index]
        accuracy.append(float(accuracy_string.split("(")[1].split(",")[0]))
    
    df = pd.DataFrame(
        {
            "Subtask": subtask,
            "Number few shot": number_few_shot,
            "Accuracy": accuracy,
        },
        index=benchmark_name,
    )
    df.index.name = "Benchmark"

    if category == "":
        category = "Uncategorized"
    category = category.replace("_", " ")

    clearml_task.get_logger().report_table("Mosaic Gauntlet", category.title(), table_plot=df)

clearml_task.mark_completed()
