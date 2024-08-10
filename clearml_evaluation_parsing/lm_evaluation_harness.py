import json
import argparse
from utils import push_to_clearml as general_push_to_clearml
from utils import open_clearml_task

def process_results(json_file):
    results = json.load(open(json_file))
    artifact = {"name": json_file, "object": results}

    scalars = []
    if "results" in results:
        for task in results["results"]:
            if "configs" in results and task in results["configs"] and "num_fewshot" in results["configs"][task]:
                num_fewshot = results["configs"][task]["num_fewshot"]
            else:
                num_fewshot = None
            for metric in results["results"][task]:
                value = results["results"][task][metric]
                if not isinstance(value, str):
                    if num_fewshot is None:
                        name = task + "/" + metric
                    else:
                        name = task + "/" + f"{num_fewshot:d}" + "shot/" + metric
                    scalars.append({"name": name, "value": value})

    return scalars, artifact

def push_to_clearml(clearml_task, json_file, clearml_model=None):
    scalars, artifact = process_results(json_file)
    general_push_to_clearml(clearml_task, scalars, artifact, clearml_model)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('json_file', type=str, help="Path to json file containing results")
    parser.add_argument('clearml_project', type=str, help='Name of ClearML project')
    parser.add_argument('clearml_task', type=str, nargs='?', help='Optional name of clearml task')

    args = parser.parse_args()

    clearml_task = open_clearml_task(args.clearml_project, args.clearml_task)
    push_to_clearml(clearml_task, args.json_file)
    clearml_task.mark_completed()

if __name__ == "__main__":
    main()