from clearml import Task
import json
import pandas

use_cases = [
    {
        "name": "code completion",
        "prefill": 256,
        "decode": 1024,
    },
    {
        "name": "docstring",
        "prefill": 768,
        "decode": 128,
    },
    {
        "name": "code fixing",
        "prefill": 1024,
        "decode": 1024,
    },
    {
        "name": "rag",
        "prefill": 1024,
        "decode": 128,
    },
    {
        "name": "instruction tuning",
        "prefill": 256,
        "decode": 128,
    },
    {
        "name": "multi-turn chat",
        "prefill": 512,
        "decode": 256,
    },
    {
        "name": "large summarization",
        "prefill": 4096,
        "decode": 512,
    },
    {
        "name": "long context RAG",
        "prefill": 10240,
        "decode": 1536,
    },
    {
        "name": "16k",
        "prefill": 16000,
        "decode": 2048,
    },
    {
        "name": "32k",
        "prefill": 32000,
        "decode": 2048,
    },
    {
        "name": "64k",
        "prefill": 64000,
        "decode": 2048,
    },
    {
        "name": "128k",
        "prefill": 128000,
        "decode": 2048,
    },
]

projects = [
    "Benchmarking/0.6.4.post1", 
]

def read_task(task):
    if "guidellm benchmarking output" not in task.artifacts:
        return None

    task_results_path = task.artifacts["guidellm benchmarking output"].get_local_copy()
    if task_results_path is None:
        return None
    if task_results_path.endswith(".txt"):
        return None

    guidellm_args = task.get_parameters_as_dict()["GuideLLM"]
    if "a100x16" in task.name:
        hardware = "a100x16"
    elif "a100x1" in task.name:
        hardware = "a100x1"
    elif "a100x2" in task.name:
        hardware = "a100x2"
    elif "a100x4" in task.name:
        hardware = "a100x4"
    elif "a100x8" in task.name:
        hardware = "a100x8"
    elif "a5000x1" in task.name:
        hardware = "a5000x1"
    elif "a5000x2" in task.name:
        hardware = "a5000x2"
    elif "a5000x4" in task.name:
        hardware = "a5000x4"
    elif "a5000x8" in task.name:
        hardware = "a5000x8"    
    elif "h100x16" in task.name:
        hardware = "h100x16"
    elif "h100x1" in task.name:
        hardware = "h100x1"
    elif "h100x2" in task.name:
        hardware = "h100x2"
    elif "h100x4" in task.name:
        hardware = "h100x4"
    elif "h100x8" in task.name:
        hardware = "h100x8"
    elif "a6000x1" in task.name or "single-a6000" in task.name:
        hardware = "a6000x1"
    elif "a6000x2" in task.name or "double-a6000" in task.name:
        hardware = "a6000x2"
    elif "a6000x4" in task.name or "quad-a6000" in task.name:
        hardware = "a6000x4"
    elif "a6000x8" in task.name or "octo-a6000" in task.name:
        hardware = "a6000x8"
    elif "l40x1" in task.name:
        hardware = "l40x1"

    try:
        task_results = json.load(open(task_results_path))
    except:
        return None

    runs = task_results["benchmarks"][0]["benchmarks"]
    for run in runs:
        del run["results"]
        del run["errors"]
        del run["concurrencies"]
        run.update(guidellm_args)
        run["hardware"] = hardware
        run["project"] = task.get_project_name()
        run["name"] = task.name

    return pandas.DataFrame(runs)
    
def filter_tasks(tasks):
    filtered_tasks = []
    filtered_tasks_names = []
    for task in tasks:
        if task.get_status() != "completed":
            continue
        if task.name + task.get_project_name() not in filtered_tasks_names:
            filtered_tasks.append(task)
            filtered_tasks_names.append(task.name + task.get_project_name())
    
    return filtered_tasks

failures = []
for use_case in use_cases:
    prefill = use_case["prefill"]
    decode = use_case["decode"]
    use_case_name = use_case["name"]
    experiments = []
    print(f"Processing {use_case_name}")
    for project_name in projects:
        identifier = f"{project_name}/{use_case_name}"
        task_name_pattern = f"{prefill}/{decode}"
        tasks = Task.get_tasks(project_name=project_name, task_name=task_name_pattern, task_filter={"status": ["completed"], "order_by": ["-last_update"]}, allow_archived=False)
        if tasks is None or len(tasks) == 0:
            failures.append(project_name + "/" + task_name_pattern)
            print(f"Failed to find {identifier}")
        else:
            tasks = filter_tasks(tasks)
            for task in tasks:
                task_results = read_task(task)
                if task_results is None:
                    failures.append(project_name + "/" + task.name)
                    print(f"Failed to parse results for {task.name}")
                else:
                    experiments.append(task_results)

    if len(experiments) > 0:
        output_file_name = f"{use_case_name}.csv"
        output_file_name = output_file_name.replace("/","__").replace(" ","_")
        experiments = pandas.concat(experiments)
        experiments.to_csv(output_file_name, index=False)
