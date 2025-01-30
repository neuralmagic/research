from clearml.automation import PipelineController
from clearml import Task
import argparse

parser = argparse.ArgumentParser(description = "Apply recipe in one-shot")

parser.add_argument("--project-name", type=str)
parser.add_argument("--task-prefix", type=str)
parser.add_argument("--pipeline-name", type=str, default="llm-compressor")
parser.add_argument("--llmcompressor-queue", type=str, default="oneshot-a100x1")
parser.add_argument("--model-id", type=str)
parser.add_argument("--recipe", type=str)
parser.add_argument("--disable-clearml-model-save", action="store_true", default=False)
parser.add_argument("--save-dir", type=str, default="output")
parser.add_argument("--dataset", type=str, default="neuralmagic/LLM_compression_calibration")
parser.add_argument("--random-fraction", type=float, default=0.)
parser.add_argument("--shuffle", action="store_true", default=False)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--trust-remote-code", action="store_true", default=False)
parser.add_argument("--tags", type=str, nargs="+", default=None)
parser.add_argument("--max-memory-per-gpu", type=str, default="hessian")

llmcompressor_args, unparsed_args = parser.parse_known_args()
llmcompressor_args = vars(llmcompressor_args)

project_name = llmcompressor_args.pop("project_name")
task_prefix = llmcompressor_args.pop("task_prefix")
pipeline_name = llmcompressor_args.pop("pipeline_name")
llmcompressor_queue = llmcompressor_args.pop("llmcompressor_queue")

Task.force_store_standalone_script()

pipe = PipelineController(
    name=pipeline_name, 
    project=project_name, 
    version="0.0.1",
    target_project=project_name,
)

evaluations = []
evaluation_args = {}
evaluation_counters = {}
for id, entry in enumerate(unparsed_args):
    if entry in ["--lmeval", "--evalplus", "--alpacaeval"]:
        eval_type = entry[2:]
        if evaluation_args != {}:
            evaluations.append(evaluation_args)
            pipe.connect_configuration(evaluation_args, evaluation_args["name"])
        if eval_type in evaluation_counters:
            evaluation_counters[eval_type] += 1
        else:
            evaluation_counters[eval_type] = 0
        name = f"{eval_type}_{evaluation_counters[eval_type]}"
        evaluation_args = {"type": eval_type, "name": name}
    elif entry.startswith("-"):
        if len(unparsed_args) > id+1:
            if unparsed_args[id+1].startswith("-"):
                value = None
            else:
                value = unparsed_args[id+1]
        evaluation_args[entry] = value

if evaluation_args != {}:
    evaluations.append(evaluation_args)
    pipe.connect_configuration(evaluation_args, evaluation_args["name"])

llmcompressor_task_id = Task.get_task(project_name="Automation",task_name="llmcompressor_oneshot", task_filter={'order_by': ["-last_update"]}).id
llmcompressor_step_name = f"{task_prefix}_llmcompressor"

oneshot_override = {f"Args/{k}": v for k, v in llmcompressor_args.items()}
pipe.add_step(
    name=llmcompressor_step_name,
    base_task_id=llmcompressor_task_id,
    execution_queue=llmcompressor_queue,
    parameter_override=oneshot_override,
    monitor_models=["*"],
)

llcompressor_model_id = f"${{{llmcompressor_step_name}.models.output.-1.id}}"

for evaluation_args in evaluations:
    eval_type = evaluation_args.pop("type")
    step_name = evaluation_args.pop("name")
    step_name = f"{llmcompressor_step_name}_{step_name}"

    if "--queue" in evaluation_args:
        queue = evaluation_args.pop("--queue")
    else:
        queue = llmcompressor_queue

    if "--monitor_metrics" in evaluation_args:
        monitor_metrics = evaluation_args.pop("--monitor_metrics")
        monitor_metrics = monitor_metrics.split(":")
        monitor_metrics = [tuple(m.split("/")) for m in monitor_metrics]
    else:
        monitor_metrics = None

    eval_task_id = Task.get_task(
        project_name="Automation",
        task_name=f"{eval_type}_vllm",
        task_filter={'order_by': ["-last_update"]}
    ).id

    eval_override = {
        "Args/model_id": llcompressor_model_id,
        "Args/clearml_model": True,
    }

    for k, v in evaluation_args.items():
        eval_override[f"{eval_type}/{k}"] = v

    print(queue)
    print(step_name)
    print(llmcompressor_step_name)
    print(eval_override)
    print(monitor_metrics)
    print(eval_task_id)

    pipe.add_step(
        name=step_name,
        parents=[llmcompressor_step_name],
        base_task_id=eval_task_id,
        execution_queue=queue,
        parameter_override=eval_override,
        monitor_metrics=monitor_metrics,
    )

pipe.start()
