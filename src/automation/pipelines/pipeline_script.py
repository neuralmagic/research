from clearml import PipelineController, Task
from automation.utils import parse_argument
import ast

def main():
    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    if "Args" in args:
        parameters = args["Args"]
    else:
        parameters = {}
    steps = ast.literal_eval(task.get_configuration_object("Steps"))

    version = args["pipeline"]["version"]

    pipeline = PipelineController(
        project=task.get_project_name(),
        name=task.name,
        target_project=task.get_project_name(),
        version=version
    )

    for name, value in parameters.items():
        pipeline.add_parameter(name, default=value)

    for step_args, step_kwargs in steps:
        pipeline.add_step(*step_args, **step_kwargs)

    pipeline.start_locally()

    job_end_callable_name = parse_argument(args["pipeline"]["job end callback"], str)
    if job_end_callable_name is not None:
        filepath = task.artifacts["job end callback"].get_local_copy()
        namespace = {}
        exec(open(filepath, "r").read(), namespace)
        job_end_callable_fn = namespace.get(job_end_callable_name)    
        job_end_callable_fn(task, pipeline)


if __name__ == "__main__":
    main()