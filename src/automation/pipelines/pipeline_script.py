from clearml import PipelineController, Task
from automation.utils import load_callable_configuration
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

    job_end_callback_fn = load_callable_configuration("job end callback")
    if job_end_callback_fn is not None:
        print("Starting job end callback")

        # Re-open task to make it available for writing, if the callback needs to do so
        # E.g., logging a new scalar
        task = Task.get_task(task_id=task.id)
        task.mark_started()

        # Runs the callback
        job_end_callback_fn(task)

        # Flushes the logger and ends the task
        task.get_logger().flush()
        task.mark_completed()
        
        print("Job end callback completed")


if __name__ == "__main__":
    main()