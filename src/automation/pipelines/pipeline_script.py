from clearml import PipelineController


def main():

    task = Task.current_task()

    args = task.get_parameters_as_dict(cast=True)
    if "Args" in args:
        parameters = args["Args"]
    else:
        parameters = []
    steps = args["Steps"]
    version = args["pipeline"]["version"]

    pipeline = PipelineController(
        project=task.get_project_name(),
        name=task.name,
        target_project=task.get_project_name(),
        version=version
    )

    for parameter_name in parameters:
        parameter_args = parameters[parameter_name].pop("args")
        if parameter_args is not None:
            pipeline.add_parameter(parameter_name, *parameter_args, **parameters[parameter_name])
        else:
            pipeline.add_parameter(parameter_name, **parameters[parameter_name])

    for step_name in steps:
        step_args = steps[step_name].pop("args")
        if step_args is not None:
            pipeline.add_step(step_name, *step_args, **steps[step_name])
        else:
            pipeline.add_step(step_name, **steps[step_name])

    pipeline.start_locally()

    for node in pipeline._monitored_nodes:
        if "metrics" in pipeline._monitored_nodes[node]:
            for title in pipeline._monitored_nodes[node]["metric"]:
                for series, value in pipeline._monitored_nodes[node]["metric"][title].items():
                    task.get_logger().report_scalar(title=title, series=series, iteration=0, value=value)


if __name__ == "__main__":
    main()