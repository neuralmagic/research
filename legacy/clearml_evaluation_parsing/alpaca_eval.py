if clearml_project is not None and clearml_task is not None:
        from clearml import Task
        task = Task.get_task(project_name=clearml_project, task_name=clearml_task)
        if task is None:
            task = Task.init(project_name=clearml_project, task_name=clearml_task)
        else:
            task.started()

        task.upload_artifact(name='alpaca-eval output', artifact_object=df_leaderboard)
        for name in df_leaderboard:
            value = df_leaderboard[name].values[0]
            if not isinstance(value, str):
                task.get_logger().report_single_value(name=name, value=value)
        task.mark_completed()
