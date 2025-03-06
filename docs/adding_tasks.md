# How to Add Specialized Tasks

Below is a template on how to create a class for a specialized task.
This task is responsible for:
- Parsing arguments and registering task parameters and configurations.
  - The core script will potentially be executed in a different environment. It only has access to the information registered as parameter and configurations.
  - Parameters are serialized as text, whereas configurations are serialized as HOCON objects (similar to json or yaml).
    - Configurations offer more flexible and robust serialization. Ideal for arguments that need to be passed to core script and **need not** be overriden by pipelines or hyperparameter optimization.
    - Parameters can be overriden by pipelines and hyperparameter optimization.
- Defining packages needed to execute the task in the target environment.
- Pointing to the core script that will execute the task in the target environment.


## Specialized Task Class

```python
from automation.tasks import BaseTask

class SpecializedTask(BaseTask):

    # Define minimum set of packages needed by this class
    specialized_packages = ["library1", "library2"]

    def __init__(
        self,
        project_name,
        task_name,
        specialized_arg1,
        specialized_arg2,
        specialized_arg3,
        config=None,
        packages=None,
        *args,
        *kwargs,
    ):

    # Parse arguments defined in config file
    config_kwargs = self.process_config(config)

    # Reconcile config kwargs with other arguments
    specialized_arg1 = ...
    specialized_arg2 = ...
    specialized_arg3 = ...

    packages = ... # Make sure to include packages specified by config file, user input and minimum set of packages
   
    # Initialize base class
    super().__init__(
        project_name=project_name,
        task_name=task_name,
        packages=packages,
        *args,
        **kwargs,
    )

    # Handle specific arguments
    self.specialized_args1 = specialized_args1
    self.specialized_args2 = specialized_args2
    # ...

    # Define location for core script (relative to the root of the repo)
    self.script_path = os.path.join(".", "src", "automation", "tasks", "scripts", "specialized_task_script.py")


    # Define script method (used in case of task.execute_locally())
    def script(self):
        from automation.tasks.scripts.specialized_task_script import main
        main()

    
    # Define configurations that control the behavior of the core script
    def get_configurations(self):
        configs = {
            "OddArgs": {
                "arg1": self.specialized_arg1,
                "arg3": self.specialized_arg3,
            },
        }

        return configs


    def get_arguments(self):
        return {
            "Args": {
                "args2": self.specialized_arg2,
            },
        }
```

## Core script
```python
from clearml import Task
from pyhocon import ConfigFactory
# Specialized imports
import library1
import library2

def main():
    # Get current task
    task = Task.current_task()

    # Collect arguments
    parameter_args = task.get_parameters_as_dict(cast=True)
    arg2 = parameter_args["Args"]["arg2"]

    config_args = ConfigFactory.parse_string(task.get_configuration_object("OddArgs"))
    arg1 = config_args["arg1"]
    arg3 = config_args["arg3"]

    # Task stuff goes here
    # ...

    # Don't forget to upload any relevant data (files, results, models)
    task.upload_artifact(name="results", artifact_object=results)
    task.get_logger().report_scalar(title="awesome results", series="accuracy", iteration=0, value=my_awesome_value)

    clearml_model = OutputModel(
        task=task, 
        name=task.name,
        framework="PyTorch", 
        tags=tags,
    )
    clearml_model.update_weights(weights_filename=model_path, auto_delete_file=False)


if __name__ == "__main__":
    main()
```

## Add to src/automation/tasks.__init__.py
```python
from automation.tasks.specialized_task.py import SpecializedTask
```

## How to use it
```python
from automation.tasks import SpecializedTask

task = SpecializedTask(
    project_name="alexandre_debug",
    task_name="specialized_task",
    specialized_arg1=1,
    specialized_arg2="cool"
    specialized_arg3=2,
)

task.execute_remotely("oneshot-a100x1")
```