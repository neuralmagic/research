# Research Automation  

This repository provides a Python interface for creating, managing, and executing ClearML tasks and pipelines using Neural Magic's queueing system. It includes:  

- General-purpose classes for task and pipeline management.  
- Specialized classes for common research workflows, such as:  
  - **llm-compressor** for quantization.
  - **LMEval** for evaluation.
  - **GuideLLM** for benchmarking.

## Repository Structure
```
/ (root)
  │── examples/ # Example scripts
  └── src/
      │
      └── automation/ # Main source code
          │ 
          ├── tasks/ # Base task class and specialized tasks
          │   │
          │   └── scripts/ # Core scripts executed in tasks
          │
          ├── pipelines/ # Base pipeline class
          │
          ├── hpo/ # Base hyperparameter optimization class
          │
          └── standards/ # Standardized tasks & pipelines for research team
              │ 
              └── scripts/ # Extensions of core scripts for standard tasks
```

## Design Principles  

- Use **lightweight wrappers** around ClearML’s existing classes and interfaces.
- Use caution to only introduce specialized depencies in the core scripts, not in class definitions. For instance, LMEvalTask does not depend on lm_eval, onlu lm_eval_script.py introduces that depency.

## Tasks & Scripts

- The classes implemented here allow the separation betweem task creation and execution.
  - This separation is achieved by using **`Task.create()` instead of `Task.init()`**.  
  - This allows task objects to be instantiated, created in the ClearML backend, and manipulated locally in Python scripts or Jupyter notebooks, even if execution happens remotely.  
  - This separation simplifies pipeline construction and prevents outdated task environments by ensuring fresh, up-to-date task creation.

- `Task.create()` points to a script (**core script**) within the repository that will be executed remotely.  
  - **Core scripts** are never executed locally -- only on remote servers.  
  - These scripts access parameters **exclusively** via `task.get_parameters()` or `task.get_parameters_as_dict()` and `task.get_configuration_object` (or `task.get_configuration_object_as_dict`).
- Each task class parses relevant arguments and defines its own `set_parameters()` and `connect_configuration` methods to pass these arguments to the core script.  
  - The `BaseTask` class (which all tasks inherit from) connects the defined parameters automatically.
- `BaseTask` implements two execution methods: `execute_remotely()` and `execute_locally()`.
  - This allows the same script to be deplpyed seamlessly locally or remotely.
  - `execute_locally()` is built on top of `Task.init()`, so it doesn't support separate task creation and execution and must be used with caution.

## Pipelines  

Pipelines are **specialized tasks** that consist of multiple tasks executed in a **Directed Acyclic Graph (DAG)**.  

- The `BasePipeline` class inherits from BaseTask, allowing a user to instantiate and create a pipeline similarly to how they can do it a regular task.
  - `pipeline_script.py` contains the logic that actually creates a **PipelineController** ClearML object.

⚠ **Note:** ClearML introduced `PipelineController.create()` in version 1.17, which is **not currently supported on our servers**.
This means that in ClearML 1.17 or newer `BasePipeline` may wrap the `PipelineController` class directly.
To be investigated when we upgrade ClearML.


## Hyperparameter optimization

ClearML natively supports hyperparameter optimization via specialized tasks.
In the classes implemented here we mimic this logic by defining a `BaseHPO` class that inherits from `BaseTask`.
The script `hpo_script.py` that is executed remotely is responsible for instantiating ClearML's `HyperParameterOptimizer` class, which orchestrates the optimization process.


## Standards  

The `standards/` folder contains **specialized task and pipeline classes** that enforce standardized execution of key research processes.  

- **Example:**  
  - `tasks/LMEvalTask`: General-purpose evaluation with the LMEval harness.  
  - `standards/OpenLLMTask`: A subclass of `LMEvalTask` specifically designed for evaluating the OpenLLM benchmark.  

By using `standards/`, researchers can ensure consistency and best practices across projects.  
