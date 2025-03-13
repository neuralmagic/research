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
  │── docs/                   # Documentation
  │── examples/               # Example scripts
  └── src/
      │
      └── automation/         # Main source code
          │ 
          ├── tasks/          # Base task class and specialized tasks
          │   └── scripts/    # Core scripts executed in tasks
          │   └── callbacks/  # Callback functions that can be optionally executed within core scripts
          │
          ├── pipelines/      # Base pipeline class
          │
          ├── hpo/            # Base hyperparameter optimization class
          │   └── callbacks/  # Callback functions that can be optionally executed within optimization
          │
          └── standards/      # Config files for standardized tasks & pipelines for research team
```


## Design Principles  

- Use **lightweight wrappers** around ClearML’s existing classes and interfaces.
- Leverage ClearML's `Task.create()` interface to separate task creation and management from task execution.
  - Tasks can be instantiated anywhere but only core scripts are executed in the target environment (remote server, locally).
- Use caution to only introduce specialized depencies in the core scripts, not in class definitions.
For instance, the LMEvalTask class manages evaluation task objects, but it does not depend on the `lm_eval` library.
The underlying script `lm_eval_script.py` introduces that depency and `lm_eval` needs only to be installed in the machine that runs the task.
- Tasks and pipelines can be instantiated via `yaml` config files.
This allows creating standard tasks and pipelines by adding config files to the `standards/` folder.


## Tasks & Core Scripts

- The **`BaseTask`** class offers light wrapping around ClearML's `Task` class.
  - `BaseTask` allows separation betweem task creation and execution.
  - This separation is achieved by using **`Task.create()` instead of `Task.init()`**.  
  - This allows task objects to be instantiated, created in the ClearML backend, and manipulated locally in Python scripts or Jupyter notebooks, even if execution happens remotely.
  - This separation simplifies pipeline construction and prevents outdated task environments by ensuring fresh, up-to-date task creation.
- **Specialized task classes**, such as `LLMCompressorTask`, inherit from `BaseTask`.
Specialized task classes are responsible for:
  - Implementing how arguments are parsed and connected as parameters (`get_paramters()` method) or configurations (`get_configurations()` method) to the underlying ClearML Task.
  - Implementing how to parse an optional `yaml` config file to define arguments.
  - Specifying the core script that will execute in the target hardware.
- **Core scripts** actually implement the execution side of tasks.
  - **Core scripts** are only executed on the target environment (e.g., remote server).
  - These scripts access parameters **exclusively** via `task.get_parameters()` (or `task.get_parameters_as_dict()`) and `task.get_configuration_object` (or `task.get_configuration_object_as_dict`).
- `BaseTask` implements two execution methods: `execute_remotely()` and `execute_locally()`.
  - This allows the same script to be deplpyed seamlessly locally or remotely.
  - `execute_locally()` is built on top of `Task.init()`, so it doesn't support separate task creation and execution and must be used with caution.


## Pipelines  

Pipelines are **specialized tasks** that consist of multiple subtasks executed in a **Directed Acyclic Graph (DAG)**.  

- The **`BasePipeline`** class inherits from `BaseTask`, allowing a user to instantiate and create a pipeline similarly to a regular task.
  - `pipeline_script.py` contains the logic that actually creates a **PipelineController** ClearML object.
- **Specialized pipeline classes**, such as `LLMCompressorLMEvalPipeline`, inherit from `BasePipeline`.
Similarly to tasks, specialized pipelines are responsible for:
  - Implementing how arguments are parsed.
  - Implementing how to parse an optional `yaml` config file to define arguments.
  - Specifying which steps and paramters are part of the pipeline

⚠ **Note:** ClearML introduced `PipelineController.create()` in version 1.17, which is **not currently supported on our servers**.
This means that in ClearML 1.17 or newer `BasePipeline` may wrap the `PipelineController` class directly.
To be investigated when we upgrade ClearML.


## Hyperparameter optimization

ClearML natively supports hyperparameter optimization via specialized tasks.
In the classes implemented here we mimic this logic by defining a **`BaseHPO`** class that inherits from `BaseTask`.
The script `hpo_script.py` that is executed remotely is responsible for instantiating ClearML's `HyperParameterOptimizer` class, which orchestrates the optimization process.


## Standards  

The `standards/` folder contains **`yaml` config files** that control the behvior of specialized tasks or pipelines.
These config files **enforce standardized** execution of key research processes.  

- **Example:**  
  - `tasks/LMEvalTask`: General-purpose evaluation with the LMEval harness.  
  - `standards/openllm.yaml`: Specifies configurations for LMEvalTask to evaluate the OpenLLM benchmark.

By using `standards/`, researchers can ensure consistency and best practices across projects. 


## Docs

Documentation on how to contribute to the repo by constructing new specialized classes or config files.


## Examples

Example scripts on how to use different task classes, pipelines and standards.
