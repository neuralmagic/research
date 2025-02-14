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
      ├── automation/ # Main source code
          │ 
          ├── tasks/ # Base task class and specialized tasks
          │   │
          │   └── scripts/ # Core scripts executed in tasks
          │
          │
          ├── pipelines/ # Base pipeline class
          │
          ├── standards/ # Standardized tasks & pipelines for research team
              │ 
              └── scripts/ # Extensions of core scripts for standard tasks
```

### Task Creation & Execution  

To ensure clear separation between task creation and execution, we use:  

- **`Task.create()` instead of `Task.init()`**  
  - `Task.create()` points to a script within the repository that will be executed remotely.  
  - This allows task objects to be instantiated, created, and manipulated locally in Python scripts or Jupyter notebooks, even if execution happens remotely.  
  - This separation simplifies pipeline construction and prevents outdated task environments by ensuring fresh, up-to-date task creation.  

## Tasks & Scripts  

- **Core scripts** are never executed locally—only on remote servers.  
- These scripts access parameters **exclusively** via `task.get_parameters()` or `task.get_parameters_as_dict()`.  
- Each task class defines its own `get_parameters()` method to parse relevant arguments.  
- The `BaseTask` class (which all tasks inherit from) connects the defined parameters automatically.  

## Pipelines  

Pipelines are **specialized tasks** that consist of multiple tasks executed in a **Directed Acyclic Graph (DAG)**.  

- The `BasePipeline` class provides a thin wrapper around **ClearML’s `PipelineController`**, allowing users to define pipeline components before initialization.  
- This approach maintains the same separation of creation and execution as regular tasks.  

⚠ **Note:** ClearML introduced `PipelineController.create()` in version 1.17, which is **not currently supported on our servers**.  
As a workaround, we provide alternative methods that allow separate creation and execution of pipelines but with some limitations:  
- A pipeline must either be **created or executed**—execution cannot occur after creation.  

## Standards  

The `standards/` folder contains **specialized task and pipeline classes** that enforce standardized execution of key research processes.  

- **Example:**  
  - `tasks/LMEvalTask`: General-purpose evaluation with the LMEval harness.  
  - `standards/OpenLLMTask`: A subclass of `LMEvalTask` specifically designed for evaluating the OpenLLM benchmark.  

By using `standards/`, researchers can ensure consistency and best practices across projects.  


## Design Principles  

- Use **lightweight wrappers** around ClearML’s existing classes and interfaces.
- Use caution to only introduce specialized depencies in the core scripts, not in the main code. For instance, LMEvalTask does not depend on lm_eval, onlu lm_eval_script.py introduces that depency.