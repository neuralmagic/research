# How to run Hyperparameter Optimization

`ClearML` suppors hyperparameter optimization (HPO) via a special category of tasks.
In this library HPO is implemented in the `BaseHPO` class and its corresponding script.

To run a hyperparameter optimization task, follow these steps:
1. Create a base task (or pipeline). A base task must
  - Expose hyperparameters to be optimized as `Arguments`.
  - Log the metric to be optimized as a `scalar` value.
2. Define the hyperparameter types and ranges using one of these `ClearML` classes:
  - `UniformParameterRange`
  - `DiscreteParameterRange`
  - `UniformIntegerParameterRange`
  - `LogUniformParameterRange`
3. Choose optimizer, one of the following:
  - `GridSearch`
  - `RandomSearch`
  - `OptimizerOptuna`
  - `OptimizerBOHB`
4. Create optimization task, setting a budget for the optimization
5. Execute the task

## Example: Optimizing GPTQ Parameters for Llama-3.2-1B-Instruct

This example optimizes the `dampening_frac` parameter and the number of samples for W4A16 quantization of the Llama-3.2-1B-Instruct model.
The goal is to maximize the OpenLLM score.
The `LLMCompressorLMEvalPipeline` class with the `pipeline_w4a16` configuration is used, which exposes these parameters and computes OpenLLM.

## 1. Create a base task

```python
from automation.pipelines import LLMCompressorLMEvalPipeline
from automation.hpo import BaseHPO
from clearml.automation import UniformParameterRange, LogUniformParameterRange

pipeline = LLMCompressorLMEvalPipeline(
    project_name="alexandre_debug",
    pipeline_name="hpo_example_pipeline",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    execution_queues=["oneshot-a100x1", "oneshot-a100x1"],
    config="pipeline_w4a16",
    lmeval_kwargs={"model_args": "gpu_memory_utilization=0.4,enable_chunked_prefill=True", "batch_size": 10}
)

pipeline.create_pipeline()
```

## 2. Define hyperparameter types and ranges
- `dampening_frac`: will use a uniform range from 0.01 to 0.1 sampled in 0.1 increments.
- `num_sampels`: will uses a log-uniform range from 512 (2^9) to 2048 (2^11).
```python
dampening_frac_range = UniformParameterRange("Args/dampening_frac", min_value=0.01, max_value=0.1, step_size=0.01)
num_samples_range = LogUniformParameterRange("Args/num_samples", min_value=9, max_value=11, "base": 2, "step_size": 1)
```

## 3. Choose optimizer
This example uses `OptimizerOptuna`, which emplys Bayesian models to estimate the value of the objective function in the hyperparameter space.
Bayesian models are akin to curve fitting that represents both a function value and an estimate of uncertainty.
Since Bayesian models require initial data [Optuna](https://optuna.org/) defaults to using `n_startup_trials=10` for random sampling before optimization begins.
However, since many HPO tasks have limited runs, reducing `n_startup_trials` can prevent excessive random sampling.
In this example:
- `n_startup_trials=2`.
- `max_number_of_concurrent_tasks=2` ensures new samples are drawn after initial trials are completed.
- `optuna_sampler="TPESample` sets the Bayesian model to `TPESampler`, the default used in optuna.


## 4. Create optimization task
```python
hpo_task = BaseHPO(
    project_name="alexandre_debug",
    task_name="hpo_example_hpo",
    objective_metric_title="openllm",
    objective_metric_series="average",
    objective_metric_sign="max",
    total_max_jobs=10,
    max_number_of_concurrent_tasks=2,
    optimizer="Optuna",
    optuna_sampler="TPESampler",
    optuna_sampler_kwargs={"n_startup_trials": 2},
    pool_period_min=5,          # Update HPO task every 5min
    report_period_min=5,        # Report results every 5min
    max_iteration_per_job=1,    # Used only in training tasks, use 1 by default
    spawn_project="alexandre_debug",
    base_task_id= pipeline.id,
)

hpo_task.add_paramter(dampening_frac_range)
hpo_task.add_paramter(num_samples_range)
```

**Note:** The HPO task will create several sub-tasks (one pipeline and corresponding tasks per sample). `spawn_project` dictates in which project these sub-tasks live.


## 4. Execute task

```python
hpo_task.execute_remotely()
```

**Note:** The HPO task itself will only manage the optimization process and the execution of the sub-tasks.
Hence, by default it will be submitted to the CPU-only queue named `services`.