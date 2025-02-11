from clearml.automation import UniformParameterRange, DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task

task = Task.init(
   project_name='alexandre_debug',
   task_name='HP_optimization_test',
   task_type=Task.TaskTypes.optimizer,
   reuse_last_task_id=False
)

# Define the search space for training hyperparameters
param_ranges = [
    DiscreteParameterRange("Args/num_samples", values=[512, 1024, 2048]), 
    UniformParameterRange("Args/dampening_frac", min_value=0.01, max_value=0.1, step_size=0.01),
]

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

# Define the HPO optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="11d663a67a3a436fa1ff60ea032862e1",  # Replace with your pipeline task ID
    objective_metric_title="performance",  # Metric from the evaluation task
    objective_metric_series="accuracy",  # Metric from the evaluation task
    objective_metric_sign="max",
    hyper_parameters=param_ranges,
    optimizer_class=OptimizerOptuna,
    execution_queue="services",  # Queue for running tasks
    total_max_jobs=5,          # Maximum number of tasks to run
    pool_period_min=0.2,       # Check every X minutes for new tasks
    max_iteration_per_job=1,  # Number of iterations per task
    spawn_project="alexandre-debug/HP optimization",  # Spawn new tasks in this project
)

task.execute_remotely(queue_name='services', exit_process=True)
# Start the optimization
optimizer.set_report_period(0.1)
optimizer.start(job_complete_callback=job_complete_callback)
# wait until optimization completed or timed-out
# set the time limit for the optimization process (2 hours)
optimizer.set_time_limit(in_minutes=10.0)
# wait until process is done (notice we are controlling the optimization process in the background)
optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
optimizer.stop()

print('We are done, good bye')