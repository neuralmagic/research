from clearml.automation.parameters import Parameter
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
from clearml.automation import RandomSearch, GridSearch
from clearml import Task
import automation.hpo.callbacks as callbacks
from pyhocon import ConfigFactory

OPTIMIZERS = {
    "GridSearch": GridSearch,
    "RandomSearch": RandomSearch,
    "OptimizerOptuna": OptimizerOptuna,
    "OptimizerBOHB": OptimizerBOHB,
    "Optuna": OptimizerOptuna,
    "BOHB": OptimizerBOHB,
    "hpbandster": OptimizerBOHB,
}

def main():

    task = Task.current_task()

    parameter_dicts = ConfigFactory.parse_string(task.get_configuration_object("Parameters"))
    
    parameters = []
    for parameter_dict in parameter_dicts:
        parameters.append(Parameter.from_dict(parameter_dict))

    optimizer_args = ConfigFactory.parse_string(task.get_configuration_object("Optimization"))

    assert "optimizer" in  optimizer_args.keys()

    optimizer_class = OPTIMIZERS[optimizer_args.pop("optimizer")]

    job_complete_callback = optimizer_args.pop("job_complete_callback")
    if job_complete_callback is not None:
        job_complete_callback = getattr(callbacks, job_complete_callback)
    report_period_min = optimizer_args.pop("report_period_min")
    time_limit_min = optimizer_args.pop("time_limit_min")

    # Define the HPO optimizer
    optimizer = HyperParameterOptimizer(
        hyper_parameters=parameters,
        optimizer_class=optimizer_class,
        execution_queue="services",
        **optimizer_args,
    )

    # Start the optimization
    if report_period_min is not None:
        optimizer.set_report_period(report_period_min)
    
    optimizer.start(job_complete_callback=job_complete_callback)
    
    # wait until optimization completed or timed-out
    # set the time limit for the optimization process (2 hours)
    if time_limit_min is not None:
        optimizer.set_time_limit(in_minutes=time_limit_min)
    
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    
    # optimization is completed, print the top performing experiments id
    print("\n\nOptimization complete\nTop experiments:")
    for experiment in  optimizer.get_top_experiments_details(top_k=3):
        print(experiment)

    # make sure background optimization stopped
    optimizer.stop()

if __name__ == "__main__":
    main()