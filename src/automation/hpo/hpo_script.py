from clearml.automation.parameters import Parameter
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
from clearml.automation import RandomSearch, GridSearch, HyperParameterOptimizer
from clearml import Task
import automation.hpo.callbacks as callbacks
from pyhocon import ConfigFactory
from functools import partial

OPTIMIZERS = {
    "GridSearch": GridSearch,
    "RandomSearch": RandomSearch,
    "OptimizerOptuna": OptimizerOptuna,
    "OptimizerBOHB": OptimizerBOHB,
    "Optuna": OptimizerOptuna,
    "BOHB": OptimizerBOHB,
    "hpbandster": OptimizerBOHB,
}

def main(configurations):

    task = Task.current_task()

    if configurations is None:
        parameter_dicts = ConfigFactory.parse_string(task.get_configuration_object("Parameters"))
        optimizer_args = ConfigFactory.parse_string(task.get_configuration_object("Optimization"))
    else:
        parameter_dicts = configurations.get("Parameters", [])
        optimizer_args = configurations.get("Optimization", {})

    
    parameters = []
    for parameter_dict in parameter_dicts:
        parameters.append(Parameter.from_dict(parameter_dict))

    assert "optimizer" in  optimizer_args.keys()

    optimizer_name = optimizer_args.pop("optimizer")
    optimizer_class = OPTIMIZERS[optimizer_name]

    if "optuna_sampler" in optimizer_args:
        from optuna import samplers
        sampler_name = optimizer_args.pop("optuna_sampler")
        sampler_class = getattr(samplers, sampler_name)
        optuna_sampler_kwargs = optimizer_args.pop("optuna_sampler_kwargs", {})
        sampler = sampler_class(**optuna_sampler_kwargs)
        optimizer_args["optuna_sampler"] = sampler

    if "optuna_pruner" in optimizer_args:
        from optuna import pruners
        pruner_name = optimizer_args.pop("optuna_pruner")
        pruner_class = getattr(pruners, pruner_name)
        optuna_pruner_kwargs = optimizer_args.pop("optuna_pruner_kwargs", {})
        pruner = pruner_class(**optuna_pruner_kwargs)
        optimizer_args["optuna_pruner"] = pruner

    job_complete_callback = optimizer_args.pop("job_complete_callback")
    job_complete_callback_kwargs = optimizer_args.pop("job_complete_callback_kwargs")
    if job_complete_callback is not None:
        job_complete_callback = getattr(callbacks, job_complete_callback)
        if job_complete_callback_kwargs is not None and len(job_complete_callback_kwargs) > 0:
            job_complete_callback = partial(job_complete_callback, **job_complete_callback_kwargs)



    optimization_complete_callback = optimizer_args.pop("optimization_complete_callback")
    optimization_complete_callback_kwargs = optimizer_args.pop("optimization_complete_callback_kwargs")
    if optimization_complete_callback is not None:
        optimization_complete_callback = getattr(callbacks, optimization_complete_callback)
        if optimization_complete_callback_kwargs is not None and len(optimization_complete_callback_kwargs) > 0:
            optimization_complete_callback = partial(optimization_complete_callback, **optimization_complete_callback_kwargs)

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
    for experiment in optimizer.get_top_experiments_details(top_k=3):
        print(experiment)

    if optimization_complete_callback is not None:
        optimization_complete_callback(optimizer.get_top_experiments(top_k=1)[0])

    # make sure background optimization stopped
    optimizer.stop()

if __name__ == "__main__":
    main()