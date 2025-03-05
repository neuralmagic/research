import numpy
from clearml import Task

def average(results:dict, metrics:dict):

    task = Task.current_task()
    if len(task.get_models()["input"]) == 1:
        clearml_model_handle = task.get_models()["input"][0]
    else:
        clearml_model_handle = None

    def compute_average(metric_name: str, options: dict):
        if metric_name in results["results"] and "series" in options:
            score = results["results"][metric_name][options["series"]]
            weight = options.get("weight", 1.0)
            normalize = options.get("normalize", False)
        
            if normalize:
                score = (score - options["random_score"]) / (1.0 - options["random_score"])

            return score * weight
        else:
            scores = []
            for _metric_name, _options in options.items():
                scores.append(compute_average(_metric_name, _options))
            average_score = numpy.mean(scores).item()

            task.get_logger().report_single_value(name=metric_name, value=average_score)
            task.get_logger().report_scalar(title=metric_name, series="average", iteration=0, value=average_score)

            if clearml_model_handle is not None:
                clearml_model_handle.report_single_value(name=metric_name, value=average_score)

            results["results"][metric_name] = {"average": average_score}

            return average_score

    for metric_name, options in metrics.items():
        compute_average(metric_name, options)

    return results


    
 