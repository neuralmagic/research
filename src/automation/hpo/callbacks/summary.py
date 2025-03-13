def job_summary(
    job_id,
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    details = {
        "task id": job_id,
        "objective value": objective_value,
        "parameters": job_parameters,
    }
    print("Job completed:", details)

    if job_id == top_performance_job_id:
        print(f"New best experiment: {job_id}")