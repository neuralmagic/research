import pandas as pd
from clearml import Task
from pathlib import Path
import time
import re


def infer_use_case(task_name_lower):
    use_cases = [
        "Code Completion", "Summarization", "Code fixing", "Docstring Generation",
        "Instruction following", "Chat", "Long RAG", "RAG", "16k", "32k", "64k"
    ]
    for tag in use_cases:
        # Match the use case
        if f"/guidellm/{tag.lower()}/" in task_name_lower:
            return tag
    return None


def infer_model_name(task_name):
    parts = task_name.split("/guidellm/", 1)
    return parts[0] if len(parts) == 2 else "unknown"

def infer_hardware(task_name_lower):
    match = re.search(r"(h\d{3}|a\d{3})(x\d+)?", task_name_lower)
    if match:
        gpu_type = match.group(1)
        count = match.group(2)
        if not count:
            return f"{gpu_type}x8" if gpu_type == "h100" else f"{gpu_type}x1"
        return f"{gpu_type}{count}"
    return "unknown"

def classify_run_type(name):
    name = str(name).lower()
    if "synchronous" in name:
        return "synchronous"
    elif "throughput" in name:
        return "throughput"
    elif "constant@" in name:
        return "rate-sweep"
    return "unknown"

def find_guidellm_csv_path(task):
    artifact = task.artifacts.get("guidellm guidance report")
    if artifact and artifact.hash:
        cache_dir = Path(Path.home(), ".clearml", "cache", "artifacts")
        matches = list(cache_dir.rglob(f"{artifact.hash}*"))
        path = matches[0] if matches else Path(artifact.get_local_copy())
        if path.is_dir():
            path = path / "guidellm-output.csv"
        return path if path.exists() else None
    return None

from pathlib import Path
import pandas as pd
import re


def create_dataframe(final_df):
    MODEL_ALIASES = {
        "nm-testing/l4-scout-int4-debug": "llama4-scout",
        "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16": "llama4-scout",
        "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": "llama4-scout",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4-scout",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "llama4-maverick",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "llama4-maverick",
    }


    def get_model_family(model_name):
        return MODEL_ALIASES.get(model_name, model_name.lower().split("/")[1].split("-")[1])

    def infer_quant_format(model_name):
        model_name = model_name.lower()
        if "int4" in model_name or "w4" in model_name:
            return "INT4"
        elif "fp8" in model_name:
            return "FP8"
        else:
            return "BF16"

    def extract_gpu_count(hw):
        match = re.search(r"x(\d+)", str(hw).lower())
        if match:
            return int(match.group(1))
        return 8 if "h100" in str(hw).lower() else 1

    # Add necessary columns
    final_df["Quant Format"] = final_df["Model"].apply(infer_quant_format)
    final_df["Hardware Class"] = final_df["Hardware"].str.extract(r"(a100|h100)", expand=False).str.upper()
    final_df["GPU Count"] = final_df["Hardware"].apply(extract_gpu_count)
    final_df["Model Family"] = final_df["Model"].apply(get_model_family)
    
    # Filter out 8-GPU runs for all quant formats of llama4-scout
    final_df = final_df[~((final_df["Model Family"] == "llama4-scout") & (final_df["GPU Count"] == 8))]

    final_df["RPS"] = final_df["Successful Requests per second mean"]
    final_df["Tokens/sec"] = final_df["Successful Tokens per second mean"]

    bf16 = final_df[final_df["Quant Format"] == "BF16"].copy()
    bf16["RPS per GPU"] = bf16["RPS"] / bf16["GPU Count"]

    # Select best BF16 based on highest per-GPU RPS
    best_bf16 = (
        bf16.loc[bf16.groupby(["Use Case", "Hardware Class", "Model Family"])["RPS per GPU"].idxmax()]
        [["Use Case", "Hardware Class", "Model Family", "Tokens/sec", "RPS", "GPU Count"]]
        .rename(columns={
            "Tokens/sec": "BF16 Tokens/sec",
            "RPS": "BF16 RPS",
            "GPU Count": "BF16 GPU Count"
        })
    )

    # Merge with original DF
    final_df = final_df.merge(best_bf16, on=["Use Case", "Hardware Class", "Model Family"], how="left")

    # Normalize all runs to BF16 GPU count
    def scale_throughput(row):
        if row["Quant Format"] == "BF16" or pd.isna(row["BF16 RPS"]) or pd.isna(row["BF16 GPU Count"]):
            return row["RPS"], row["Tokens/sec"]
        scale_factor = row["BF16 GPU Count"] / row["GPU Count"] if row["GPU Count"] else 1.0
        return row["RPS"] * scale_factor, row["Tokens/sec"] * scale_factor

    scaled = final_df.apply(
        lambda row: pd.Series(scale_throughput(row), index=["Scaled RPS", "Scaled Tokens/sec"]),
        axis=1
    )
    final_df = pd.concat([final_df, scaled], axis=1)

    # Speedup columns
    final_df["RPS Speedup"] = final_df["Scaled RPS"] / final_df["BF16 RPS"]
    final_df["Tokens/sec Speedup"] = final_df["Scaled Tokens/sec"] / final_df["BF16 Tokens/sec"]

    return final_df


def summarize_speedups_by_hardware_class(df, label):
    summary = (
        df.groupby(["Hardware Class", "Quant Format"])
        .agg(
            Avg_Scaled_RPS=("Scaled RPS", "mean"),
            Avg_BF16_RPS=("BF16 RPS", "mean"),
            Avg_RPS_Speedup=("RPS Speedup", "mean"),
            Num_Use_Cases=("Use Case", "nunique")
        )
        .reset_index()
    )
    summary["Run Type"] = label
    return summary

def process_all_tasks_and_save_per_use_case(project_name, keep_columns):
    all_tasks = Task.get_tasks(
        project_name=project_name,
        task_filter={"status": ["completed"], "order_by": ["-last_update"]},
        allow_archived=False,
    )
    latest_tasks = {}
    for task in all_tasks:
        task_name = task.name
        if "guidellm" not in task_name.lower():
            continue
        existing = latest_tasks.get(task_name)
        if not existing or task._get_last_update() > existing._get_last_update():
            latest_tasks[task_name] = task

    merged_rows = []

    for task_name, task in latest_tasks.items():
        task_name_lower = task_name.lower()

        # Filter out tasks that contain mr_ (e.g., mr_10)
        if "mr_" in task_name_lower:
            continue

        use_case = infer_use_case(task_name_lower)
        if use_case is None:
            continue
        csv_path = find_guidellm_csv_path(task)
        if not csv_path:
            continue
        try:
            df = pd.read_csv(csv_path)
            df["Use Case"] = use_case
            df["Hardware"] = infer_hardware(task_name_lower)
            df["Model"] = infer_model_name(task_name)
            df["Task Name"] = task_name
            df["Run Type"] = df["Name"].apply(classify_run_type)
            selected = df[keep_columns + ["Use Case", "Hardware", "Model", "Task Name", "Run Type"]]
            merged_rows.append(selected)
        except Exception:
            continue

    if not merged_rows:
        return

    final_df = pd.concat(merged_rows, ignore_index=True)
    all_df = create_dataframe(final_df)
    all_df.to_csv("all_usecase_data.csv", index=False)
    throughput_df = create_dataframe(final_df[final_df["Run Type"] == "throughput"].copy())
    sync_df = create_dataframe(final_df[final_df["Run Type"] == "synchronous"].copy())
    sweep_df = create_dataframe(final_df[final_df["Run Type"] == "rate-sweep"].copy())

    throughput_df.to_csv("usecase_throughput_no8gpuforscout.csv", index=False)
    sync_df.to_csv("usecase_synchronous_no8gpuforscout.csv", index=False)
    sweep_df.to_csv("usecase_rate_sweep_no8gpuforscout.csv", index=False)

    # Add summary across use cases per hardware class
    summary_sync = summarize_speedups_by_hardware_class(sync_df, "synchronous")
    summary_throughput = summarize_speedups_by_hardware_class(throughput_df, "throughput")
    summary_df = pd.concat([summary_sync, summary_throughput])
    summary_df.to_csv("summary_speedup_per_hardware_class_long.csv", index=False)

    return {
        "throughput": throughput_df,
        "synchronous": sync_df,
        "rate_sweep": sweep_df,
        "hardware_summary": summary_df,
    }

# Define columns to retain
keep_columns = [
    "Name", "Duration",
    "Successful Requests per second mean",
    "Successful Output tokens per second mean", "Successful Tokens per second mean", "Total Tokens per second mean",
    "Successful Prompt token count mean", "Successful Output token count mean",
    "Successful Request latency mean", "Successful Time to first token ms mean",
    "Successful Time per output token ms mean", "Successful Inter token latency ms mean", "Successful Request concurrency mean",
    "Args"
]

# Execute
dfs = process_all_tasks_and_save_per_use_case("guidellm-test", keep_columns)
