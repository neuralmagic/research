"""Generate summary data JSON from an evaluation run.

Handles both lm-eval output (single JSON file per task+seed) and
lighteval output (directory per task+seed containing JSON files).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_lm_eval_scores(result_file: Path) -> dict[str, float]:
    """Extract numeric metric scores from an lm-eval result JSON."""
    scores: dict[str, float] = {}
    try:
        data = json.loads(result_file.read_text())
        for task_data in data.get("results", {}).values():
            for key, value in task_data.items():
                if "_stderr" not in key and key not in ("name", "alias", "sample_len"):
                    if isinstance(value, (int, float)):
                        scores[key] = value
    except Exception as e:
        print(f"WARNING: Could not parse lm-eval result {result_file}: {e}", file=sys.stderr)
    return scores


def _load_lighteval_scores(result_dir: Path) -> dict[str, float]:
    """Extract numeric metric scores from a lighteval output directory.

    lighteval writes results_*.json files inside its output directory tree.
    We scan recursively for any JSON with a top-level 'results' key.
    """
    scores: dict[str, float] = {}
    for json_file in result_dir.rglob("results_*.json"):
        try:
            data = json.loads(json_file.read_text())
            results = data.get("results", {})
            for task_data in results.values():
                if isinstance(task_data, dict):
                    for key, value in task_data.items():
                        if isinstance(value, (int, float)):
                            scores[key] = value
        except Exception as e:
            print(f"WARNING: Could not parse lighteval result {json_file}: {e}", file=sys.stderr)
    return scores


def generate_summary_data(run_dir: Path) -> dict:
    """Extract all deterministic facts from a run for the agent to summarize."""
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    events_path = run_dir / "events.jsonl"
    events = []
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    results_dir = run_dir / "results"
    if not results_dir.exists():
        return _empty_summary(manifest, events, error="Results directory missing")

    # Build a map of (task_name, seed) -> harness from manifest benchmarks
    harness_map: dict[str, str] = {b["name"]: b.get("harness", "lm-eval") for b in manifest.get("benchmarks", [])}

    # Collect scores per task, averaged across seeds
    scores_by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for completed_event in (e for e in events if e["event"] == "eval_completed"):
        task_name = completed_event["task"]
        seed = completed_event["seed"]
        harness = harness_map.get(task_name, "lm-eval")

        if harness == "lm-eval":
            result_file = results_dir / f"{task_name}_seed{seed}.json"
            scores = _load_lm_eval_scores(result_file) if result_file.exists() else {}
        else:
            result_dir = results_dir / f"{task_name}_seed{seed}"
            scores = _load_lighteval_scores(result_dir) if result_dir.exists() else {}

        for metric, value in scores.items():
            scores_by_task[task_name][metric].append(value)

    # Average scores across seeds
    results: dict[str, dict] = {}
    for task, metrics in scores_by_task.items():
        results[task] = {
            metric: {
                "mean": sum(vals) / len(vals),
                "values": vals,
                "count": len(vals),
            }
            for metric, vals in metrics.items()
        }

    # Timing per (task, seed)
    task_timings: dict[tuple, dict] = {}
    for e in events:
        key = (e.get("task"), e.get("seed"))
        if e["event"] == "eval_started":
            task_timings[key] = {"start": e["elapsed_s"]}
        elif e["event"] == "eval_completed" and key in task_timings:
            task_timings[key]["duration"] = e["elapsed_s"] - task_timings[key]["start"]

    timing_by_task: dict[str, list] = defaultdict(list)
    for (task, seed), times in task_timings.items():
        if task and "duration" in times:
            timing_by_task[task].append({"seed": seed, "duration_seconds": times["duration"]})

    timing_stats: dict[str, dict] = {}
    for task, timings in timing_by_task.items():
        durations = [t["duration_seconds"] for t in timings]
        timing_stats[task] = {
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "per_seed": timings,
            "anomaly": max(durations) > 2 * min(durations) if len(durations) > 1 else False,
        }

    run_start = next((e for e in events if e["event"] == "run_started"), None)
    server_ready = next((e for e in events if e["event"] == "server_ready"), None)
    server_stopped = next((e for e in events if e["event"] == "server_stopped"), None)
    run_complete = next((e for e in events if e["event"] == "run_completed"), None)
    completed = [e for e in events if e["event"] == "eval_completed"]
    failed = [e for e in events if e["event"] == "eval_failed"]

    return {
        "run_id": manifest["run_id"],
        "model": manifest["model"],
        "category": manifest["category"],
        "config": {
            "server_cmd": manifest["server_cmd"],
            "base_gen_params": manifest["base_gen_params"],
            "port": manifest["port"],
            "num_concurrent": manifest["num_concurrent"],
            "timeout": manifest["timeout"],
            "max_length": manifest["max_length"],
            "seeds": manifest.get("seeds", []),
        },
        "timing": {
            "total_seconds": run_complete["elapsed_s"] if run_complete else 0,
            "server_startup_seconds": server_ready["elapsed_s"] if server_ready else 0,
            "by_task": timing_stats,
        },
        "results": results,
        "status": {
            "completed": len(completed),
            "failed": len(failed),
            "failures": [
                {"task": e["task"], "seed": e["seed"], "log": e.get("log")}
                for e in failed
            ],
        },
        "timestamps": {
            "start": run_start["timestamp"] if run_start else None,
            "server_ready": server_ready["timestamp"] if server_ready else None,
            "server_stopped": server_stopped["timestamp"] if server_stopped else None,
            "complete": run_complete["timestamp"] if run_complete else None,
        },
    }


def _empty_summary(manifest: dict, events: list, error: str) -> dict:
    return {
        "run_id": manifest["run_id"],
        "model": manifest["model"],
        "category": manifest["category"],
        "config": {},
        "timing": {},
        "results": {},
        "status": {"completed": 0, "failed": 0, "failures": [], "error": error},
        "timestamps": {},
    }
