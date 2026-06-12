"""Unified benchmark runner.

Loads the registry, iterates tasks × seeds, and routes each evaluation
to the appropriate harness (lm-eval or lighteval) via explicit venv binary paths.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from eval_agent.audit import RunAudit
from eval_agent.lm_eval_runner import build_lm_eval_command
from eval_agent.lighteval_runner import (
    build_litellm_config,
    write_litellm_config,
    build_lighteval_command,
)

_REGISTRY_PATH = Path(__file__).parent / "benchmarks" / "registry.yaml"
_SMOKE_SAMPLES = 100


def load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_tasks_for_category(category: str) -> list[dict]:
    registry = load_registry()
    cat = registry.get("categories", {}).get(category)
    if cat is None:
        valid = list(registry.get("categories", {}).keys())
        raise ValueError(f"Unknown category {category!r}. Valid: {valid}")
    return cat["tasks"]


def get_seeds() -> list[int]:
    return load_registry().get("seeds", [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901])


def compute_recommended_max_model_len(category: str) -> int:
    """Return max(task.max_gen_tokens) + 4096 for the given category.

    For long_context the agent must supply the model's native max length
    from config.json — this function is not meaningful there.
    """
    tasks = get_tasks_for_category(category)
    return max(t["max_gen_tokens"] for t in tasks) + 4096


class BenchmarkRunner:
    """Runs the full benchmark suite for a given category."""

    def __init__(
        self,
        *,
        audit: RunAudit,
        model: str,
        category: str,
        base_gen_params: dict,
        port: int = 8000,
        num_concurrent: int = 64,
        timeout: int = 1200,
        max_length: int,
        lm_eval_venv: str,
        lighteval_venv: str,
        smoke_only: bool = False,
        skip_completed: bool = False,
    ):
        self.audit = audit
        self.model = model
        self.category = category
        self.base_gen_params = base_gen_params
        self.port = port
        self.num_concurrent = num_concurrent
        self.timeout = timeout
        self.max_length = max_length
        self.lm_eval_bin = str(Path(lm_eval_venv) / "bin" / "lm_eval")
        self.lighteval_bin = str(Path(lighteval_venv) / "bin" / "lighteval")
        self.smoke_only = smoke_only
        self.skip_completed = skip_completed
        self.tasks = get_tasks_for_category(category)
        self.seeds = get_seeds()

        self.completed_work: set[tuple[str, int]] = set()
        if skip_completed:
            for e in audit.read_events():
                if e["event"] == "eval_completed":
                    self.completed_work.add((e["task"], e["seed"]))

    def run_all(self) -> dict:
        results: dict = {"completed": [], "failed": [], "skipped": []}

        if self.smoke_only:
            return self._run_smoke(results)

        for task in self.tasks:
            task_seeds = self.seeds[: task["n_repetitions"]]
            for seed in task_seeds:
                task_name = task["name"]
                if (task_name, seed) in self.completed_work:
                    results["skipped"].append({"task": task_name, "seed": seed})
                    self.audit.log_event(
                        "eval_skipped", task=task_name, seed=seed, reason="already_completed"
                    )
                    continue

                success = self._run_single(task, seed)
                entry = {"task": task_name, "seed": seed}
                (results["completed"] if success else results["failed"]).append(entry)

        return results

    def _run_smoke(self, results: dict) -> dict:
        """Run each task once (first seed, --max-samples / --limit 100)."""
        seed = self.seeds[0]
        for task in self.tasks:
            success = self._run_single(task, seed, limit=_SMOKE_SAMPLES)
            entry = {"task": task["name"], "seed": seed, "smoke": True}
            (results["completed"] if success else results["failed"]).append(entry)

        if not results["failed"]:
            self.audit.log_event("smoke_passed", tasks=[t["name"] for t in self.tasks])

        return results

    def _effective_max_gen(self, task: dict) -> int:
        """Cap task's max_gen_tokens by the server's max_length minus prompt buffer."""
        return min(task["max_gen_tokens"], self.max_length - 4096)

    def _run_single(self, task: dict, seed: int, limit: Optional[int] = None) -> bool:
        """Dispatch to the correct harness runner. Returns True on success."""
        if task["harness"] == "lm-eval":
            return self._run_lm_eval(task, seed, limit)
        else:
            return self._run_lighteval(task, seed, limit)

    def _run_lm_eval(self, task: dict, seed: int, limit: Optional[int]) -> bool:
        output_path = str(self.audit.task_lm_eval_result_path(task["name"], seed))
        log_path = self.audit.task_log_path(task["name"], seed)
        effective_max_gen = self._effective_max_gen(task)

        cmd = build_lm_eval_command(
            task=task,
            model=self.model,
            seed=seed,
            gen_params=self.base_gen_params,
            effective_max_gen_tokens=effective_max_gen,
            port=self.port,
            num_concurrent=self.num_concurrent,
            timeout=self.timeout,
            max_length=self.max_length,
            output_path=output_path,
            lm_eval_bin=self.lm_eval_bin,
            limit=limit,
        )
        return self._execute(cmd, task["name"], seed, log_path)

    def _clear_lighteval_caches(self) -> None:
        # litellm creates .litellm_cache/ in the CWD between runs; stale entries
        # cause different seeds to receive identical (cached) responses.
        litellm_cache = Path(".litellm_cache")
        if litellm_cache.exists():
            shutil.rmtree(litellm_cache, ignore_errors=True)

        # lighteval caches evaluation results under $HF_HOME/lighteval
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        lighteval_cache = hf_home / "lighteval"
        if lighteval_cache.exists():
            shutil.rmtree(lighteval_cache, ignore_errors=True)

    def _run_lighteval(self, task: dict, seed: int, limit: Optional[int]) -> bool:
        self._clear_lighteval_caches()
        config_path = self.audit.configs_dir / f"litellm_{task['name']}_seed{seed}.yaml"
        output_dir = str(self.audit.task_lighteval_result_dir(task["name"], seed))
        effective_max_gen = self._effective_max_gen(task)
        log_path = self.audit.task_log_path(task["name"], seed)

        config = build_litellm_config(
            model=self.model,
            port=self.port,
            timeout=self.timeout,
            num_concurrent=self.num_concurrent,
            gen_params=self.base_gen_params,
            seed=seed,
            max_new_tokens=effective_max_gen,
        )
        write_litellm_config(config, config_path)

        cmd = build_lighteval_command(
            task=task,
            config_path=str(config_path),
            output_dir=output_dir,
            lighteval_bin=self.lighteval_bin,
            max_samples=limit,
        )
        return self._execute(cmd, task["name"], seed, log_path)

    def _execute(self, cmd: str, task_name: str, seed: int, log_path: Path) -> bool:
        self.audit.log_command(cmd, description=f"{task_name} seed={seed}")
        self.audit.log_event("eval_started", task=task_name, seed=seed)

        with open(log_path, "w") as log_f:
            proc = subprocess.run(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)

        if proc.returncode == 0:
            self.audit.log_event("eval_completed", task=task_name, seed=seed, returncode=0)
            return True
        else:
            self.audit.log_event(
                "eval_failed",
                task=task_name,
                seed=seed,
                returncode=proc.returncode,
                log=str(log_path),
            )
            return False
