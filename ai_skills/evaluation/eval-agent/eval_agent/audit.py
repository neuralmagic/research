"""Structured audit trail for evaluation runs."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class RunAudit:
    """Manages the audit trail for a single evaluation run."""

    EVENT_TYPES = frozenset({
        "run_started",
        "run_resumed",
        "server_started",
        "server_ready",
        "smoke_passed",
        "eval_started",
        "eval_completed",
        "eval_failed",
        "eval_skipped",
        "server_stopped",
        "run_completed",
        "resume_failed",
    })

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        self.configs_dir = self.run_dir / "configs"
        self._events_path = self.run_dir / "events.jsonl"
        self._commands_path = self.run_dir / "commands.jsonl"
        self._manifest_path = self.run_dir / "manifest.json"

    def init_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)

    def write_manifest(
        self,
        *,
        model: str,
        category: str,
        server_cmd: str,
        base_gen_params: dict,
        port: int,
        num_concurrent: int,
        timeout: int,
        max_length: int,
        smoke_only: bool,
        lm_eval_venv: str,
        lighteval_venv: str,
        registry_snapshot: dict,
    ) -> dict:
        manifest = {
            "run_id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "category": category,
            "server_cmd": server_cmd,
            "base_gen_params": base_gen_params,
            "port": port,
            "num_concurrent": num_concurrent,
            "timeout": timeout,
            "max_length": max_length,
            "smoke_only": smoke_only,
            "lm_eval_venv": lm_eval_venv,
            "lighteval_venv": lighteval_venv,
            "seeds": registry_snapshot.get("seeds", []),
            "benchmarks": [
                {
                    "name": t["name"],
                    "harness": t["harness"],
                    "n_repetitions": t["n_repetitions"],
                    "max_gen_tokens": t["max_gen_tokens"],
                }
                for t in registry_snapshot.get("tasks", [])
            ],
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def load_manifest(self) -> Optional[dict]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return None

    def log_event(self, event_type: str, **data: Any) -> dict:
        if event_type not in self.EVENT_TYPES:
            raise ValueError(
                f"Unknown event type {event_type!r}. "
                f"Valid: {sorted(self.EVENT_TYPES)}"
            )
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - _PROCESS_START, 2),
            "event": event_type,
            **data,
        }
        with open(self._events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def log_command(self, cmd: str, *, description: str = "") -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": cmd,
            "description": description,
        }
        with open(self._commands_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def read_events(self) -> list[dict]:
        if not self._events_path.exists():
            return []
        events = []
        for line_num, line in enumerate(self._events_path.read_text().splitlines(), 1):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    import sys
                    print(f"WARNING: Skipping corrupted event at line {line_num}: {e}", file=sys.stderr)
        return events

    def server_log_path(self) -> Path:
        return self.logs_dir / "vllm_server.log"

    def task_log_path(self, task_name: str, seed: int) -> Path:
        return self.logs_dir / f"{task_name}_seed{seed}.log"

    def task_lm_eval_result_path(self, task_name: str, seed: int) -> Path:
        """JSON output file for lm-eval (--output_path target)."""
        return self.results_dir / f"{task_name}_seed{seed}.json"

    def task_lighteval_result_dir(self, task_name: str, seed: int) -> Path:
        """Output directory for lighteval (--output-dir target)."""
        return self.results_dir / f"{task_name}_seed{seed}"


_PROCESS_START = time.monotonic()
