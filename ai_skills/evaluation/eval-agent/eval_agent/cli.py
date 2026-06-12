"""eval-agent CLI: run | resume | status | cleanup"""

import argparse
import json
import sys
import time
from pathlib import Path

from eval_agent.audit import RunAudit
from eval_agent.runner import BenchmarkRunner, get_tasks_for_category, load_registry
from eval_agent.server import VLLMServer, ServerError
from eval_agent.summary import generate_summary_data


def _write_summary_data(run_dir: Path) -> None:
    print("\nGenerating summary data...", file=sys.stderr)
    summary = generate_summary_data(run_dir)
    out = run_dir / "summary_data.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Summary data: {out}", file=sys.stderr)
    print("Agent should write summary.md from this data.", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    audit = RunAudit(run_dir)

    # Parse gen_params
    try:
        base_gen_params = json.loads(args.gen_params) if args.gen_params else {}
    except json.JSONDecodeError as e:
        print(f"ERROR: --gen-params is not valid JSON: {e}", file=sys.stderr)
        return 1

    registry = load_registry()
    tasks = get_tasks_for_category(args.category)

    # Validate max_length: must fit at least the smallest task + 4096 prompt buffer
    # (long_context is exempt — agent sizes max_length to model's native max)
    if args.category != "long_context":
        min_needed = min(t["max_gen_tokens"] for t in tasks) + 4096
        if args.max_length < min_needed:
            print(
                f"ERROR: --max-length {args.max_length} is too small for {args.category}. "
                f"Minimum needed: {min_needed}",
                file=sys.stderr,
            )
            return 1

    # Validate venv binaries exist
    from pathlib import Path as _P
    lm_eval_bin = _P(args.lm_eval_venv) / "bin" / "lm_eval"
    lighteval_bin = _P(args.lighteval_venv) / "bin" / "lighteval"
    has_lm_eval = any(t["harness"] == "lm-eval" for t in tasks)
    has_lighteval = any(t["harness"] == "lighteval" for t in tasks)
    if has_lm_eval and not lm_eval_bin.exists():
        print(f"ERROR: lm-eval binary not found: {lm_eval_bin}", file=sys.stderr)
        print("Install with: .venvs/lm-eval/bin/pip install lm_eval[api,ifeval,multilingual]", file=sys.stderr)
        return 1
    if has_lighteval and not lighteval_bin.exists():
        print(f"ERROR: lighteval binary not found: {lighteval_bin}", file=sys.stderr)
        print("Install with: .venvs/lighteval/bin/pip install lighteval[extended]", file=sys.stderr)
        return 1

    audit.init_dirs()
    manifest = audit.write_manifest(
        model=args.model,
        category=args.category,
        server_cmd=args.server_cmd,
        base_gen_params=base_gen_params,
        port=args.port,
        num_concurrent=args.num_concurrent,
        timeout=args.timeout,
        max_length=args.max_length,
        smoke_only=args.smoke_only,
        lm_eval_venv=args.lm_eval_venv,
        lighteval_venv=args.lighteval_venv,
        registry_snapshot={"seeds": registry["seeds"], "tasks": tasks},
    )
    audit.log_event("run_started", run_id=manifest["run_id"], model=args.model)

    server = VLLMServer(
        server_cmd=args.server_cmd,
        audit=audit,
        port=args.port,
        health_timeout=args.health_timeout,
    )
    try:
        server.start()
    except ServerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        audit.log_event("run_completed", status="server_failed", error=str(e))
        return 1

    try:
        runner = BenchmarkRunner(
            audit=audit,
            model=args.model,
            category=args.category,
            base_gen_params=base_gen_params,
            port=args.port,
            num_concurrent=args.num_concurrent,
            timeout=args.timeout,
            max_length=args.max_length,
            lm_eval_venv=args.lm_eval_venv,
            lighteval_venv=args.lighteval_venv,
            smoke_only=args.smoke_only,
        )
        results = runner.run_all()
    finally:
        server.stop()

    status = "completed" if not results["failed"] else "completed_with_failures"
    audit.log_event(
        "run_completed",
        status=status,
        completed=len(results["completed"]),
        failed=len(results["failed"]),
        failures=results["failed"],
    )
    print(json.dumps(results, indent=2))
    _write_summary_data(run_dir)
    return 0 if not results["failed"] else 1


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    audit = RunAudit(run_dir)

    manifest = audit.load_manifest()
    if manifest is None:
        print(f"No run found in {run_dir}", file=sys.stderr)
        return 1

    events = audit.read_events()
    print(f"Run ID:   {manifest['run_id']}")
    print(f"Model:    {manifest['model']}")
    print(f"Category: {manifest['category']}")
    print(f"Events:   {len(events)}")

    if events:
        last = events[-1]
        print(f"Last event: {last['event']} at {last['timestamp']}")

    completed = [e for e in events if e["event"] == "eval_completed"]
    failed = [e for e in events if e["event"] == "eval_failed"]
    print(f"Evals completed: {len(completed)}, failed: {len(failed)}")

    if args.follow:
        _follow_events(audit)

    return 0


def _follow_events(audit: RunAudit) -> None:
    seen = len(audit.read_events())
    try:
        while True:
            events = audit.read_events()
            for event in events[seen:]:
                ts = event.get("timestamp", "")
                payload = {k: v for k, v in event.items() if k not in ("timestamp", "elapsed_s", "event")}
                print(f"[{ts}] {event['event']}: {json.dumps(payload)}")
            seen = len(events)
            if any(e["event"] == "run_completed" for e in events):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    audit = RunAudit(run_dir)

    manifest = audit.load_manifest()
    if manifest is None:
        print(f"ERROR: No run found in {run_dir}", file=sys.stderr)
        return 1

    events = audit.read_events()
    completed = [(e["task"], e["seed"]) for e in events if e["event"] == "eval_completed"]
    failed = [(e["task"], e["seed"]) for e in events if e["event"] == "eval_failed"]

    print(f"Resuming run: {manifest['run_id']}")
    print(f"Model: {manifest['model']}, Category: {manifest['category']}")
    print(f"Already completed: {len(completed)} evaluations")
    if completed:
        for task, seed in sorted(completed):
            print(f"  Skipping: {task} seed={seed}")
    print(f"Previously failed: {len(failed)} evaluations (will retry)")

    timeout = args.timeout if args.timeout else manifest["timeout"]
    num_concurrent = args.num_concurrent if args.num_concurrent else manifest["num_concurrent"]

    server = VLLMServer(
        server_cmd=manifest["server_cmd"],
        audit=audit,
        port=manifest["port"],
        health_timeout=args.health_timeout,
    )
    try:
        server.start()
    except ServerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        audit.log_event("resume_failed", error=str(e))
        return 1

    try:
        audit.log_event("run_resumed", run_id=manifest["run_id"])
        runner = BenchmarkRunner(
            audit=audit,
            model=manifest["model"],
            category=manifest["category"],
            base_gen_params=manifest["base_gen_params"],
            port=manifest["port"],
            num_concurrent=num_concurrent,
            timeout=timeout,
            max_length=manifest["max_length"],
            lm_eval_venv=manifest["lm_eval_venv"],
            lighteval_venv=manifest["lighteval_venv"],
            smoke_only=False,
            skip_completed=True,
        )
        results = runner.run_all()
    finally:
        server.stop()

    status = "completed" if not results["failed"] else "completed_with_failures"
    audit.log_event(
        "run_completed",
        status=status,
        completed=len(results["completed"]),
        failed=len(results["failed"]),
        skipped=len(results.get("skipped", [])),
        failures=results["failed"],
    )
    print(json.dumps(results, indent=2))
    _write_summary_data(run_dir)
    return 0 if not results["failed"] else 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    audit = RunAudit(run_dir)
    events = audit.read_events()

    server_started = [e for e in events if e["event"] == "server_started"]
    server_stopped = [e for e in events if e["event"] == "server_stopped"]

    if len(server_stopped) >= len(server_started):
        print("Server already stopped.")
        return 0

    last_start = server_started[-1] if server_started else None
    pid = last_start.get("pid") if last_start else None

    if pid:
        import os
        import signal
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            print(f"Sent SIGTERM to process group {pgid}")
        except (ProcessLookupError, OSError) as e:
            print(f"Process {pid} not found: {e}")

    audit.log_event("server_stopped", pid=pid, reason="manual_cleanup")
    print("Cleanup done. Run `chg status` to verify GPUs are free.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-agent",
        description="Unified LLM evaluation orchestrator (lm-eval + lighteval)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Execute evaluation benchmarks")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--server-cmd", required=True)
    p_run.add_argument("--gen-params", default="{}",
                       help="JSON dict of generation params (no seed, no max tokens)")
    p_run.add_argument("--category", required=True,
                       choices=["instruct", "reasoning", "coding", "long_context"])
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--max-length", type=int, required=True,
                       help="vLLM --max-model-len used in server-cmd")
    p_run.add_argument("--num-concurrent", type=int, default=64,
                       help="Concurrent requests for both harnesses")
    p_run.add_argument("--timeout", type=int, default=1200,
                       help="Per-request timeout in seconds")
    p_run.add_argument("--lm-eval-venv", default=".venvs/lm-eval",
                       help="Path to the lm-eval venv")
    p_run.add_argument("--lighteval-venv", default=".venvs/lighteval",
                       help="Path to the lighteval venv")
    p_run.add_argument("--health-timeout", type=int, default=600)
    p_run.add_argument("--smoke-only", action="store_true",
                       help="Run each task once with 100 samples (first seed only)")
    p_run.add_argument("--run-dir", required=True)

    # --- status ---
    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("--run-dir", required=True)
    p_status.add_argument("--follow", action="store_true")

    # --- resume ---
    p_resume = sub.add_parser("resume", help="Resume an interrupted run")
    p_resume.add_argument("--run-dir", required=True)
    p_resume.add_argument("--health-timeout", type=int, default=600)
    p_resume.add_argument("--timeout", type=int, help="Override timeout from manifest")
    p_resume.add_argument("--num-concurrent", type=int, help="Override concurrency from manifest")

    # --- cleanup ---
    p_cleanup = sub.add_parser("cleanup", help="Kill orphan server processes")
    p_cleanup.add_argument("--run-dir", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {"run": cmd_run, "resume": cmd_resume, "status": cmd_status, "cleanup": cmd_cleanup}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
