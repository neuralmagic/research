"""inspect_ai (Inspect Evals) harness command builder.

Builds `inspect eval` invocations using the harness binary from the dedicated
inspect-ai venv (.venvs/inspect-ai/bin/inspect). Points Inspect's `vllm` model
provider at the vLLM server that eval-agent already started via
`--model-base-url`, so Inspect never spawns a second server.

Registry tasks for this harness use `task_str` (an `inspect_evals/<task_name>`
registry path) instead of a bare benchmark name, and an optional `task_args`
dict whose entries are passed through as `-T key=value` task parameters
(e.g. `{"fewshot": 0}`).
"""

import shlex
from pathlib import Path
from typing import Optional

# Maps the shared JSON gen-params dict (the same dict used for the lm-eval
# and lighteval harnesses) onto native `inspect eval` CLI flags. Keys with no
# direct equivalent are ignored rather than erroring, so the same
# --gen-params value can be reused unmodified across all three harnesses.
_GEN_PARAM_FLAGS = {
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "presence_penalty": "--presence-penalty",
    "frequency_penalty": "--frequency-penalty",
}


def gen_params_to_inspect_flags(params: dict) -> list[str]:
    """Convert the shared JSON gen params dict to `inspect eval` CLI flags."""
    flags: list[str] = []
    for key, value in params.items():
        flag = _GEN_PARAM_FLAGS.get(key)
        if flag is not None:
            flags.extend([flag, str(value)])
    return flags


def build_inspect_ai_command(
    *,
    task: dict,
    model: str,
    seed: int,
    gen_params: dict,
    effective_max_gen_tokens: int,
    port: int,
    num_concurrent: int,
    timeout: int,
    log_dir: str,
    inspect_bin: str,
    limit: Optional[int] = None,
) -> str:
    """Build a single `inspect eval` invocation string.

    Uses Inspect's `vllm` model provider with an explicit `--model-base-url`
    so it talks to the already-running vLLM server rather than starting its
    own. Writes plain JSON eval logs (`--log-format json`) into `log_dir` so
    `summary.py` can parse results the same way it parses lm-eval/lighteval
    JSON output, with no dependency on the inspect-ai venv at summary time.
    """
    base_url = f"http://127.0.0.1:{port}/v1"

    parts = [
        inspect_bin,
        "eval",
        shlex.quote(task["task_str"]),
        "--model", shlex.quote(f"vllm/{model}"),
        "--model-base-url", shlex.quote(base_url),
        "--seed", str(seed),
        "--max-tokens", str(effective_max_gen_tokens),
        "--max-connections", str(num_concurrent),
        "--timeout", str(timeout),
        "--log-dir", shlex.quote(log_dir),
        "--log-format", "json",
    ]

    parts.extend(gen_params_to_inspect_flags(gen_params))

    for arg_name, arg_value in task.get("task_args", {}).items():
        parts.extend(["-T", shlex.quote(f"{arg_name}={arg_value}")])

    if limit is not None:
        parts.extend(["--limit", str(limit)])

    return " ".join(parts)


def find_eval_log_json(result_dir: Path) -> Optional[Path]:
    """Return the single `.json` eval log written into `result_dir`, if any."""
    matches = sorted(result_dir.glob("*.json"))
    return matches[0] if matches else None
