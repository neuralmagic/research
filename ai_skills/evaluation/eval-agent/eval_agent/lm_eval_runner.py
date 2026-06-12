"""lm-eval harness command builder.

Builds lm_eval invocations using the harness binary from the dedicated
lm-eval venv (.venvs/lm-eval/bin/lm_eval). Generation params are
received as a dict and converted to the lm-eval gen_kwargs string format.
"""

import shlex
from pathlib import Path
from typing import Optional


def gen_params_to_lm_eval_kwargs(params: dict, seed: int, max_gen_toks: int) -> str:
    """Convert JSON gen params dict to lm-eval --gen_kwargs string.

    Adds do_sample=True when any non-zero sampling param is present.
    Always injects seed and max_gen_toks.
    """
    has_sampling = any(
        k in params and params[k] not in (0, 0.0, False)
        for k in ("temperature", "top_p", "top_k")
    )
    parts = []
    if has_sampling:
        parts.append("do_sample=True")
    for k, v in params.items():
        parts.append(f"{k}={v}")
    parts.append(f"seed={seed}")
    parts.append(f"max_gen_toks={max_gen_toks}")
    return ",".join(parts)


def build_lm_eval_command(
    *,
    task: dict,
    model: str,
    seed: int,
    gen_params: dict,
    effective_max_gen_tokens: int,
    port: int,
    num_concurrent: int,
    timeout: int,
    max_length: int,
    output_path: str,
    lm_eval_bin: str,
    limit: Optional[int] = None,
) -> str:
    """Build a single lm_eval invocation string."""
    base_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    model_args = (
        f"model={model},"
        f"max_length={max_length},"
        f"base_url={base_url},"
        f"num_concurrent={num_concurrent},"
        f"max_retries=3,"
        f"tokenized_requests=False,"
        f"tokenizer_backend=None,"
        f"timeout={timeout}"
    )

    gen_kwargs = gen_params_to_lm_eval_kwargs(gen_params, seed, effective_max_gen_tokens)

    parts = [
        lm_eval_bin,
        "--model", "local-chat-completions",
        "--tasks", task["name"],
        "--model_args", shlex.quote(model_args),
        "--num_fewshot", str(task["num_fewshot"]),
        "--output_path", shlex.quote(output_path),
        "--seed", str(seed),
        "--gen_kwargs", shlex.quote(gen_kwargs),
    ]

    if task.get("apply_chat_template", False):
        parts.append("--apply_chat_template")

    if task.get("fewshot_as_multiturn", False):
        parts.append("--fewshot_as_multiturn")

    if limit is not None:
        parts.extend(["--limit", str(limit)])

    return " ".join(parts)
