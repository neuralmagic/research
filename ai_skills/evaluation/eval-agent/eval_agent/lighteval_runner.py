"""lighteval harness command builder.

Builds lighteval invocations using the harness binary from the dedicated
lighteval venv (.venvs/lighteval/bin/lighteval). Writes a per-(task, seed)
litellm_config.yaml into <run_dir>/configs/ and points lighteval at it.
"""

import shlex
from pathlib import Path
from typing import Optional

import yaml


def build_litellm_config(
    *,
    model: str,
    port: int,
    timeout: int,
    num_concurrent: int,
    gen_params: dict,
    seed: int,
    max_new_tokens: int,
) -> dict:
    """Build the litellm_config.yaml content dict for one (task, seed) run."""
    generation_parameters = {**gen_params, "seed": seed, "max_new_tokens": max_new_tokens}
    return {
        "model_parameters": {
            "provider": "hosted_vllm",
            "model_name": f"hosted_vllm/{model}",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "api_key": "",
            "timeout": timeout,
            "concurrent_requests": num_concurrent,
            "generation_parameters": generation_parameters,
        }
    }


def write_litellm_config(config: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def build_lighteval_command(
    *,
    task: dict,
    config_path: str,
    output_dir: str,
    lighteval_bin: str,
    max_samples: Optional[int] = None,
) -> str:
    """Build a single lighteval invocation string."""
    parts = [
        lighteval_bin,
        "endpoint",
        "litellm",
        config_path,
        shlex.quote(task["task_str"]),
        "--output-dir", output_dir,
        "--save-details",
    ]
    if max_samples is not None:
        parts.extend(["--max-samples", str(max_samples)])
    return " ".join(parts)
