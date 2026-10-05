# SWE-bench Verified with local models

SWE-bench Verified evaluates an agent that edits a repository, followed by a
separate harness that applies the patch and runs the task's tests. It is not a
single-prompt completion benchmark and it is not currently an `eval-agent run`
category. Keep the inference agent, model settings, and test harness fixed
across checkpoints when measuring quantization changes.

## Qwen3.6 starting settings

The [Qwen3.6-35B-A3B model card](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B)
reports 73.4 on SWE-bench Verified using an internal bash/file-edit agent. Its
published SWE-bench settings are temperature `1.0`, `top_p=0.95`, and a 200K
context window. This is a reference point, not a target for mini-SWE-agent:
the agent scaffold and implementation details differ. The model card recommends
vLLM `>=0.19.0` for the BF16 checkpoint.

The [NVFP4 checkpoint card](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)
describes a preliminary checkpoint with both weights and activations quantized,
and says it was tested against vLLM `main`. It specifically recommends the
FlashInfer CUTLASS MoE backend. This comparison therefore measures the released
weight-plus-activation NVFP4 checkpoint against BF16; it does not isolate the
effect of weight quantization. Its model card does not report a SWE-bench
Verified score, so evaluate the exact checkpoint with a matched BF16 run. Pin
and report the vLLM build used for both checkpoints; matching the base model's
minimum version alone does not establish that a given build supports this
NVFP4 checkpoint.

For a text-only repository task, start the BF16 model with the model card's
context and reasoning settings, plus the Qwen tool-call parser used by
mini-SWE-agent:

```bash
vllm serve RedHatAI/Qwen3.6-35B-A3B \
  --served-model-name qwen36 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --max-model-len 200000 \
  --moe-backend flashinfer_cutlass
```

For `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, use the same settings, replace the model
ID, and add the MoE backend required by its model card:

```bash
vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen36 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --max-model-len 200000 \
  --moe-backend flashinfer_cutlass
```

Use a vLLM release supported by the model card. `--language-model-only` avoids
loading or profiling the vision path for text-only SWE-bench tasks. The
`qwen3` reasoning parser and `qwen3_coder` tool-call parser are important here:
mini-SWE-agent sends a Bash tool call through the OpenAI-compatible API. A
plain text-only completion endpoint will not run the agent correctly. Use the
FlashInfer CUTLASS backend for the BF16 control too when supported: on vLLM
0.30.0, automatic backend selection chose TRTLLM for BF16 on our B200 host,
while the NVFP4 card requires CUTLASS. Leaving these defaults unmatched adds
the MoE backend as another variable in the comparison. FlashInfer/CUTLASS may
compile kernels on the first server startup. On hosts where `/tmp` is small,
set `TMPDIR` to a filesystem with enough free space; set `MAX_JOBS` to a
conservative value such as `8` if parallel compilation exhausts scratch space.
The compiled kernels are cached under `~/.cache/flashinfer`, so later starts
should avoid most of this cold-start work.

The model card does not publish a per-action output-token cap or an agent
iteration limit. The example below uses a 16,384-token response cap and
mini-SWE-agent's default 250-step limit as operational starting points. If
logs show `finish_reason=length` before a tool call, raise the response cap and
repeat the same smoke subset for both checkpoints. The Qwen3.6 card lists an
80K output cap for Terminal-Bench, but that is a different agent benchmark and
should not be treated as a validated SWE-bench setting.

## Install the agent and benchmark harness

Create an optional venv in the evaluation workspace. These pinned versions
expose the CLI options used below; if you upgrade them, re-check the commands
and record the new versions.

```bash
uv venv .venvs/swebench --python 3.12
uv pip install --python .venvs/swebench/bin/python \
  "mini-swe-agent==2.4.6" "swebench==5.0.2"

.venvs/swebench/bin/mini-extra swebench --help
.venvs/swebench/bin/swebench eval verified --help
```

SWE-bench task tests run inside containers. Check the daemon's actual storage
location and free space before starting a full run; free space on `$HOME` does
not help if Docker stores its layers on a nearly-full root filesystem.

```bash
docker info --format '{{.DockerRootDir}}'
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

The SWE-bench harness recommends at least 120 GB for its environment-image
cache. Do not prune a shared Docker daemon to make room. On a host where the
Docker root is constrained, a rootless Podman service can provide a separate
Docker-compatible API and store its images under the user's home directory:

```bash
podman info --format '{{.Store.GraphRoot}}'
podman system service --time=0 unix:///home/$USER/podman-swebench.sock
```

Leave that service running in one terminal, and in the evaluation terminal set:

```bash
export DOCKER_HOST="unix:///home/$USER/podman-swebench.sock"
```

Verify the endpoint before using it:

```bash
DOCKER_HOST="$DOCKER_HOST" .venvs/swebench/bin/python -c \
  'import docker; c=docker.from_env(); print(c.ping(), c.version()["ApiVersion"])'
```

If rootless Podman fails while unpacking an image with an error such as
`potentially insufficient UIDs or GIDs` or `lchown: invalid argument`, check
the user's `/etc/subuid` and `/etc/subgid` ranges. Some task images contain
owners outside the allocated range; the preferred fix is to have the host
administrator extend those ranges. The storage option
[`ignore_chown_errors`](https://github.com/containers/storage/blob/main/docs/containers-storage.conf.5.md)
can unpack such images, but it squashes their UIDs/GIDs to one container ID and
removes user separation. If that workaround is necessary, use a dedicated
rootless storage directory and socket, then verify a gold patch for each
affected image before evaluating model patches. On our host, the Matplotlib
Verified image required UID 197609 while the mapping covered only 65,536 IDs;
an isolated store using this option passed the Matplotlib gold-patch test.

The full Verified split has 500 tasks and can consume substantial image-cache
space. A small smoke slice checks both generation and test execution; it is not
a benchmark score and should not be reported as one.

## Configure mini-SWE-agent for vLLM

Create a model overlay such as `qwen36.yaml`:

```yaml
environment:
  pull_timeout: 600
model:
  model_name: hosted_vllm/qwen36
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
    api_key: EMPTY
    temperature: 1.0
    top_p: 0.95
    seed: 42
    max_tokens: 16384
  cost_tracking: ignore_errors
```

The Docker environment's `pull_timeout` defaults to 120 seconds and also bounds
the initial `docker run` call that creates each task container. With a cold
rootless Podman image store, this can raise `TimeoutExpired` before an agent
gets a usable environment. Allow several minutes (600 seconds in this example)
for image startup. When retrying only failed tasks, pass `--redo-existing` with
a filter for those IDs: mini-SWE-agent skips every ID already present in
`preds.json`, including entries whose `model_patch` is empty.

When supplying `-c`, explicitly include mini-SWE-agent's bundled benchmark
config first; a custom `-c` replaces the default config rather than extending
it. Keep its system prompt, tool environment, and default `step_limit` fixed
for each model under comparison.

## Generate and test patches

First run a small fixed slice to verify the agent and container setup.
`--slice 0:10` selects the same first ten test instances for each checkpoint.
For a final result, omit `--slice` and generate predictions for all 500
instances.

Do not assume a contiguous slice is representative: SWE-bench Verified is
ordered by repository, and in our `swebench==5.0.2` run `--slice 0:10`
contained only Astropy tasks. That is useful for a smoke test, but it cannot
estimate performance across the benchmark. For an outcome comparison before a
full run, choose a fixed list of task IDs spanning repositories and pass the
same `--filter` expression for each checkpoint. Record the task IDs with the
results. Do not independently shuffle or sample each checkpoint.

```bash
RUN_DIR=/path/to/runs/qwen36_bf16_smoke
mkdir -p "$RUN_DIR"

.venvs/swebench/bin/mini-extra swebench \
  -c swebench.yaml -c /path/to/qwen36.yaml \
  --subset verified --split test --slice 0:10 \
  --output "$RUN_DIR/inference" --workers 1
```

The command writes `preds.json` and per-instance trajectories under the output
directory. Test the generated patches with unique run IDs for each checkpoint
and prediction set; the harness caches task results by run ID, so reusing an ID
can return stale results after predictions change.

`mini-extra swebench` marking an instance `Submitted` only means it emitted a
submission. The SWE-bench harness must still apply the patch and run the task
tests. Review `error_instances` as well as resolved and unresolved counts:
malformed or missing patches are reported as errors, separate from patches
that apply but fail tests.

```bash
.venvs/swebench/bin/swebench eval verified \
  --predictions "$RUN_DIR/inference/preds.json" \
  --run-id qwen36-bf16-smoke \
  --workers 1 --timeout 1800 \
  --report-dir "$RUN_DIR/evaluation"
```

For the NVFP4 comparison, use the same instance slice, sampling seed,
mini-SWE-agent config, vLLM build, and SWE-bench version; change only the
served checkpoint and use a new run directory and harness `--run-id`. For
smoke slices, pass the generated `instance_id` values with repeated
`--instance` options to
`swebench eval` so the harness only evaluates those predictions. Use the
unfiltered command above only when predictions cover the complete dataset.

Treat a short slice as a setup check, not evidence of a score difference. For a
quantization comparison, evaluate the same task IDs for both checkpoints and
report the paired outcomes as well as each resolved count. Use the full 500-task
split for a benchmark result; if comparing repeated seeds, keep the seeds and
all agent settings matched across checkpoints.

Before model evaluation, validate the harness itself against a gold patch on
one instance:

```bash
.venvs/swebench/bin/swebench eval verified --gold \
  --instance django__django-11099 \
  --run-id swebench-gold-smoke --workers 1
```

Report the resolved count divided by evaluated instances, along with the model
checkpoint, vLLM version and flags, mini-SWE-agent version/config, sampling
parameters, task IDs, Docker/Podman storage backend, and SWE-bench version. Do
not describe a 10-task smoke run as a SWE-bench Verified score.

## Matched Qwen3.6 full-run observation

In an October 2026 full-set comparison, BF16
(`RedHatAI/Qwen3.6-35B-A3B`) resolved 330/500 tasks (66%), while the released
NVFP4 checkpoint (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) resolved 300/500 (60%).
Both runs used the same 500 task IDs, mini-SWE-agent 2.4.6, SWE-bench 5.0.2,
vLLM 0.30.0 with PyTorch 2.13.0+cu132, 200K context, Qwen3 reasoning and
`qwen3_coder` tool parsers, FlashInfer CUTLASS MoE backend, temperature 1.0,
top-p 0.95, seed 42, and a 16,384-token response cap. The NVFP4 checkpoint
quantizes both weights and activations, so this compares the released
checkpoint with BF16 rather than isolating weight-only quantization.

| Checkpoint | Resolved / 500 | Completed | Unresolved | Errors | Ambiguous | Infra failures | Empty patches |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 330 (66%) | 464 | 134 | 36 | 12 | 0 | 0 |
| NVFP4 | 300 (60%) | 450 | 150 | 50 | 12 | 0 | 0 |

On the paired task IDs, both checkpoints resolved 258 tasks; BF16 alone
resolved 72, NVFP4 alone resolved 42, and neither resolved 128. This run
therefore favored BF16 by 30 net resolutions (6 percentage points), though
one sampling seed is not enough to characterize run-to-run variation. The
model-card score uses a different agent scaffold and is not directly
comparable.

Interpret the final JSON report rather than the progress bar: the harness can
print “ran successfully” for every attempted instance even when a generated
patch fails to apply. In these runs there were no likely infrastructure
failures, but both reports included malformed or non-applicable model patches,
12 ambiguous failures, and a 1,800-second timeout on
`scikit-learn__scikit-learn-14710`. The Requests task
`psf__requests-2317` also took about 18 minutes before completing. Keep the
per-instance timeout explicit, and inspect the running test process before
interrupting a slow task; a live pytest process can be consuming CPU or waiting
on a network-dependent test rather than being stuck. Report `resolved`,
`unresolved`, `error`, `ambiguous`, and infrastructure counts separately.
