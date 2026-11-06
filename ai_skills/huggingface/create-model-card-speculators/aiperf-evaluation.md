# Speculators AIPerf evaluation artifacts

Use this reference when model-card inputs come from the AIPerf-based evaluator
in `vllm-project/speculators`. The selected checkout's evaluation guide and
current CLI help are authoritative:

<https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/>

## Collect artifacts

Pass the common parent of the saved evaluation outputs to the collector. The
option may be repeated for separate acceptance and speedup trees:

```bash
python collect_evaluations.py <model-path-or-id> \
  --eval-dir <evaluation-output>
```

The collector recognizes:

- `acceptance.csv`;
- `acceptance_by_prompt_length.csv`;
- `acceptance_length_by_category.csv`;
- `acceptance_rate_by_category.csv`;
- `profile_export_aiperf.json` summaries;
- `eval_command.txt` and `aiperf_command.txt` provenance;
- saved PNG, JPEG, SVG, and WebP plots.

It does not recalculate metrics or declare a run valid. Check the native
summaries for completion, failed requests, and missing speculative metrics
before using any values.

## Acceptance semantics

`acceptance.csv` contains request-derived pooled values. Counters are summed
before division:

- acceptance length = `1 + accepted draft tokens / verification steps`;
- acceptance rate = `accepted draft tokens / proposed draft tokens`;
- position k = fraction of verification steps accepting at least k draft
  tokens.

The category CSVs are native `aiperf speed-bench-report` outputs derived from
each category run's server metrics. Their acceptance rates are fractions from
0 to 1 and their `Overall` column is the arithmetic mean of category values.
That value is not the request-weighted pooled result in `acceptance.csv`.
Preserve both when useful and label them separately.

`acceptance_by_prompt_length.csv` is the source for long-context tables. Keep
overflow and `unknown` bins, disclose errors, and report MRCR settings and the
maximum successful prompt length when available.

## Speedup plots

Use only matched server and workload configurations. Every plot caption must
name the x/y statistics, hardware, tensor parallelism, precision, context,
sampling/output cap, draft length, adaptive-verification state, warmup,
measurement window, grace period, repetitions, and completed concurrency
points.

AIPerf's `output_token_throughput_per_user.avg` is a mean of per-request
reciprocal inter-token latencies. It is not the same as
`1000 / inter_token_latency.avg`. Label whichever statistic the plot actually
uses. The aggregate-throughput y-axis is
`output_token_throughput.avg`.

Do not infer speedup from acceptance length or from an image alone. Compare
only matched measured points against the stated MTP or non-speculative
baseline, and define the numerator, denominator, and operating point for any
reported speedup ratio.

## Source precedence

For the same run, prefer validated AIPerf artifacts over numbers copied into
an existing README. Do not merge measurements from different collectors or
configurations into one table. Keep older GuideLLM results when they are the
only source, and identify that collector explicitly.
