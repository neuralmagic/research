# Q-Fin Long Context Length Benchmark

## How to Reproduce
Update `methodology.md` to reflect your model and configuration. Then run the following:
```bash
uv sync --python=3.12
canhazgpu run --gpus 1 -- uv run benchmark.py
uv run report.py
```
