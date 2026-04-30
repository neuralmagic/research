# Q-Fin Long Context Length Benchmark

## How to Reproduce:
```bash
uv sync --python=3.12
canhazgpu run --gpus 1 -- uv run benchmark.py
uv run report.py
```
