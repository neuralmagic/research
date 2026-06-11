# Utility Pods

Interactive debug pods for ad-hoc work on the cluster. These use `kind: Pod`
(not Job) because they run `sleep infinity` for shell access.

> **Note:** Replace `<YOUR_NAME>` in the YAML files with your identifier
> before applying.

| File | Node type | GPU | Deadline |
|------|-----------|-----|----------|
| `cpu.yml` | Any worker (non-ceph, non-control-plane) | None | 1 hour |
| `h100.yml` | H100 GPU node | 1 GPU | 2 hours |

## Usage

```bash
# Start a CPU debug pod
oc apply -f cpu.yml

# Start an H100 debug pod
oc apply -f h100.yml

# Shell into it
oc rsh mlr-cpu-<YOUR_NAME>
oc rsh mlr-h100-<YOUR_NAME>

# Clean up
oc delete pod mlr-cpu-<YOUR_NAME> -n machine-learning
oc delete pod mlr-h100-<YOUR_NAME> -n machine-learning
```
